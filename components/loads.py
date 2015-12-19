import math
import numpy as np
from components import DynamicsComponent
import theano
from theano import tensor as T
from theano.ifelse import ifelse

def rot_matrix(theta):
    if isinstance(theta, float) or isinstance(theta, int):
        sin = math.sin(theta)
        cos = math.cos(theta)
    else:
        sin = T.sin(theta)
        cos = T.cos(theta)
    coss = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])*cos
    sins = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])*sin
    tensor = coss + sins + np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    return tensor

class IntegratingLoad(DynamicsComponent):

    def __init__(self, components, mass):
        self.mass = mass
        super().__init__(components)

    def build_integration(self, dt):
        fake_travel = self.state["velocity"]*dt
        self.build_state_tensors(fake_travel, self.state["velocity"])
        force = self.get_force_tensor()
        acceleration = theano.printing.Print("acceleration")(force/self.mass)
        if acceleration.ndim == 0:
            acceleration_derivative = theano.grad(acceleration, self.state["velocity"])
        else:
            acceleration_derivative = theano.gradient.jacobian(acceleration, self.state["velocity"])

        # There are currently two methods for us to compute the integral. One is with matrix exponentiation, which fails
        # when the acceleration_derivative matrix is singular. The other is a taylor series approximation, which can
        # handle singular matrices, but can sometimes require hundreds of iterations to converge. We configure theano
        # to calculate both, allowing it to optimize away the unused one.

        # Matrix exponential integral
        deriv_inv = T.inv(acceleration_derivative)
        special_e = T.exp(acceleration_derivative*dt)
        e_integral_approximation = theano.printing.Print("E integral!")(T.dot(deriv_inv, special_e - 1))

        # Taylor series integral
        def series_advance(i, last_term, acc_deriv, dt):
            next_term = T.dot(last_term, acc_deriv)*dt/i
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-7))

        if acceleration.ndim == 0:
            init_term = dt
        else:
            init_term = dt*T.identity_like(acceleration_derivative)
        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 500)],
                               non_sequences=[acceleration_derivative, dt],
                               outputs_info=init_term,
                               )
        taylor_integral_approximation = T.sum(terms, axis=0) + init_term

        # Switch between the two integration methods, preferring the exponential method unless acceleration_derivative
        # is singular
        if True:
            integral_approximation = ifelse(T.any(T.isinf(deriv_inv)), taylor_integral_approximation, e_integral_approximation)
        else:
            integral_approximation = taylor_integral_approximation

        new_velocity = self.state["velocity"] + T.dot(acceleration, integral_approximation)
        real_travel = (self.state["velocity"] + new_velocity)*dt/2
        self.build_state_tensors(real_travel, new_velocity)


class OneDimensionalLoad(IntegratingLoad):
    """
    Simulates the dynamics of a one-dimensional load provided any number of motive forces.
    """

    def __init__(self, components, mass):
        super().__init__(components, mass)
        self.state = {
            "velocity": theano.shared(0.0, theano.config.floatX),
            "position": theano.shared(0.0, theano.config.floatX)
        }

    def add_input(self, component, inverted=False):
        self.input_components.append({
            "component": component,
            "inverted": inverted
        })

    def get_force_tensor(self):
        total_force = 0
        for component in self.input_components:
            total_force += component["component"].get_force_tensor()[1] * (-1 if component["inverted"] else 1)
        return total_force

    def build_state_tensors(self, travel, velocity):
        self.state_tensors = {
            "velocity": velocity,
            "position": self.state["position"] + travel
        }
        cast = np.array([0, 1])
        self.build_input_state_tensors(travel*cast, velocity*cast)

    def build_functions(self):
        dt = T.scalar(dtype=theano.config.floatX)
        self.build_integration(dt)
        shared_vars = self.get_update_tensors()
        self.update_state = theano.function([dt], [self.state_tensors["position"], self.state_tensors["velocity"]], updates=shared_vars, profile=False)


class TwoDimensionalLoad(OneDimensionalLoad):
    """
    Simulates the dynamics of a load that can move in two dimensions and rotate in its plane provided any number of motive forces.
    """

    def __init__(self, mass):
        super().__init__(None, mass)
        self.state = {
            "position": theano.shared(np.array([0.0, 0.0, 0.0]), theano.config.floatX),
            "velocity": theano.shared(np.array([0.0, 0.0, 0.0]), theano.config.floatX)
        }

    def add_input(self, component, x_origin=0, y_origin=0, r_origin=0):
        """
        Adds a motive force to the load located at (x_origin, y_origin) away from the cog and rotated r_origin radians
        from forward-facing
        :param source: The force-providing object
        :param x_origin: The x distance, in feet, from the load's center of gravity
        :param y_origin: The y distance, in feet, from the load's center of gravity
        :param r_origin: The angle, in radians, to apply the force at
        """
        self.input_components.append({
            "component": component,
            "origin": [x_origin, y_origin, r_origin],
            "distance_to_cog": math.sqrt(x_origin**2 + y_origin**2),
            "angle_to_perpendicular": math.pi-math.atan2(y_origin, x_origin)-r_origin
        })

    def get_force_tensor(self):
        bot_to_world_matrix = rot_matrix(self.state["position"][2])
        robot_force = self.get_input_force_tensor()
        return T.dot(robot_force, bot_to_world_matrix)

    def get_input_force_tensor(self):
        input_force = [0.0, 0.0, 0.0]
        for component in self.input_components:
            origin_to_bot_rot = rot_matrix(component["origin"][2])
            origin_to_bot_rot += np.array([[0,0,1],[0,0,0],[0,0,0]]) * math.sin(component["angle_to_perpendicular"])/component["distance_to_cog"]
            origin_to_bot_rot += np.array([[0,0,0],[0,0,1],[0,0,0]]) * math.cos(component["angle_to_perpendicular"])/component["distance_to_cog"]

            force = component["component"].get_force_tensor()
            force = T.concatenate((force, np.array([0])))
            input_force += T.dot(force, origin_to_bot_rot)
        return theano.printing.Print("input force")(input_force)

    def build_state_tensors(self, travel, velocity):
        self.state_tensors = {
            "position": self.state["position"]+travel,
            "velocity": velocity
        }
        world_to_robot_rot = rot_matrix(-(self.state["position"][2]+travel[2]/2))
        self.build_input_state_tensors(T.dot(travel, world_to_robot_rot),
                                       T.dot(velocity, world_to_robot_rot))

    def build_input_state_tensors(self, travel, velocity):

        for component in self.input_components:
            bot_to_origin_rot = rot_matrix(-component["origin"][2])
            bot_to_origin_rot += np.array([[0,0,0],[0,0,0],[1,0,0]]) * math.sin(-component["angle_to_perpendicular"])/component["distance_to_cog"]
            bot_to_origin_rot += np.array([[0,0,0],[0,0,0],[0,1,0]]) * math.cos(-component["angle_to_perpendicular"])/component["distance_to_cog"]

            component["component"].build_state_tensors(
                T.dot(travel, bot_to_origin_rot)[:2],
                T.dot(velocity, bot_to_origin_rot)[:2])