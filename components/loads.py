import math
import numpy as np
import theano
from theano import tensor as T
from components import DynamicsComponent
from utilities import rot_matrix, integrate_via_ode


class OneDimensionalLoad(DynamicsComponent):
    """
    Simulates the dynamics of a one-dimensional load provided any number of motive forces.
    """

    def __init__(self, components, mass):
        self.mass = mass
        super().__init__(components)
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

    def build_state_tensors(self, travel, velocity, dt):

        # Get the current force out in terms of velocity
        self.state_tensors = {
            "velocity": velocity,
            "position": self.state["position"] + travel
        }
        self.build_input_state_tensors(travel, velocity, dt)
        force = self.get_force_tensor()

        # Integrate it to find new_velocity and travel
        acceleration = force/self.mass
        new_velocity = integrate_via_ode(acceleration, velocity, dt, self.state["velocity"])
        new_travel = (self.state["velocity"] + new_velocity)*dt/2

        # Recalculate state tensors
        self.state_tensors = {
            "velocity": new_velocity,
            "position": self.state["position"] + new_travel
        }
        self.build_input_state_tensors(new_travel, new_velocity, dt)

    def build_input_state_tensors(self, travel, velocity, dt):
        cast = np.array([0, 1])
        for component in self.input_components:
            component["component"].build_state_tensors(travel*cast, velocity*cast, dt)

    def build_functions(self):
        dt = T.scalar(dtype=theano.config.floatX)
        self.build_state_tensors(self.state["velocity"]*dt, self.state["velocity"], dt)
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
        return input_force

    def build_input_state_tensors(self, travel, velocity, dt):
        world_to_robot_rot = rot_matrix(-self.state_tensors["position"][2] + travel[2] / 2)
        bot_travel = T.dot(travel, world_to_robot_rot)
        bot_velocity = T.dot(velocity, world_to_robot_rot)

        for component in self.input_components:
            bot_to_origin_rot = rot_matrix(-component["origin"][2])
            bot_to_origin_rot += np.array([[0,0,0],[0,0,0],[1,0,0]]) * math.sin(-component["angle_to_perpendicular"])/component["distance_to_cog"]
            bot_to_origin_rot += np.array([[0,0,0],[0,0,0],[0,1,0]]) * math.cos(-component["angle_to_perpendicular"])/component["distance_to_cog"]

            component["component"].build_state_tensors(
                T.dot(bot_travel, bot_to_origin_rot)[:2],
                T.dot(bot_velocity, bot_to_origin_rot)[:2],
                dt)

    def build_integration(self, dt):
        fake_travel = self.state["velocity"]*dt
        self.build_state_tensors(fake_travel, self.state["velocity"])
        force = self.get_force_tensor()
        acceleration = force/self.mass

        acceleration_derivative = theano.gradient.jacobian(acceleration, self.state["velocity"])

        # Taylor series integral
        def series_advance(i, last_term, acc_deriv, dt):
            next_term = T.dot(acc_deriv, last_term)*dt/i
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-7))

        init_term = dt*T.identity_like(acceleration_derivative)

        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 500)],
                               non_sequences=[acceleration_derivative, dt],
                               outputs_info=init_term,
                               )
        new_deltav = T.dot(T.sum(terms, axis=0) + init_term, acceleration)

        new_velocity = self.state["velocity"] + new_deltav
        real_travel = (self.state["velocity"] + new_velocity)*dt/2
        self.build_state_tensors(real_travel, new_velocity)