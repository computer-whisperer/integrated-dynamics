import math
import numpy as np
import theano
import theano.tensor as T

def rot_tensor(theta):
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

class OneDimensionalLoad:
    """
    Simulates the dynamics of a one-dimensional load provided any number of motive forces.
    """

    def __init__(self, mass):
        self.mass = mass
        self.motive_forces = []
        self.state_tensors = {}
        self.state = {
            "position": theano.shared(0.0),
            "velocity": theano.shared(0.0)
        }
        self.dt = None

    def add_motive_force(self, source, inverted=False):
        direction = -1 if inverted else 1
        self.motive_forces.append({
            "source": source,
            "direction": direction
        })

    def build_functions(self):
        force = self.get_tensors({
            "position": self.state["position"],
            "velocity": self.state["velocity"]
        })["force"]
        acceleration = theano.printing.Print("acceleration")(force/self.mass)
        acceleration_derivative = theano.printing.Print("acceleration_derivative")(theano.gradient.jacobian(acceleration, self.state["velocity"]))
        self.dt = T.dscalar("dt")

        # Taylor series approximation of integral
        # http://www.wolframalpha.com/input/?i=integral+of+e^%28as%29b+ds+from+0+to+t
        coef = 1.0
        integral_approx = acceleration * self.dt
        for c in range(2, 7):
            coef /= c
            integral_approx += coef * T.dot(acceleration_derivative**(c-1), acceleration) * self.dt**c

        special_e = theano.printing.Print("special e")(T.exp(acceleration_derivative*self.dt))
        new_velocity = T.dot(special_e, theano.printing.Print("old_vel")(self.state["velocity"])) + integral_approx

        # Equation for new velocity:
        # http://math.stackexchange.com/questions/1550036/recursive-integral-definition-for-the-dynamics-of-a-dc-motor
        # http://wolfr.am/8BhS0HIW
        #new_velocity = T.dot(T.dot(force, (special_e-1)), T.inv(force_derivative))+self.state["velocity"]

        # Equation for new position:
        # http://www.wolframalpha.com/input/?i=integral+of+%28T%2FD%29%28e^%28D+t%2Fm%29-1%29+%2B+o+dt
        #new_position = T.dot(T.dot(force, (special_e-acceleration_derivative*self.dt)), T.inv(T.dot(acceleration_derivative, acceleration_derivative))) + self.state["velocity"]*self.dt + self.state["position"]
        new_position = self.state["position"] + new_velocity*self.dt

        self.state_tensors = {
            "position": new_position,
            "velocity": new_velocity
        }
        shared_vars = self.get_shared()
        #theano.printing.prettyprint(new_position)
        self.update_state = theano.function([self.dt], [new_position, new_velocity], updates=shared_vars)

    def get_tensors(self, tensors_in):
        self.state_tensors = {
            "position": tensors_in["position"],
            "velocity": tensors_in["velocity"]
        }
        force = 0
        for motive_force in self.motive_forces:
            out = motive_force["source"].get_tensors({
                "ground_velocity": [0, 1] * tensors_in["velocity"] * motive_force["direction"],
                "ground_travel": [0, 1] * (tensors_in["position"] - self.state["position"]) * motive_force["direction"]
            })
            force += out["force"][1]
        return {
            "force": force
        }

    def get_shared(self):
        shared_vals = []
        for motive_force in self.motive_forces:
            shared_vals.extend(motive_force["source"].get_shared())
        shared_vals.append((self.state["position"], self.state_tensors["position"]))
        shared_vals.append((self.state["velocity"], self.state_tensors["velocity"]))
        return shared_vals

class TwoDimensionalLoad(OneDimensionalLoad):
    """
    Simulates the dynamics of a load that can move in two dimensions and rotate in its plane provided any number of motive forces.
    """

    def __init__(self, mass):
        OneDimensionalLoad.__init__(self, mass)
        self.state = {
            "position": theano.shared(np.array([0.0, 0.0, 0.0])),
            "velocity": theano.shared(np.array([0.0, 0.0, 0.0]))
        }

    def add_motive_force(self, source, x_origin=0, y_origin=0, r_origin=0):
        """
        Adds a motive force to the load located at (x_origin, y_origin) away from the cog and rotated r_origin radians
        from forward-facing
        :param source: The force-providing object
        :param x_origin: The x distance, in feet, from the load's center of gravity
        :param y_origin: The y distance, in feet, from the load's center of gravity
        :param r_origin: The angle, in radians, to apply the force at
        """
        self.motive_forces.append({
            "source": source,
            "origin": [x_origin, y_origin, r_origin],
            "distance_to_cog": math.sqrt(x_origin**2 + y_origin**2),
            "angle_to_perpendicular": math.pi-math.atan2(y_origin, x_origin)-r_origin
        })

    def get_tensors(self, tensors_in):
        self.state_functions = {
            "position": theano.function([], tensors_in["position"]),
            "velocity": theano.function([], tensors_in["velocity"])
        }

        # Robot rotation matrix
        world_to_bot_matrix = rot_tensor(-tensors_in["position"][2])

        bot_to_world_matrix = rot_tensor(tensors_in["position"][2])

        delta_pos = tensors_in["position"] - self.state["position"]
        robot_delta_pos = T.dot(delta_pos, world_to_bot_matrix)
        robot_velocity = T.dot(tensors_in["velocity"], world_to_bot_matrix)

        robot_force = [0.0, 0.0, 0.0]
        for motive_force in self.motive_forces:

            # Shift force and force derivative from being wheel-centric to being robot-centric
            #bot_to_origin_rot = np.array([
            #    [math.cos(-motive_force["origin"][2]), math.sin(-motive_force["origin"][2]), 0],
            #    [math.sin(-motive_force["origin"][2]), math.cos(-motive_force["origin"][2]), 0],
            #    [math.sin(-motive_force["angle_to_perpendicular"])/motive_force["distance_to_cog"],
            #     math.cos(-motive_force["angle_to_perpendicular"])/motive_force["distance_to_cog"], 1]
            #])
            bot_to_origin_rot = rot_tensor(-motive_force["origin"][2])
            bot_to_origin_rot += np.array([[0,0,0],[0,0,0],[1,0,0]]) * T.sin(-motive_force["angle_to_perpendicular"])/motive_force["distance_to_cog"]
            bot_to_origin_rot += np.array([[0,0,0],[0,0,0],[0,1,0]]) * T.sin(-motive_force["angle_to_perpendicular"])/motive_force["distance_to_cog"]

            origin_to_bot_rot = rot_tensor(motive_force["origin"][2])
            origin_to_bot_rot += np.array([[0,0,0],[0,0,0],[1,0,0]]) * T.sin(motive_force["angle_to_perpendicular"])/motive_force["distance_to_cog"]
            origin_to_bot_rot += np.array([[0,0,0],[0,0,0],[0,1,0]]) * T.sin(motive_force["angle_to_perpendicular"])/motive_force["distance_to_cog"]

            force = motive_force["source"].get_tensors({
                "ground_velocity": T.dot(robot_velocity, bot_to_origin_rot),
                "ground_travel": T.dot(robot_delta_pos, bot_to_origin_rot),
            })["force"]
            force = T.concatenate((force, np.array([0])))
            robot_force += T.dot(force, origin_to_bot_rot)

        return {
            "force": T.dot(robot_force, bot_to_world_matrix)
        }