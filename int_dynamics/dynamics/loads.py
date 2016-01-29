import math
import numpy as np
import theano
from theano import tensor as T
from .utilities import rot_matrix


class OneDimensionalLoad:
    """
    Simulates the dynamics of a one-dimensional load provided any number of motive forces.
    """

    def __init__(self, components, mass):
        self.mass = theano.shared(mass, theano.config.floatX)
        self.velocity = theano.shared(0.0, theano.config.floatX)
        self.position = theano.shared(0.0, theano.config.floatX)
        self.wheels = []
        for component in components:
            self.add_wheel(component)

    def add_wheel(self, component, inverted=False):
        self.wheels.append({
            "wheel": component,
            "inverted": inverted
        })

    def get_state_derivatives(self):
        state_derivatives = {
            self.position: self.velocity,
            self.velocity: 0
        }
        for wheel in self.wheels:
            caster = -1 if wheel["inverted"] else 1
            wheel["wheel"].velocity = np.array([0, 1]) * self.velocity * caster
            state_derivatives.update(wheel["wheel"].get_state_derivatives(self.mass))
            state_derivatives[self.velocity] += state_derivatives[wheel["wheel"].velocity][1] * caster
        self.local_accel = state_derivatives[self.velocity]
        return state_derivatives


class TwoDimensionalLoad:
    """
    Simulates the dynamics of a load that can move in two dimensions and rotate in its plane provided any number of motive forces.
    """

    def __init__(self, mass):
        self.mass = mass
        self.velocity = theano.shared(np.array([0.0, 0.0, 0.0]), theano.config.floatX)
        self.position = theano.shared(np.array([0.0, 0.0, 0.0]), theano.config.floatX)
        self.wheels = []

    def add_wheel(self, component, x_origin=0, y_origin=0, r_origin=0):
        """
        Adds a motive force to the load located at (x_origin, y_origin) away from the cog and rotated r_origin radians
        from forward-facing
        :param source: The force-providing object
        :param x_origin: The x distance, in feet, from the load's center of gravity
        :param y_origin: The y distance, in feet, from the load's center of gravity
        :param r_origin: The angle, in radians, to apply the force at
        """
        self.wheels.append({
            "wheel": component,
            "origin": [x_origin, y_origin, r_origin],
            "distance_to_cog": math.sqrt(x_origin**2 + y_origin**2),
            "angle_to_perpendicular": math.pi-math.atan2(y_origin, x_origin)-r_origin
        })

    def get_state_derivatives(self):
        state_derivatives = {self.position: self.velocity}

        bot_to_world = rot_matrix(self.position[2])
        world_to_bot = rot_matrix(-self.position[2])

        robot_velocity = T.dot(self.velocity, world_to_bot)
        robot_acceleration = []

        for wheel in self.wheels:

            bot_to_wheel = rot_matrix(-wheel["origin"][2])[:, 0:2]
            bot_to_wheel += np.array([[0,0],[0,0],[1,0]]) * math.sin(-wheel["angle_to_perpendicular"])/wheel["distance_to_cog"]
            bot_to_wheel += np.array([[0,0],[0,0],[0,1]]) * math.cos(-wheel["angle_to_perpendicular"])/wheel["distance_to_cog"]

            wheel_to_bot = rot_matrix(wheel["origin"][2])[0:2]
            wheel_to_bot += np.array([[0,0,1],[0,0,0]]) * math.sin(wheel["angle_to_perpendicular"])/wheel["distance_to_cog"]
            wheel_to_bot += np.array([[0,0,0],[0,0,1]]) * math.cos(wheel["angle_to_perpendicular"])/wheel["distance_to_cog"]

            wheel["wheel"].velocity = T.dot(robot_velocity, bot_to_wheel)
            state_derivatives.update(wheel["wheel"].get_state_derivatives(self.mass))
            robot_acceleration.append(T.dot(state_derivatives[wheel["wheel"].velocity], wheel_to_bot))
        total_acc = T.sum(robot_acceleration, axis=0)
        self.local_accel = total_acc
        state_derivatives[self.velocity] = T.dot(total_acc, bot_to_world)
        return state_derivatives
