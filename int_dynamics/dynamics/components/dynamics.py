import math

import numpy as np
import theano
from theano import tensor as T

from int_dynamics.utilities import rot_matrix


class Dynamic:

    def __init__(self, components):
        if not isinstance(components, list):
            components = [components]
        self.components = components

    def get_state_derivatives(self, load_moment):
        derivatives = {}
        for component in self.components:
            derivatives.update(component.get_state_derivatives(load_moment))
        return derivatives

    def get_variance_sources(self):
        sources = {}
        for component in self.components:
            sources.update(component.get_variance_sources())
        return sources


class GearBox(Dynamic):
    """
    Simulates the dynamics of a gearbox with one or more motors attached
    """

    def __init__(self, motors, gear_ratio=10, dynamic_friction=.5, dynamic_friction_variance=0.05):
        self.friction = theano.shared(dynamic_friction)
        self.friction_variance = dynamic_friction_variance
        self.gear_ratio = gear_ratio
        self.position = theano.shared(0.0, theano.config.floatX)
        self.velocity = theano.shared(0.0, theano.config.floatX)
        super().__init__(motors)

    def get_state_derivatives(self, load_moment):
        for motor in self.components:
            motor.velocity = self.velocity*self.gear_ratio
        state_derivatives = {
            self.position: self.velocity
        }
        force_in = 0
        for motor in self.components:
            state_derivatives.update(motor.get_state_derivatives(load_moment))
            force_in += state_derivatives[motor.velocity]*self.gear_ratio*load_moment
        force_in -= self.friction * T.clip(self.velocity, -1, 1)
        state_derivatives[self.velocity] = force_in/load_moment
        return state_derivatives

    def get_variance_sources(self):
        sources = super().get_variance_sources()
        sources[self.friction] = self.friction_variance
        return sources


class SimpleArm(Dynamic):
    """
    Simulates the dynamics of a simple lever arm attached to a gearbox
    """

    def __init__(self, gearbox, length):
        self.length = length
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)
        self.gearbox = gearbox
        super().__init__(gearbox)

    def get_state_derivatives(self, load_mass):
        circumference = (math.pi * self.length*2)
        self.gearbox.velocity = self.velocity[1]/circumference

        state_derivatives = self.gearbox.get_state_derivatives(load_mass*self.length**2)
        state_derivatives[self.velocity] = np.array([0, 1])*state_derivatives[self.gearbox.velocity]*circumference
        return state_derivatives


class SimpleWheels(Dynamic):
    """
    Simulates the dynamics of a wheel without friction calculations
    """

    def __init__(self, gearbox, diameter):
        self.diameter = diameter/12
        self.gearbox = gearbox
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)
        super().__init__(gearbox)

    def get_state_derivatives(self, load_mass):
        circumference = (math.pi * self.diameter)
        self.gearbox.velocity = self.velocity[1]/circumference

        state_derivatives = self.gearbox.get_state_derivatives(load_mass*(self.diameter/2)**2)
        state_derivatives[self.velocity] = np.array([0, 1])*state_derivatives[self.gearbox.velocity]*circumference
        return state_derivatives


class SolidWheels(Dynamic):
    """
    Simulates the dynamics of a wheel with friction calculations
    """

    def __init__(self, gearbox, count, diameter, static_cof, dynamic_cof, normal_force, friction_error=0.05):
        self.diameter = diameter/12
        self.gearbox = gearbox
        self.mass = .25*count
        self.friction_error = friction_error
        self.total_static_cof = theano.shared(normal_force*static_cof)
        self.total_dynamic_cof = theano.shared(normal_force*dynamic_cof)
        # Ground velocity
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)
        super().__init__(gearbox)

    def get_state_derivatives(self, load_mass):
        # Difference between wheel surface velocity and ground velocity
        slip = theano.shared(0.0, theano.config.floatX)

        circumference = (math.pi * self.diameter)
        self.gearbox.velocity = (self.velocity[1] + slip)/circumference
        state_derivatives = self.gearbox.get_state_derivatives(load_mass*(self.diameter/2)**2)

        force_in = state_derivatives[self.gearbox.velocity]*load_mass*circumference

        force_out = np.array([0, 1]) * T.clip(force_in, -self.total_static_cof, self.total_static_cof) + \
                    T.clip(slip, -self.total_dynamic_cof, self.total_dynamic_cof)
        force_out += np.array([1, 0]) * T.clip(-self.velocity[0], -self.total_dynamic_cof, self.total_dynamic_cof)
        state_derivatives[slip] = (force_in - force_out[1])/self.mass
        state_derivatives[self.velocity] = force_out/self.mass
        return state_derivatives

    def get_variance_sources(self):
        sources = super().get_variance_sources()
        sources[self.total_dynamic_cof] = self.friction_error
        sources[self.total_static_cof] = self.friction_error
        return sources


class KOPWheels(SolidWheels):
    def __init__(self, gearbox, diameter, count, normal_force):
        SolidWheels.__init__(self, gearbox, diameter, count, 1.07, .9, normal_force)


class Gravity(Dynamic):
    """
    Simulates the dynamics of Gravity
    """

    def __init__(self):
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)
        super().__init__([])

    def get_state_derivatives(self, load_mass):
        return {self.velocity: np.array([0, -32])}


class OneDimensionalLoad:
    """
    Simulates the dynamics of a one-dimensional load provided any number of motive forces.
    """

    def __init__(self, components, weight):
        self.mass = theano.shared(weight/32, theano.config.floatX)
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

    def get_state(self):
        return {
            "position": self.position.get_value(),
            "velocity": self.velocity.get_value()
        }

    def get_variance_sources(self):
        sources = {}
        for component in self.wheels:
            sources.update(component["wheel"].get_variance_sources())
        return sources


class TwoDimensionalLoad:
    """
    Simulates the dynamics of a load that can move in two dimensions and rotate in its plane provided any number of motive forces.
    """

    def __init__(self, weight):
        self.mass = theano.shared(weight/32, theano.config.floatX)
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
        total_acc = sum(robot_acceleration)
        self.local_accel = total_acc
        state_derivatives[self.velocity] = T.dot(total_acc, bot_to_world)
        return state_derivatives

    def get_state(self):
        return {
            "position": self.position.get_value(),
            "velocity": self.velocity.get_value()
        }

    def get_variance_sources(self):
        sources = {}
        for component in self.wheels:
            sources.update(component["wheel"].get_variance_sources())
        return sources