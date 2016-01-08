import math
import numpy as np
import theano
import theano.tensor as T


class SimpleWheels:
    """
    Simulates the dynamics of a wheel without friction calculations
    """

    def __init__(self, gearbox, diameter):
        self.diameter = diameter/12
        self.gearbox = gearbox
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)

    def get_state_derivatives(self, load_mass):
        circumference = (math.pi * self.diameter)
        self.gearbox.velocity = self.velocity[1]/circumference

        state_derivatives = self.gearbox.get_state_derivatives(load_mass*(self.diameter/2)**2)
        state_derivatives[self.velocity] = np.array([0, 1])*state_derivatives[self.gearbox.velocity]*circumference
        return state_derivatives


class SolidWheels:
    """
    Simulates the dynamics of a wheel with friction calculations
    """

    def __init__(self, gearbox, count, diameter, static_cof, dynamic_cof, normal_force):
        self.diameter = diameter/12
        self.gearbox = gearbox
        self.mass = .25*count
        self.total_static_cof = normal_force*static_cof
        self.total_dynamic_cof = normal_force*dynamic_cof
        # Ground velocity
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)

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


class KOPWheels(SolidWheels):
    def __init__(self, gearbox, diameter, count, normal_force):
        SolidWheels.__init__(self, gearbox, diameter, count, 1.07, .9, normal_force)
