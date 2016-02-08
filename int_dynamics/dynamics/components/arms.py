import math
import numpy as np
import theano
import theano.tensor as T


class SimpleArm:
    """
    Simulates the dynamics of a simple lever arm attached to a gearbox
    """

    def __init__(self, gearbox, length):
        self.length = length
        self.gearbox = gearbox
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)

    def get_state_derivatives(self, load_mass):
        circumference = (math.pi * self.length*2)
        self.gearbox.velocity = self.velocity[1]/circumference

        state_derivatives = self.gearbox.get_state_derivatives(load_mass*self.length**2)
        state_derivatives[self.velocity] = np.array([0, 1])*state_derivatives[self.gearbox.velocity]*circumference
        return state_derivatives
