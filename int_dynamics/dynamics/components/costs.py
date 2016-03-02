import theano
import theano.tensor as T
import numpy as np


class Cost:

    def get_cost(self):
        pass


class TwoDimensionalGaussianPositionalCost(Cost):

    def __init__(self, load):
        self.positions = theano.shared(np.array([[0.0, 0.0, 0.0]]), theano.config.floatX)
        self.values = theano.shared(np.array([0.0]), theano.config.floatX)
        self.deviations = theano.shared(np.array([0.0]), theano.config.floatX)
        self.load = load

    def get_cost(self):
        delta_position = self.positions - self.load.position.dimshuffle(0, 'x')
        goal_distance = T.sum(delta_position**2, axis=1)
        cost = self.values*T.exp(-goal_distance/(2*self.deviations)**2)
        return T.sum(cost)
