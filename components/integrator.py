__author__ = 'christian'
import theano
import theano.tensor as T

class ComponentIntegrator:

    def __init__(self, component, mass):
        self.component = component
        self.mass = mass
        self.dt = T.iscalar()


