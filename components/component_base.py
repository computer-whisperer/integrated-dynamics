__author__ = 'christian'
import numpy as np
import theano
import theano.tensor as T

class DynamicsComponent:

    def __init__(self, input_components=None):
        self.input_components = []
        self.state_tensors = {}
        self.state = {}
        if input_components is not None:
            for component in input_components:
                self.add_input(component)

    def add_input(self, component):
        self.input_components.append({"component": component})

    def get_input_force_tensor(self):
        return T.sum([component["component"].get_force_tensor() for component in self.input_components])

    def get_force_tensor(self):
        return 0

    def build_state_tensors(self, travel, velocity):
        self.build_input_state_tensors(travel, velocity)

    def build_input_state_tensors(self, travel, velocity):
        for component in self.input_components:
            component["component"].build_state_tensors(travel, velocity)

    def get_update_tensors(self):
        tensors = []
        for component in self.input_components:
            tensors.extend(component["component"].get_update_tensors())
        for state in self.state_tensors:
            tensors.append([self.state[state], self.state_tensors[state]])
        return tensors


class LinearerDynamicsComponent(DynamicsComponent):

    def __init__(self, input_components=None, dimensions=1):
        DynamicsComponent.__init__(self, input_components)
        self.dimensions = dimensions
        self.state = {
            "position": theano.shared(np.zeros([dimensions]), theano.config.floatX),
            "velocity": theano.shared(np.zeros([dimensions]), theano.config.floatX)
        }

    def add_input(self, component):
        self.input_components.append({
            "component": component
        })

    def get_input(self, component):
        return component.get_force_tensor()

    def _get_input_force(self):
        total_force = T.sum([T.dot(component["component"].get_force_tensor(), component["force_transform"])
                            for component in self.input_components])
        return total_force

    def get_force_tensor(self):
        return self._get_input_force()

    def build_state_tensors(self, travel, velocity):
        self.state_tensors = {
            "position": self.state["position"] + travel,
            "velocity": velocity
        }
        for state in self.state:
            assert self.state[state].ndim == self.state_tensors[state].ndim
            assert self.state[state].type == self.state_tensors[state].type
        for component in self.input_components:
            component_travel = T.dot(travel, component["travel_transform"])
            component_velocity = T.dot(velocity, component["travel_transform"])
            component["component"].build_state_tensors(component_travel, component_velocity)
