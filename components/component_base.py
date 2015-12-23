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
        return self.get_input_force_tensor()

    def build_state_tensors(self, travel, velocity, dt):
        self.build_input_state_tensors(travel, velocity, dt)

    def build_input_state_tensors(self, travel, velocity, dt):
        for component in self.input_components:
            component["component"].build_state_tensors(travel, velocity, dt)

    def get_update_tensors(self):
        tensors = []
        for component in self.input_components:
            tensors.extend(component["component"].get_update_tensors())
        for state in self.state_tensors:
            tensors.append([self.state[state], self.state_tensors[state]])
        return tensors
