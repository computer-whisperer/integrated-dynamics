__author__ = 'christian'
import numpy as np
import theano
import theano.tensor as T


class DynamicsComponent:

    def __init__(self, input_components=None):
        self.input_components = []
        self.state_derivatives = {}
        self.state_updates = {}
        self.state = {}
        if input_components is not None:
            for component in input_components:
                self.add_input(component)

    def add_input(self, component):
        self.input_components.append({"component": component})

    def get_input_force_tensor(self, load_state):
        return T.sum([component["component"].get_force_tensor(load_state) for component in self.input_components])

    def get_force_tensor(self, load_state):
        return self.get_input_force_tensor(load_state)

    def build_state_updates(self):
        for component in self.input_components:
            component["component"].build_state_updates()

    def get_state_updates(self):
        updates = []
        derivatives = []
        for component in self.input_components:
            new_updates, new_derivatives = component["component"].get_update_tensors()
            updates.extend(new_updates)
            derivatives.extend(new_derivatives)
        for state in self.state_derivatives:
            derivatives.append([self.state[state], self.state_derivatives[state]])
        for state in self.state_updates:
            updates.append([self.state[state], self.state_updates[state]])
        return updates, derivatives
