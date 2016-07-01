import numpy as np
import itertools

import theano
import theano.tensor as T


class IntegratorBase:
    state_tensor = None

    def __init__(self):
        self.state_tensor = None
        self.default_state = None
        self.root_body = None
        self.bodies = []
        self.body_combinations = []
        self.body_positions = []
        self.body_rotations = []
        self.body_distances = []

    def set_root_body(self, root_body):
        self.root_body = root_body
        self.bodies = [self.root_body]
        self.bodies.extend(self.root_body.get_all_children())

    def build_simulation_tensors(self, root_body):
        self.set_root_body(root_body)

        self.default_state = self.root_body.get_def_variables()
        self.state_tensor = theano.shared(self.default_state, theano.config.floatX)
        self.root_body.set_variables(self.state_tensor)
        self.root_body.calculate_frames()
        #self.root_body.build_inertia()
        # self.body_combinations = itertools.combinations(range(len(self.bodies)))

    def build_simulation_functions(self):
        raise NotImplementedError()

    def step_time(self):
        raise NotImplementedError()

    def get_time(self):
        raise NotImplementedError()


class EulerIntegrator(IntegratorBase):
    state_tensor = None


