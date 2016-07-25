from .types import *
from .symbolic_types import *


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

        default_pose = self.root_body.get_def_pose_vector()
        default_motion = self.root_body.get_def_motion_vector()
        self.default_state = np.concatenate([default_pose, default_motion])
        self.state_tensor = symbolic_types.VariableNode(self.default_state)
        self.state_explicit_matrix = ExplicitMatrix([self.state_tensor], (1, self.default_state.shape[0]))
        pose_tensor, motion_tensor = self.state_explicit_matrix.vsplit(default_pose.shape[0])
        self.root_body.set_variables(pose_tensor, motion_tensor)
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


