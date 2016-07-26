from .types import *
from .symbolic_types import *


class EulerIntegrator:

    def __init__(self):
        self.current_state = None
        self.default_state = None
        self.pose_tensor = None
        self.motion_tensor = None
        self.root_body = None
        self.time = 0

    def set_root_body(self, root_body):
        self.root_body = root_body

    def build_frames(self, root_body):
        self.set_root_body(root_body)

        default_pose = self.root_body.get_def_pose_vector()
        default_motion = self.root_body.get_def_motion_vector()
        self.default_state = np.concatenate([default_pose, default_motion])
        self.current_state = symbolic_types.VariableNode(self.default_state)
        self.state_explicit_matrix = ExplicitMatrix([self.current_state], (1, self.default_state.shape[0]))
        self.pose_tensor, self.motion_tensor = self.state_explicit_matrix.vsplit(default_pose.shape[0])
        self.root_body.set_variables(self.pose_tensor, self.motion_tensor)
        self.root_body.calculate_frames()

    def build_simulation_tensors(self):
        # Compute the forward dynamics problem with the Composite-Rigid-Body algorithm
        # Get C (the force vector required for zero acceleration) from the RNEA inverse dynamics solver
        crb_C = self.root_body.get_inverse_dynamics(self.motion_tensor*0)[0]
        # Get H by successive calls to RNEA
        columns = []
        for x in range(self.motion_tensor.shape[1]):
            accel_vector = self.motion_tensor*0
            accel_vector.columns[0][x] = 1
            columns.append(self.root_body.get_inverse_dynamics(accel_vector)[0])
        crb_H = ExplicitMatrix.hstack(columns)

        forces = self.root_body.get_total_forces()

        joint_accel = crb_H.solve(forces-crb_C)

        self.dt = VariableNode(0.1)
        new_joint_motion = self.motion_tensor + joint_accel*self.dt
        new_joint_pose = self.root_body.integrate_motion(new_joint_motion, self.dt)
        self.new_state = ExplicitMatrix.vstack([new_joint_pose, new_joint_motion])

    def build_simulation_functions(self):
        self.forward_dynamics_func = build_symbolic_function(self.new_state)

    def step_time(self, dt=0.1):
        self.dt.set_value(dt)
        new_state = self.forward_dynamics_func()
        self.current_state.set_value(new_state)
        self.time += dt

    def get_time(self):
        return self.time

