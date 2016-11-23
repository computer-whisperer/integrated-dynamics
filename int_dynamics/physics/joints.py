from .utilities import list_to_vector
import sympy
from sympy.physics import vector


class Joint:

    state_symbols = None
    state_defaults = None
    state_symbol_derivatives = None
    parent_body = None
    child_body = None

    base_frame = None
    base_point = None
    end_frame = None
    end_point = None

    @classmethod
    def free_joint(cls, name=None,
                 joint_base_lin=None, joint_base_ang=None,
                 joint_pose_lin=None, joint_pose_ang=None,
                 joint_motion_lin=None, joint_motion_ang=None,
                 body_pose_lin=None, body_pose_ang=None):
        return cls(
            name,
            joint_base_lin, joint_base_ang,
            joint_pose_lin, joint_pose_ang,
            joint_motion_lin, joint_motion_ang,
            body_pose_lin, body_pose_ang,
            "joint_pose_ang_w joint_pose_ang_x joint_pose_ang_y joint_pose_ang_z "
            "joint_pose_lin_x joint_pose_lin_y joint_pose_lin_z "
            "joint_motion_ang_x joint_motion_ang_y joint_motion_ang_z "
            "joint_motion_lin_x joint_motion_lin_y joint_motion_lin_z"
        )


    @classmethod
    def trans_joint(cls, name=None,
                 joint_base_lin=None, joint_base_ang=None,
                 joint_pose_lin=None, joint_pose_ang=None,
                 joint_motion_lin=None, joint_motion_ang=None,
                 body_pose_lin=None, body_pose_ang=None):
        return cls(
            name,
            joint_base_lin, joint_base_ang,
            joint_pose_lin, joint_pose_ang,
            joint_motion_lin, joint_motion_ang,
            body_pose_lin, body_pose_ang,
            "joint_pose_lin_x joint_pose_lin_y joint_pose_lin_z "
            "joint_motion_lin_x joint_motion_lin_y joint_motion_lin_z"
        )

    @classmethod
    def ball_joint(cls, name=None,
                 joint_base_lin=None, joint_base_ang=None,
                 joint_pose_lin=None, joint_pose_ang=None,
                 joint_motion_lin=None, joint_motion_ang=None,
                 body_pose_lin=None, body_pose_ang=None):
        return cls(
            name,
            joint_base_lin, joint_base_ang,
            joint_pose_lin, joint_pose_ang,
            joint_motion_lin, joint_motion_ang,
            body_pose_lin, body_pose_ang,
            "joint_pose_ang_w joint_pose_ang_x joint_pose_ang_y joint_pose_ang_z "
            "joint_motion_ang_x joint_motion_ang_y joint_motion_ang_z"
        )

    @classmethod
    def elbow_joint(cls, name=None,
                 joint_base_lin=None, joint_base_ang=None,
                 joint_pose_lin=None, joint_pose_ang=None,
                 joint_motion_lin=None, joint_motion_ang=None,
                 body_pose_lin=None, body_pose_ang=None, axis="x"):
        return cls(
            name,
            joint_base_lin, joint_base_ang,
            joint_pose_lin, joint_pose_ang,
            joint_motion_lin, joint_motion_ang,
            body_pose_lin, body_pose_ang,
            "joint_pose_ang_w joint_pose_ang_{axis} joint_motion_ang_{axis}".format(axis=axis)
        )

    @classmethod
    def fixed_joint(cls, name=None,
                 joint_base_lin=None, joint_base_ang=None,
                 joint_pose_lin=None, joint_pose_ang=None,
                 joint_motion_lin=None, joint_motion_ang=None,
                 body_pose_lin=None, body_pose_ang=None):
        return cls(
            name,
            joint_base_lin, joint_base_ang,
            joint_pose_lin, joint_pose_ang,
            joint_motion_lin, joint_motion_ang,
            body_pose_lin, body_pose_ang
        )

    def __init__(self, name=None,
                 joint_base_lin=None, joint_base_ang=None,
                 joint_pose_lin=None, joint_pose_ang=None,
                 joint_motion_lin=None, joint_motion_ang=None,
                 body_pose_lin=None, body_pose_ang=None,
                 symbolic_axes=None):
        self.name = name
        self.joint_base_lin = joint_base_lin or [0, 0, 0]
        self.joint_base_ang = joint_base_ang or [1, 0, 0, 0]

        self.joint_pose_lin = joint_pose_lin or [0, 0, 0]
        self.joint_pose_ang = joint_pose_ang or [1, 0, 0, 0]
        self.joint_motion_lin = joint_motion_lin or [0, 0, 0]
        self.joint_motion_ang = joint_motion_ang or [0, 0, 0]

        self.body_pose_lin = body_pose_lin or [0, 0, 0]
        self.body_pose_ang = body_pose_ang or [1, 0, 0, 0]

        self.symbolic_axes = symbolic_axes or []
        if isinstance(self.symbolic_axes, str):
            self.symbolic_axes = symbolic_axes.split(" ")

    def init_symbols(self, parent_body, child_body):
        self.parent_body = parent_body
        self.child_body = child_body
        self.name = self.name or "joint_{}_{}".format(self.parent_body.name, self.child_body.name)
        self.state_symbols = {}
        self.state_symbol_derivatives = {}
        self.state_defaults = {}

        for component in self.symbolic_axes:
            if component not in self.state_symbols:
                self.set_symbol(component)

        ## Verify terms assigned to joint_pose_ang quaternion value
        #if not isinstance(self.joint_pose_ang[0], sympy.Symbol):
        #    for val in self.joint_pose_ang:
        #        if isinstance(val, sympy.Symbol):
        #            self.set_symbol("joint_pose_ang_w")
        #            break

    def init_frames(self, root_frame):
        self.base_frame = self.parent_body.frame.orientnew("{}_base".format(self.name), "Quaternion", self.joint_base_ang)
        self.base_frame.set_ang_vel(self.parent_body.frame, 0)
        self.base_point = self.parent_body.point.locatenew("{}_base".format(self.name), list_to_vector(self.parent_body.frame, self.joint_base_lin))
        self.base_point.set_vel(self.parent_body.frame, 0)

        self.end_frame = self.base_frame.orientnew("{}_end".format(self.name), "Quaternion", self.joint_pose_ang)
        self.end_frame.set_ang_vel(self.base_frame, list_to_vector(self.base_frame, self.joint_motion_ang))
        self.end_point = self.base_point.locatenew("{}_end".format(self.name), list_to_vector(self.base_frame, self.joint_pose_lin))
        self.end_point.set_vel(self.base_frame, list_to_vector(self.base_frame, self.joint_motion_lin))

        self.child_body.frame.orient(self.end_frame, "Quaternion", self.body_pose_ang)
        self.child_body.frame.set_ang_vel(self.end_frame, 0)
        self.child_body.point.set_pos(self.end_point, list_to_vector(self.end_frame, self.body_pose_lin))
        self.child_body.point.set_vel(self.end_frame, 0)

        #P.v2pt_theory(O, N, B)
        self.base_point.v1pt_theory(self.parent_body.point, root_frame, self.parent_body.frame)
        self.end_point.v1pt_theory(self.base_point, root_frame, self.base_frame)
        self.child_body.point.v1pt_theory(self.end_point, root_frame, self.end_frame)
        #self.end_point.a2pt_theory(self.base_point, root_frame, self.e
        # nd_frame)

    def set_symbol(self, component, symbol=None, d_symbol=None):
        full_name = "_".join([self.parent_body.name, self.child_body.name, component])
        symbol = symbol or vector.dynamicsymbols(full_name)
        d_symbol = d_symbol or vector.dynamicsymbols(full_name, 1)
        state_vector = getattr(self, component[:-2])
        element = {"w": -4, "x": -3, "y": -2, "z": -1}[component[-1]]
        def_val = state_vector[element]
        state_vector[element] = symbol
        self.state_symbols[component] = symbol
        self.state_defaults[component] = def_val
        self.state_symbol_derivatives[component] = d_symbol

    def get_symbolic_axes(self):
        pose_symbol_axes = []
        motion_symbol_axes = []
        for component in self.symbolic_axes:
            if component.startswith("joint_pose"):
                pose_symbol_axes.append(component)
            if component.startswith("joint_motion"):
                motion_symbol_axes.append(component)
        return pose_symbol_axes, motion_symbol_axes

    def get_symbol_names(self):
        pose_symbol_axes, motion_symbol_axes = self.get_symbolic_axes()
        pose_symbol_names = ["_".join([self.parent_body.name, self.child_body.name, axis]) for axis in pose_symbol_axes]
        motion_symbol_names = ["_".join([self.parent_body.name, self.child_body.name, axis]) for axis in motion_symbol_axes]
        return pose_symbol_names, motion_symbol_names

    def get_dynamic_symbols(self):
        pose_symbol_axes, motion_symbol_axes = self.get_symbolic_axes()
        pose_dynamic_symbols = [self.state_symbols[axis] for axis in pose_symbol_axes]
        motion_dynamic_symbols = [self.state_symbols[axis] for axis in motion_symbol_axes]
        return pose_dynamic_symbols, motion_dynamic_symbols

    def get_def_symbol_values(self):
        pose_symbol_axes, motion_symbol_axes = self.get_symbolic_axes()
        pose_def_symbol_values = [self.state_defaults[axis] for axis in pose_symbol_axes]
        motion_def_symbol_values = [self.state_defaults[axis] for axis in motion_symbol_axes]
        return pose_def_symbol_values, motion_def_symbol_values

    # def get_rel_vel(self, rel_frame, rel_point):
    #     point_vel = self.end_point.vel(rel_frame)
    #     rel_point_vel = rel_point.vel(rel_frame)
    #     lin_vel = point_vel - rel_point_vel
    #     ang_vel = self.end_frame.ang_vel_in(rel_frame)
    #     return lin_vel, ang_vel
    #
    # def get_rel_acc(self, rel_frame, rel_point):
    #     point_acc = self.end_point.acc(rel_frame)
    #     rel_point_acc = rel_point.acc(rel_frame)
    #     lin_acc = point_acc - rel_point_acc
    #     ang_acc = self.end_frame.ang_acc_in(rel_frame)
    #     return lin_acc, ang_acc

    def get_force_substitutions(self, lin_force, ang_force):
        lin_force_mat = lin_force.to_matrix(self.base_frame)
        ang_force_mat = ang_force.to_matrix(self.base_frame)
        subs = {}
        for component in self.symbolic_axes:
            mat = lin_force_mat if component.startswith("joint_motion_lin") else ang_force_mat
            element = {"x": 0, "y": 1, "z": 2}[component[-1]]
            force_value = mat[0, element]
            symbol = self.state_symbols[component]
            subs[symbol] = force_value
        return subs

    def get_derivative_substitutions(self):
        substitutions = {}
        ang_comps = []
        for component in self.state_symbol_derivatives:
            if component.startswith("joint_pose_lin"):
                d_symbol = self.state_symbol_derivatives[component]
                symbol_comp = "joint_motion" + component[10:]
                symbol = self.state_symbols[symbol_comp]
                substitutions[d_symbol] = symbol
            else:
                ang_comps.append(self.state_symbol_derivatives[component])

        q_w, q_x, q_y, q_z = self.joint_pose_ang
        w_x, w_y, w_z = self.joint_motion_ang

        eqs = sympy.physics.vector.functions.kinematic_equations([w_x, w_y, w_z], [q_w, q_x, q_y, q_z], "Quaternion", "")
        substitutions.update(sympy.solve(eqs, ang_comps))

                # theta = sympy.sqrt(q_x**2 + w_y**2 + w_z**2)
                # if theta == 0:
                #     theta_inv = 0
                # else:
                #     theta_inv = sympy.Piecewise((1/theta, abs(theta) > 0.001), (0, True))
                # sin = sympy.sin(theta/2)
                # cos = sympy.cos(theta/2)
                # if component == "joint_pose_ang_w":
                #     d_q_w = self.state_symbol_derivatives[component]
                #     substitutions[d_q_w] = q_w*(sin-1) - q_x*cos*theta_inv - q_y*cos*w_y*theta_inv - q_z*cos*w_z*theta_inv
                # elif component == "joint_pose_ang_x":
                #     d_q_x = self.state_symbol_derivatives[component]
                #     substitutions[d_q_x] = q_w*cos*w_x*theta_inv + q_x*(sin-1) + q_y*cos*w_z*theta_inv - q_z*cos*w_y*theta_inv
                # elif component == "joint_pose_ang_y":
                #     d_q_y = self.state_symbol_derivatives[component]
                #     substitutions[d_q_y] = q_w*cos*w_y*theta_inv - q_x*cos*w_z*theta_inv + q_y*(sin-1) + q_z*cos*w_x/theta
                # elif component == "joint_pose_ang_z":
                #     d_q_z = self.state_symbol_derivatives[component]
                #     substitutions[d_q_z] = q_w*cos*w_z*theta_inv + q_x*cos*w_y*theta_inv + q_y*cos*w_x/theta + q_z*(sin-1)
        return substitutions



