from .utilities import list_to_vector
import sympy
from sympy.physics import vector


class Joint:

    state_symbols = None
    state_defaults = None
    parent_body = None
    child_body = None

    base_frame = None
    base_point = None
    end_frame = None
    end_point = None

    def __init__(self, name=None,
                 joint_base_lin=None, joint_base_ang=None,
                 joint_pose_lin=None, joint_pose_ang=None,
                 joint_motion_lin=None, joint_motion_ang=None,
                 body_pose_lin=None, body_pose_ang=None,
                 symbolic_axes=None):
        self.name = name
        self.joint_base_lin = joint_base_lin or [0, 0, 0]
        self.joint_base_ang = joint_base_ang or [0, 0, 0, 0]

        self.joint_pose_lin = joint_pose_lin or [0, 0, 0]
        self.joint_pose_ang = joint_pose_ang or [0, 0, 0, 0]
        self.joint_motion_lin = joint_motion_lin or [0, 0, 0]
        self.joint_motion_ang = joint_motion_ang or [0, 0, 0]

        self.body_pose_lin = body_pose_lin or [0, 0, 0]
        self.body_pose_ang = body_pose_ang or [0, 0, 0, 0]

        self.symbolic_axes = symbolic_axes or []
        if isinstance(self.symbolic_axes, str):
            self.symbolic_axes = symbolic_axes.split(" ")

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
            "joint_pose_ang_x joint_pose_ang_y joint_pose_ang_z "
            "joint_pose_lin_x joint_pose_lin_y joint_pose_lin_z "
            "joint_motion_ang_x joint_motion_ang_y joint_motion_ang_z "
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
            "joint_pose_ang_x joint_pose_ang_y joint_pose_ang_z joint_motion_ang_x joint_motion_ang_y joint_motion_ang_z"
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
            "joint_pose_ang{axis} joint_motion_ang_{axis}".format(axis=axis)
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

    def init_symbols(self, parent_body, child_body):
        self.name = self.name or "joint_{}_{}".format(self.parent_body.name, self.child_body.name)
        self.parent_body = parent_body
        self.child_body = child_body
        self.state_symbols = {}
        self.state_defaults = {}

        for component in self.symbolic_axes:
            if component not in self.state_symbols:
                self.set_symbol(component)

        # Verify terms assigned to joint_pose_ang quaternion value
        if not isinstance(self.joint_pose_ang[0], sympy.Symbol):
            for val in self.joint_pose_ang:
                if isinstance(val, sympy.Symbol):
                    self.set_symbol("joint_pose_ang_w")
                    break

    def init_frames(self):
        self.base_frame = self.parent_body.frame.orientnew("{}_base".format(self.name), "quaternion", self.joint_base_ang)
        self.base_point = self.parent_body.locatenew("{}_base".format(self.name), list_to_vector(self.parent_body.frame, self.joint_base_lin))

        self.end_frame = self.base_frame.frame.orientnew("{}_end".format(self.name), "quaternion", self.joint_pose_ang)
        self.end_frame.set_ang_vel(self.base_frame, list_to_vector(self.base_frame, self.joint_motion_ang))
        self.end_point = self.base_point.locatenew("{}_end".format(self.name), list_to_vector(self.base_frame, self.joint_pose_lin))
        self.end_point.set_vel(self.base_frame, list_to_vector(self.base_frame, self.joint_pose_lin))

        self.end_frame.orient(self.child_body.frame, "quaternion", self.body_pose_ang)
        self.end_point.locate(self.child_body.point, list_to_vector(self.end_frame, self.body_pose_lin))

    def set_symbol(self, component, symbol=None):
        symbol = symbol or self.new_state_symbol(component)
        state_vector = getattr(self, component[:-2])
        element = {"w": -3, "x": -3, "y": -2, "z": -1}[component[-1]]
        def_val = vector[element]
        state_vector[element] = symbol
        self.state_symbols[component] = symbol
        self.state_defaults[symbol] = def_val

        if component.startswith("joint_pose_ang") and not isinstance(self.joint_pose_ang[0], sympy.Symbol):
            self.set_symbol(sympy.symbols("joint_pose_ang_{}"))

    def new_state_symbol(self, component):
        name = "_".join(["joint_pose_ang_w", self.parent_body.name, self.child_body.name])
        if component.startswith("joint_motion"):
            return vector.dynamicsymbols(name)
        else:
            return sympy.symbols(name)