import sympy
from sympy.physics import vector, mechanics


class Body:

    def __init__(self, body_mass, name=None):
        self.children = []
        if name is None:
            name = self.__class__.__name__
        self.name = name

        self.frame = vector.ReferenceFrame(name)
        self.point = vector.Point(name)
        self.point.set_vel(self.frame, 0)

        self.position_symbols = []
        self.motion_symbols = []

        self.root_body = self

        self.vertices = None

        self.body_mass = body_mass
        self.articulated_inertia = 0
        self.articulated_mass = body_mass
        self.articulated_com = 0

        self.rigid_body = None

        self.forces = []
        self.net_force = None

    def add_child(self, body, joint):
        body.root_body = self
        body.joint = joint
        self.children.append(body)
        joint.init_symbols(self, body)
        joint.init_frames(root_frame=self.root_body.frame)

        body.local_inertia = body.get_rigid_body_inertia()
        body.rigid_body = mechanics.RigidBody(body.name, body.point, body.frame, body.body_mass, (body.local_inertia, body.point))

    def get_all_children(self):
        result = []
        for child in self.children:
            result.append(child)
            result.extend(child.get_all_children())
        return result

    def get_pose_symbol_components(self):
        pose_symbols = []
        for child in self.children:
            pose_symbols.extend(child.joint.get_pose_symbolic_axes())
            pose_symbols.extend(child.get_pose_symbols())
        return pose_symbols

    def get_motion_symbol_components(self):
        motion_symbols = []
        for child in self.children:
            motion_symbols.extend(child.joint.get_motion_symbolic_axes())
            motion_symbols.extend(child.get_motion_symbols())
        return motion_symbols

    def get_pose_symbols(self):
        pose_symbols = []
        for child in self.children:
            pose_symbols.extend(child.joint.get_pose_symbols())
            pose_symbols.extend(child.get_pose_symbols())
        return pose_symbols

    def get_motion_symbols(self):
        motion_symbols = []
        for child in self.children:
            motion_symbols.extend(child.joint.get_motion_symbols())
            motion_symbols.extend(child.get_motion_symbols())
        return motion_symbols

    def get_def_pose_symbol_values(self):
        default_values = []
        for child in self.children:
            default_values.extend(child.joint.get_def_pose_symbol_values())
            default_values.extend(child.get_def_pose_symbol_values())
        return default_values

    def get_def_motion_symbol_values(self):
        default_values = []
        for child in self.children:
            default_values.extend(child.joint.get_def_motion_symbol_values())
            default_values.extend(child.get_def_motion_symbol_values())
        return default_values

    def get_rigid_body_inertia(self):
        raise NotImplementedError()

    def get_force_tuples(self):
        force_tuples = []
        force_tuples.extend(self.forces)
        for child in self.children:
            force_tuples.extend(child.get_force_tuples())
        return force_tuples

    def get_total_forces(self):
        local_forces = []
        for child in self.children:
            if len(child["joint_forces"]) > 0:
                joint_force_sum = sum(child["joint_forces"])
            else:
                joint_force_sum = ForceVector(frame=self.frame)
            local_forces.extend(joint_force_sum.get_values(components=child["joint_motion"].get_symbol_components()))
            local_forces.extend(child["body"].get_total_forces())
        return local_forces

    def get_edges(self):
        raise NotImplementedError()

    def get_derivative_substitutions(self):
        substiutions = self.joint.get_derivative_substitutions()
        for child in self.children:
            substiutions.update(child.get_derivative_substitutions())
        return substiutions

    def get_sympy_rigid_bodies(self):
        rigid_bodies = [self.rigid_body]
        for child in self.children:
            rigid_bodies.extend(child.get_sympy_rigid_bodies())
        return rigid_bodies


class WorldBody(Body):
    """
    A special body representing the entire world!
    """

    def __init__(self):
        Body.__init__(self, 0)

    def get_rigid_body_inertia(self):
        return mechanics.inertia(self.frame, 1, 1, 1)

    def get_derivative_substitutions(self):
        substitutions = {}
        for child in self.children:
            substitutions.update(child.get_derivative_substitutions())
        return substitutions

    def get_sympy_rigid_bodies(self):
        rigid_bodies = []
        for child in self.children:
            rigid_bodies.extend(child.get_sympy_rigid_bodies())
        return rigid_bodies


    def get_edges(self):
        vertices = []
        for child in self.children:
            vertices.extend(child.get_edges())
        return vertices


class CubeBody(Body):

    def __init__(self, x_dim, y_dim, z_dim, body_mass, name=None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        Body.__init__(self, body_mass, name=name)

    def get_rigid_body_inertia(self):
        #https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        x_sq = self.x_dim**2
        y_sq = self.y_dim**2
        z_sq = self.z_dim**2
        components = [1/12 * self.body_mass * comp for comp in (y_sq + z_sq, x_sq + z_sq, x_sq + y_sq)]
        #old_inertia = InertiaMoment.from_comps(sympy.diag(components.b, components.c, components.d), XYZVector(), self.body_mass, self.frame)
        return mechanics.inertia(self.frame, *components)

    def get_edges(self):
        hx = self.frame.x*self.x_dim/2
        hy = self.frame.y*self.y_dim/2
        hz = self.frame.z*self.z_dim/2
        center_pos = self.point.pos_from(self.root_body.point)
        v1 = (center_pos + hx + hy + hz).to_matrix(self.root_body.frame)
        v2 = (center_pos + hx + hy - hz).to_matrix(self.root_body.frame)
        v3 = (center_pos + hx - hy + hz).to_matrix(self.root_body.frame)
        v4 = (center_pos + hx - hy - hz).to_matrix(self.root_body.frame)
        v5 = (center_pos - hx + hy + hz).to_matrix(self.root_body.frame)
        v6 = (center_pos - hx + hy - hz).to_matrix(self.root_body.frame)
        v7 = (center_pos - hx - hy + hz).to_matrix(self.root_body.frame)
        v8 = (center_pos - hx - hy - hz).to_matrix(self.root_body.frame)
        vertices = [
            (v1, v2),
            (v1, v3),
            (v1, v5),
            (v2, v4),
            (v2, v6),
            (v3, v4),
            (v3, v7),
            (v4, v8),
            (v5, v6),
            (v5, v7),
            (v6, v8),
            (v7, v8)
        ]
        for child in self.children:
            vertices.extend(child.get_edges())
        return vertices



