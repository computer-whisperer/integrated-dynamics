import sympy
from sympy.physics import vector, mechanics
from .types import *


class Body:

    def __init__(self, body_mass, name=None):
        self.children = []
        if name is None:
            name = self.__class__.__name__
        self.name = name

        self.frame = vector.ReferenceFrame(name)
        self.point = vector.Point(name)
        self.particle = mechanics.Particle(self.name, self.point, body_mass)
        self.position_symbols = []
        self.motion_symbols = []

        self.root_body = self

        self.vertices = None

        self.body_mass = body_mass
        self.articulated_inertia = 0
        self.articulated_mass = body_mass
        self.articulated_com = 0

        self.forces = []
        self.net_force = None

    def add_child(self, body, joint):
        body.root_body = self
        body.joint = joint
        self.children.append(body)
        joint.init_symbols(self, body)
        joint.init_frames(root_frame=self.root_body.frame)

        body.local_inertia = body.get_rigid_body_inertia()
        body.articulated_inertia = body.local_inertia
        #body.world_inertia = body.frame.root_pose.transform_inertia(body.local_inertia)

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

    def get_inverse_dynamics(self, root_frame, root_point, accel_subs):

        force_subs = {}
        lin_forces = []
        ang_forces = []

        for child in self.children:
            # These are the world accelerations for the child link
            child_world_accel_lin, child_world_accel_ang = (x.subs(accel_subs) for x in child.joint.get_rel_acc(root_frame, root_point))
            # Get the forces exerted on this link by grand-child joints
            child_force_subs, child_joint_forces_lin, child_joint_forces_ang = child.get_inverse_dynamics(root_frame,
                                                                                                    root_point,
                                                                                                    accel_subs)
            # These are the forces that cause the above accelerations
            link_lin_force = self.body_mass*child_world_accel_lin + child_joint_forces_lin
            link_ang_force = self.local_inertia*child_world_accel_ang + child_joint_forces_ang

            # Find the joint force that causes the above link force
            joint_lin_force = link_lin_force
            joint_ang_force = link_ang_force + child.joint.end_point.pos_from(child.joint.base_point).cross(link_lin_force)

            force_subs.update(child.joint.get_force_substitutions(joint_lin_force, joint_ang_force))

            parent_lin_force = joint_lin_force
            parent_ang_force = joint_ang_force + child.joint.base_point.pos_from(self.point).cross(joint_lin_force)

            lin_forces.append(parent_lin_force)
            ang_forces.append(parent_ang_force)

        return force_subs, sum(lin_forces), sum(ang_forces)

    def get_crb(self, root_frame, root_point, accel_subs):

        force_subs = {}
        applied_force_lin = 0
        applied_force_ang = 0
        inertia_components = [self.articulated_inertia]
        mass_components = [self.articulated_mass]
        com_components = [self.articulated_com]

        for child in self.children:
            inertia_components.append(child.articulated_inertia)
            mass_components.append(child.articulated_mass)
            com_components.append(child.joint.end_point.pos_from(self.point) + child.articulated_com)

            # These are the world accelerations for the child link
            child_world_accel_lin, child_world_accel_ang = (x.subs(accel_subs) for x in
                                                            child.joint.get_rel_acc(root_frame, root_point))

            #joint_accel_symbol_values, child_accel_symbol_values, accel_values = split_list(accel_values, [len(motion_symbol_components), len(child["body"].get_motion_symbol_components())])
            if child_world_accel_lin != 0 or child_world_accel_ang != 0:

                # These are the forces that cause the above accelerations
                link_lin_force = child.articulated_mass * child_world_accel_lin
                link_ang_force = child.articulated_inertia * child_world_accel_ang

                # Find the joint force that causes the above link force
                joint_lin_force = link_lin_force
                joint_ang_force = link_ang_force + child.joint.end_point.pos_from(child.joint.base_point).cross(
                    link_lin_force)

                force_subs.update(child.joint.get_force_substitutions(joint_lin_force, joint_ang_force))

                # Transform force to parent link coordinates
                applied_force_lin += joint_lin_force
                applied_force_ang += joint_ang_force + child.joint.base_point.pos_from(self.point).cross(joint_lin_force)
                break
            else:
                child_force_subs, child_applied_force_lin, child_applied_force_ang = child.get_crb(root_frame, root_point, accel_subs)

                # Transform force to joint coordinates
                joint_lin_force = child_applied_force_lin
                joint_ang_force = child_applied_force_ang + child.joint.end_point.pos_from(child.joint.base_point).cross(
                    child_applied_force_lin)

                force_subs.update(child.joint.get_force_substitutions(joint_lin_force, joint_ang_force))

                # Transform force to parent link coordinates
                applied_force_lin += joint_lin_force
                applied_force_ang += joint_ang_force + child.joint.base_point.pos_from(self.point).cross(joint_lin_force)

        self.articulated_mass = sum(mass_components)
        self.articulated_com = sum([a*b for a, b in zip(mass_components, com_components)])/self.articulated_mass
        for inertia, com in zip(inertia_components, com_components):
            com_comp = com.to_matrix(self.frame)
            c_tilde = mechanics.inertia(0, -com_comp[2], com_comp[1],
                                        com_comp[2], 0, -com_comp[0],
                                        -com_comp[1], com_comp[0], 0)
        self.articulated_inertia = sum([a*b for a, b in zip(inertia_components, com_components)])
        #TODO calculate articulated inertia and mass
        return force_subs, applied_force_lin, applied_force_ang

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

    def integrate_motion(self, motion_symbol_values, dt):
        pose_values = []
        for child in self.children:
            joint_motion_symbol_components = child["joint_motion"].get_symbol_components()
            joint_motion_values = child["joint_motion"].get_values()
            child_motion_symbol_components = child["body"].get_motion_symbol_components()

            joint_motion_symbol_values, child_motion_symbol_values, motion_symbol_values = \
                split_list(motion_symbol_values, [len(joint_motion_symbol_components), len(child_motion_symbol_components)])

            joint_motion = MotionVector()
            joint_motion.set_values(joint_motion_values)
            joint_motion.set_values(joint_motion_symbol_values, components=joint_motion_symbol_components)
            new_joint_pose = child["joint_pose"].integrate_motion(joint_motion, dt)
            pose_values.extend(new_joint_pose.get_values(components=child["joint_pose"].get_symbol_components()))
            pose_values.extend(child["body"].integrate_motion(child_motion_symbol_values, dt))
        return pose_values

    def get_edges(self):
        raise NotImplementedError()

    def get_substitutions(self):
        substitutions = {}
        for child in self.children:
            substitutions.update(child["body"].get_substitutions())
            substitutions.update(child["joint_frame_base"].get_substitutions())
            substitutions.update(child["joint_frame_end"].get_substitutions())
            substitutions.update(child["body"].frame.get_substitutions())
        return substitutions


class WorldBody(Body):
    """
    A special body representing the entire world!
    """

    def __init__(self):
        Body.__init__(self, 0)
        self.frame.root_pose = PoseVector(frame=self.frame, end_frame=self.frame)
        self.frame.root_motion = MotionVector(frame=self.frame)

    def get_rigid_body_inertia(self):
        return mechanics.inertia(self.frame, 1, 1, 1)

    def get_edges(self):
        vertices = []
        for child in self.children:
            vertices.extend(child["body"].get_edges())
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
        root_pose = self.frame.root_pose
        hx = self.x_dim/2
        hy = self.y_dim/2
        hz = self.z_dim/2
        v1 = root_pose.transform_vector(XYZVector(+hx, +hy, +hz))
        v2 = root_pose.transform_vector(XYZVector(+hx, +hy, -hz))
        v3 = root_pose.transform_vector(XYZVector(+hx, -hy, +hz))
        v4 = root_pose.transform_vector(XYZVector(+hx, -hy, -hz))
        v5 = root_pose.transform_vector(XYZVector(-hx, +hy, +hz))
        v6 = root_pose.transform_vector(XYZVector(-hx, +hy, -hz))
        v7 = root_pose.transform_vector(XYZVector(-hx, -hy, +hz))
        v8 = root_pose.transform_vector(XYZVector(-hx, -hy, -hz))
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
            vertices.extend(child["body"].get_edges())
        return vertices



