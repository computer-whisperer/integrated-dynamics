import sympy
from .types import *


class Body:

    def __init__(self, body_mass, name=None):
        self.children = []
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.frame = Frame(name=name)

        self.root_body = None

        self.vertices = None

        self.body_mass = body_mass
        self.articulated_inertia = None
        self.total_mass = 0
        self.forces = []
        self.net_force = None

    def add_child(self, body, pose=None, joint_base=None, joint_pose=None, joint_motion=None):
        """
        Add an articulated child to this body.
        The type of joint is controlled by the variables set in joint_motion.
        Each joint has four frames associated with it: parent body, joint base, joint end, and child body

        :param body: The body instance to add.
        :param body_pose: The pose of the child relative to the joint.
        :param joint_base: The pose of the joint base in local frame coordinates.(optional)
        :param joint_pose: The initial pose of the joint, with variables corresponding to degrees of freedom (optional).
        :param joint_motion: The initial motion of this joint, with variables corresponding to degrees of freedom (optional).
        """

        if pose is None:
            pose = PoseVector()
        if joint_base is None:
            joint_base = PoseVector()
        if joint_pose is None:
            joint_pose = PoseVector()
        if joint_motion is None:
            joint_motion = MotionVector()


        joint_pose.init_symbols(body.name+"_joint_pose")
        joint_motion.init_symbols(body.name+"_joint_motion")

        joint_frame_base = Frame(name="joint_base_{}_{}".format(self.name, body.name))
        joint_frame_end = Frame(name="joint_end_{}_{}".format(self.name, body.name))

        joint_base.frame = self.frame
        joint_base.end_frame = joint_frame_base
        joint_pose.frame = joint_frame_base
        joint_pose.end_frame = joint_frame_end
        joint_motion.frame = joint_frame_base
        pose.frame = joint_frame_end
        pose.end_frame = body.frame

        joint_frame_base.set_parent_frame(self.frame, joint_base)
        joint_frame_end.set_parent_frame(joint_frame_base, joint_pose, joint_motion)
        body.frame.set_parent_frame(joint_frame_end, pose)
        body.local_inertia = body.get_rigid_body_inertia()
        body.world_inertia = body.frame.root_pose.transform_inertia(body.local_inertia)

        self.children.append({
            "body": body,
            "pose": pose,
            "joint_base": joint_base,
            "joint_pose": joint_pose,
            "joint_motion": joint_motion,
            "joint_forces": [],
            "joint_frame_base": joint_frame_base,
            "joint_frame_end": joint_frame_end
        })

    def get_all_children(self):
        result = []
        for child in self.children:
            result.append(child["body"])
            result.extend(child["body"].get_all_children())
        return result

    def get_pose_symbol_components(self):
        pose_symbol_components = []
        for child in self.children:
            pose_symbol_components.extend(["_pose_".join([child["body"].name, comp])
                                           for comp in child["joint_pose"].get_symbol_components()
                                           ])
            pose_symbol_components.extend(["_".join([child["body"].name, comp])
                                           for comp in child["body"].get_pose_symbol_components()
                                           ])
        return pose_symbol_components

    def get_motion_symbol_components(self):
        motion_symbol_components = []
        for child in self.children:
            motion_symbol_components.extend(["_motion_".join([child["body"].name, comp])
                                             for comp in child["joint_motion"].get_symbol_components()
                                             ])
            motion_symbol_components.extend(["_".join([child["body"].name, comp])
                                             for comp in child["body"].get_motion_symbol_components()
                                             ])
        return motion_symbol_components

    def get_pose_symbols(self):
        pose_symbols = []
        for child in self.children:
            pose_symbols.extend(child["joint_pose"].get_symbols())
            pose_symbols.extend(child["body"].get_pose_symbols())
        return pose_symbols

    def get_motion_symbols(self):
        motion_symbols = []
        for child in self.children:
            motion_symbols.extend(child["joint_motion"].get_symbols())
            motion_symbols.extend(child["body"].get_motion_symbols())
        return motion_symbols

    def get_def_pose_symbol_values(self):
        default_values = []
        for child in self.children:
            default_values.extend(child["joint_pose"].get_def_symbol_values())
            default_values.extend(child["body"].get_def_pose_symbol_values())
        return default_values

    def get_def_motion_symbol_values(self):
        default_values = []
        for child in self.children:
            default_values.extend(child["joint_motion"].get_def_symbol_values())
            default_values.extend(child["body"].get_def_motion_symbol_values())
        return default_values

    def get_rigid_body_inertia(self):
        raise NotImplementedError()

    def get_inverse_dynamics(self, accel_values, frame_accel=None):
        """
        Recursively compute the inverse dynamics problem using RNEA given the current body's pose and motion, and the
        provided acceleration vector.
        :param accel_values: A a list of acceleration values to calculate applied force from.
        These acceleration values correspond to velocity symbols given in the same order from get_def_motion_vector().
        :param local_accel: The acceleration of this body in root-relative coordinates.
        :returns: An ExplicitMatrix vector of the same size as accel_vector, but with force values corresponding to the acceleration values.
        :returns: A ForceVector with the sum of all forces applied by child joints, in world-space coordinates.
        """
        if frame_accel is None:
            frame_accel = MotionVector(frame=self.frame)

        # First thing is to build motion vectors from accel_vector
        force_values = [] # List of joint force variables
        child_forces = [] # List of ForceVector instances corresponding to joint forces

        for child in self.children:
            # Load the acceleration values into a MotionVector we can work with
            child_accel = MotionVector(frame=child["joint_motion"].frame)
            motion_symbol_components = child["joint_motion"].get_symbol_components()
            child_accel_vars, accel_values = split_list(accel_values, [len(motion_symbol_components)])
            child_accel.set_values(child_accel_vars, motion_symbol_components)

            # Calculate the world-relative acceleration vector with the local, joint-space acceleration vector
            child_world_accel = frame_accel + \
                                child["body"].frame.root_motion.cross(
                                    child["joint_frame_base"].root_pose.transform_motion(child["joint_motion"])
                                ) + \
                                child["joint_frame_base"].root_pose.transform_motion(child_accel)

            # Use the world acceleration vector to calculate the force applied to the child body in world coordinates.
            child_force = child["body"].world_inertia.dot(child_world_accel) + \
                child["body"].frame.root_motion.cross(
                    child["body"].world_inertia.dot(child["body"].frame.root_motion)
                )

            # Recurse to find any force acting on the child from its children, and use it to calculate the force on
            # the joint in question in world coordinates.
            child_recurse_vars, accel_values = split_list(accel_values, [len(child["body"].get_motion_symbol_components())])
            child_force_values, child_joint_forces = child["body"].get_inverse_dynamics(child_recurse_vars, child_world_accel)

            joint_force = child_force + child_joint_forces

            # Use the inverse of the root pose of the joint base to transform the world force to local coordinates.
            local_joint_force = child["joint_frame_base"].root_pose.inverse().transform_force(joint_force)

            force_values.extend(local_joint_force.get_values(components=child["joint_motion"].get_symbol_components()))
            force_values.extend(child_force_values)
            child_forces.append(joint_force)
        if len(child_forces) > 0:
            child_force_sum = sum(child_forces)
        else:
            child_force_sum = ForceVector(frame=self.frame.root_pose.frame)
        return force_values, child_force_sum

    def get_crb(self, accel_values):
        force_values = []
        applied_force = None
        inertia_components = []
        for child in self.children:
            inertia_components.append(child["body"].articulated_inertia)
            # Load the acceleration values into a MotionVector we can work with
            child_accel = MotionVector(frame=child["joint_motion"].frame)
            motion_symbol_components = child["joint_motion"].get_symbol_components()

            joint_accel_symbol_values, child_accel_symbol_values, accel_values = split_list(accel_values, [len(motion_symbol_components), len(child["body"].get_motion_symbol_components())])
            if 1 in joint_accel_symbol_values:
                child_accel.set_values(joint_accel_symbol_values, motion_symbol_components)
                world_child_accel = child["joint_base_frame"].root_pose.transform_motion(child_accel)
                child_force = child["body"].articulated_inertia.dot(world_child_accel)
                force_values.extend(child_force.get_values(motion_symbol_components))
                applied_force = child_force
                break
            elif 1 in child_accel_symbol_values:
                child_force_values, child_applied_force = child["body"].get_crb(motion_symbol_components)
                local_joint_force = child["joint_frame_base"].root_pose.inverse().transform_force(child_applied_force)
                force_values.extend(local_joint_force.get_values(components=motion_symbol_components))
                force_values.extend(child_force_values)
        if self.articulated_inertia is None:
            self.articulated_inertia = sum(inertia_components)
        return force_values, applied_force

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
        return InertiaMoment.from_comps(sympy.eye(3), XYZVector(), 0, self.frame)

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
        components = 1/12 * self.body_mass * XYZVector(y_sq + z_sq, x_sq + z_sq, x_sq + y_sq)
        return InertiaMoment.from_comps(sympy.diag(components.b, components.c, components.d), XYZVector(), self.body_mass, self.frame)

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



