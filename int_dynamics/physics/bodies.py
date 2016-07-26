from .types import *


class Body:

    def __init__(self, body_mass, name=None):
        self.children = []
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.frame = Frame(name=name)

        self.root_body = None

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
        :param pose: The pose of the child relative to the joint.
        :param joint_base: The pose of the joint base in local frame coordinates.(optional)
        :param joint_pose: The initial pose of the joint, with variables corresponding to degrees of freedom (optional).
        :param joint_motion: The initial motion of this joint, with variables corresponding to degrees of freedom (optional).
        """

        if pose is None:
            pose = PoseVector(variable=False)
        if joint_base is None:
            joint_base = PoseVector(variable=False)
        if joint_pose is None:
            joint_pose = PoseVector(variable=True)
        if joint_motion is None:
            joint_motion = MotionVector(variable=True)

        joint_frame_base = Frame(name="joint_base_{}_{}".format(self.name, body.name))
        joint_frame_end = Frame(name="joint_end_{}_{}".format(self.name, body.name))

        joint_base.frame = self.frame
        joint_base.end_frame = joint_frame_base
        joint_pose.frame = joint_frame_base
        joint_pose.end_frame = joint_frame_end
        joint_motion.frame = joint_frame_base
        pose.frame = joint_frame_end
        pose.end_frame = body.frame
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

    def get_def_pose_vector(self):
        default_pose = []
        for child in self.children:
            default_pose.append(child["joint_pose"].get_def_variables())
            default_pose.append(child["body"].get_def_pose_vector())

        if len(default_pose) > 0:
            return np.concatenate(default_pose)
        else:
            return np.array([])

    def get_def_motion_vector(self):
        default_motion = []
        for child in self.children:
            default_motion.append(child["joint_motion"].get_def_variables())
            default_motion.append(child["body"].get_def_motion_vector())

        if len(default_motion) > 0:
            return np.concatenate(default_motion)
        else:
            return np.array([])

    def set_variables(self, pose_state_vector, motion_state_vector):
        """
        Sets the values of all position and velocity values.
        :param state_vector: A theano vector of position variables to set.
        :param vel_vector: A theano vector of velocity variables to set.
        """
        for child in self.children:
            joint_pose_vector, pose_state_vector = \
                pose_state_vector.vsplit(child["joint_pose"].get_def_variables().shape[0])
            child["joint_pose"].set_variables(joint_pose_vector)

            joint_motion_vector, motion_state_vector = \
                motion_state_vector.vsplit(child["joint_motion"].get_def_variables().shape[0])
            child["joint_motion"].set_variables(joint_motion_vector)

            rec_pose_vector, pose_state_vector = \
                pose_state_vector.vsplit(child["body"].get_def_pose_vector().shape[0])
            rec_motion_vector, motion_state_vector = \
                motion_state_vector.vsplit(child["body"].get_def_motion_vector().shape[0])

            child["body"].set_variables(rec_pose_vector, rec_motion_vector)

    def calculate_frames(self):
        for child in self.children:
            child["joint_frame_base"].set_parent_frame(self.frame, child["joint_base"])
            child["joint_frame_end"].set_parent_frame(child["joint_frame_base"], child["joint_pose"], child["joint_motion"])
            child["body"].frame.set_parent_frame(child["joint_frame_end"], child["pose"])
            child["body"].calculate_frames()
            child["body"].local_inertia = child["body"].get_rigid_body_inertia()
            child["body"].world_inertia = child["body"].frame.root_pose.transform_inertia(child["body"].local_inertia)


    def get_rigid_body_inertia(self):
        raise NotImplementedError()

#    def build_local_inertia(self):
#        self.local_inertia = self.get_rigid_body_inertia()
#        for child in self.free_children + self.fixed_children + self.articulated_children:
#            child["body"].build_local_inertia()
#        for child in self.fixed_children:
#            self.local_inertia += child["body"].local_inertia

    def get_inverse_dynamics(self, accel_vector, frame_accel=None):
        """
        Recursively compute the inverse dynamics problem using RNEA given the current body's pose and motion, and the
        provided acceleration vector.
        :param accel_vector: An ExplicitMatrix vector corresponding to the acceleration values to calculate applied
        force from.
        These acceleration values correspond to velocity variables given in the same order from get_def_motion_vector().
        :param local_accel: The acceleration of this body in root-relative coordinates.
        :returns: An ExplicitMatrix vector of the same size as accel_vector, but with force values corresponding to the acceleration values.
        :returns: A ForceVector with the sum of all forces applied by child joints, in world-space coordinates.
        """
        if frame_accel is None:
            frame_accel = MotionVector(frame = self.frame)
        # for debugging
        full_accel_vector = accel_vector

        # First thing is to build motion vectors from accel_vector
        force_variables = [] # List of ExplicitMatrix elements corresponding to joint force variables
        child_forces = [] # List of ForceVector instances corresponding to joint forces

        for child in self.children:
            # Load the acceleration values into a MotionVector we can work with
            child_accel = MotionVector(frame=child["joint_motion"].frame)
            child_accel_vars, accel_vector = accel_vector.vsplit(child["joint_motion"].get_def_variables().shape[0])
            child_accel.set_variables_from(child["joint_motion"], child_accel_vars)

            # Calculate the world-relative acceleration vector with the local, joint-space acceleration vector
            child_world_accel = frame_accel + \
                                child["body"].frame.root_motion.cross(
                                    child["joint_frame_base"].root_pose.transform_motion(child["joint_motion"])
                                ) + \
                                child["joint_frame_base"].root_pose.transform_motion(child_accel)

            # Use the world acceleration vector to calculate the force applied to the child body in world coordinates.
            child_force = child["body"].world_inertia.motion_dot(child_world_accel) + \
                child["body"].frame.root_motion.cross(
                    child["body"].world_inertia.motion_dot(child["body"].frame.root_motion)
                )

            # Recurse to find any force acting on the child from it's children, and use it to calculate the force on
            # the joint in question in world coordinates.
            child_recurse_vars, accel_vector = accel_vector.vsplit(child["body"].get_def_motion_vector().shape[0])
            child_force_variables, child_joint_forces = child["body"].get_inverse_dynamics(child_recurse_vars, child_world_accel)

            joint_force = child_force + child_joint_forces

            # Use the inverse of the root pose of the joint base to transform the world force to local coordinates.
            local_joint_force = child["joint_frame_base"].root_pose.inverse().transform_force(joint_force)

            force_variables.append(local_joint_force.get_variables_from(child["joint_motion"]))
            force_variables.append(child_force_variables)
            child_forces.append(joint_force)
        if len(force_variables) > 0:
            force_vector = ExplicitMatrix.vstack(force_variables)
        else:
            force_vector = ExplicitMatrix([[]])
        if len(child_forces) > 0:
            child_force_sum = sum(child_forces)
        else:
            child_force_sum = ForceVector(frame=self.frame.root_pose.frame)
        return force_vector, child_force_sum

    def get_total_forces(self):
        local_forces = []
        for child in self.children:
            if len(child["joint_forces"]) > 0:
                joint_force_sum = sum(child["joint_forces"])
            else:
                joint_force_sum = ForceVector(frame=self.frame)
            local_forces.append(joint_force_sum.get_variables_from(child["joint_motion"]))
            local_forces.append(child["body"].get_total_forces())
        if len(local_forces) > 0:
            return ExplicitMatrix.vstack(local_forces)
        else:
            return ExplicitMatrix([[]])

    def integrate_motion(self, joint_motion_vector, dt):
        pose_vectors = []
        for child in self.children:
            def_joint_motion = child["joint_motion"].get_def_variables()
            child_joint_motion_vars, joint_motion_vector = joint_motion_vector.vsplit(def_joint_motion.shape[0])
            child_joint_motion = MotionVector()
            child_joint_motion.set_variables_from(child["joint_motion"], child_joint_motion_vars)
            new_joint_pose = child["joint_pose"].integrate_motion(child_joint_motion, dt)
            pose_vectors.append(new_joint_pose.get_variables())

            rec_joint_motion_vector, joint_motion_vector = joint_motion_vector.vsplit(child["body"].get_def_motion_vector().shape[0])
            pose_vectors.append(child["body"].integrate_motion(rec_joint_motion_vector, dt))
        if len(pose_vectors) > 0:
            return ExplicitMatrix.vstack(pose_vectors)
        else:
            return ExplicitMatrix([[]])

    def build_integration(self, dt, root_position, root_motion):
        for child in self.free_children:
            total_impulse = child["child"].total_force * dt
            root_motion = child["child"].frame.root_motion + self.articulated_inertia.force_dot(total_impulse)
            root_position = child["child"].frame.root_position + self.articulated_inertia.force_dot(total_impulse)
        total_impulse = total_force * dt
        delta_vel = self.articulated_inertia.force_dot(total_impulse)




class WorldBody(Body):
    """
    A special body representing the entire world!
    """

    def __init__(self):
        Body.__init__(self, 0)
        self.frame.root_pose = PoseVector(variable=False, frame=self.frame, end_frame=self.frame)
        self.frame.root_motion = MotionVector(variable=False, frame=self.frame)

    def get_rigid_body_inertia(self):
        return InertiaMoment(DiagonalMatrix3X3(1, 1, 1), XYZVector(), np.inf, self.frame)


class CubeBody(Body):

    def __init__(self, x_dim, y_dim, z_dim, body_mass):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        Body.__init__(self, body_mass)

    def get_rigid_body_inertia(self):
        #https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        x_sq = self.x_dim**2
        y_sq = self.y_dim**2
        z_sq = self.z_dim**2
        components = 1/12 * self.body_mass * XYZVector(y_sq + z_sq, x_sq + z_sq, x_sq + y_sq)
        return InertiaMoment(DiagonalMatrix3X3(components.b, components.c, components.d), XYZVector(), self.body_mass, self.frame)
