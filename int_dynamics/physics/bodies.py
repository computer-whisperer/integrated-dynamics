from .types import *


class Body:

    def __init__(self, body_mass, name=None):
        self.fixed_children = []
        self.free_children = []
        self.articulated_children = []
        if name is None:
            name = self.__class__.__name__
        self.frame = Frame(name=name)

        self.root_body = None

        self.body_mass = body_mass
        self.articulated_inertia = None
        self.total_mass = 0
        self.forces = []
        self.net_force = None

    def add_free_child(self, body, pose=None, motion=None):
        if pose is None:
            pose = PoseVector(variable=True)
        if motion is None:
            motion = MotionVector(variable=True)
        pose.frame = self.frame
        pose.end_frame = body.frame
        motion.frame = self.frame
        self.free_children.append(({"body": body, "pose": pose, "motion": motion}))

    def add_fixed_child(self, body, pose=None):
        if pose is None:
            pose = PoseVector(variable=False)
        pose.frame = self.frame
        pose.end_frame = body.frame
        self.fixed_children.append(({"body": body, "pose": pose}))

    def add_articulated_child(self, body, body_pose, joint_axis, joint_position=None, joint_motion=None):
        if joint_position is None:
            joint_position = Quaternion(0, 0, 0, 0, variables="a")
        if joint_motion is None:
            joint_motion = Quaternion(0, 0, 0, 0, variables="a")
        joint_axis.frame = self.frame
        body_pose.frame = self.frame
        body_pose.end_frame = body.frame
        joint_axis.frame = self.frame
        self.articulated_children.append({"body": body, "pose": body_pose, "joint_axis": joint_axis, "joint_position": joint_position, "joint_motion": joint_motion})

    def get_all_children(self):
        result = []
        for child in self.fixed_children + self.free_children:
            result.append(child["body"])
            result.extend(child["body"].get_all_children())
        return result

    def get_def_pose_vector(self):
        default_pose = []
        for child in self.fixed_children:
            default_pose.append(child["body"].get_def_pose_vector())
        for child in self.free_children:
            default_pose.append(child["pose"].get_def_variables())
            default_pose.append(child["body"].get_def_pose_vector())
        for child in self.articulated_children:
            default_pose.append(child["joint_position"].get_def_variables())
            default_pose.append(child["body"].get_def_pose_vector())

        if len(default_pose) > 0:
            return np.concatenate(default_pose)
        else:
            return np.array([])

    def get_def_motion_vector(self):
        default_motion = []
        for child in self.fixed_children:
            default_motion.append(child["body"].get_def_motion_vector())
        for child in self.free_children:
            default_motion.append(child["motion"].get_def_variables())
            default_motion.append(child["body"].get_def_motion_vector())
        for child in self.articulated_children:
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
        pose_i = 0
        motion_i = 0
        for child in self.fixed_children:
            def_pose_state = child["body"].get_def_pose_vector()
            def_motion_state = child["body"].get_def_motion_vector()
            child["body"].set_variables(pose_state_vector[pose_i:pose_i + def_pose_state.shape[0]],
                                        motion_state_vector[motion_i:motion_i + def_motion_state.shape[0]])
            pose_i += def_pose_state.shape[0]
            motion_i += def_motion_state.shape[0]

        for child in self.free_children:
            pose_size = child["pose"].get_def_variables().shape[0]
            child["pose"].set_variables(pose_state_vector[pose_i:pose_i+pose_size])
            pose_i += pose_size

            motion_size = child["motion"].get_def_variables().shape[0]
            child["motion"].set_variables(motion_state_vector[motion_i:motion_i+motion_size])
            motion_i += motion_size

            def_pose_state = child["body"].get_def_pose_vector()
            def_motion_state = child["body"].get_def_motion_vector()
            child["body"].set_variables(pose_state_vector[pose_i:pose_i + def_pose_state.shape[0]],
                                        motion_state_vector[motion_i:motion_i + def_motion_state.shape[0]])
            pose_i += def_pose_state.shape[0]
            motion_i += def_motion_state.shape[0]

        for child in self.articulated_children:
            pose_size = child["joint_pos"].get_def_variables().shape[0]
            child["joint_pos"].set_variables(pose_state_vector[pose_i:pose_i+pose_size])
            pose_i += pose_size

            motion_size = child["joint_motion"].get_def_variables().shape[0]
            child["joint_motion"].set_variables(motion_state_vector[motion_i:motion_i+motion_size])
            motion_i += motion_size

            def_pose_state = child["body"].get_def_pose_vector()
            def_motion_state = child["body"].get_def_motion_vector()
            child["body"].set_variables(pose_state_vector[pose_i:pose_i + def_pose_state.shape[0]],
                                        motion_state_vector[motion_i:motion_i + def_motion_state.shape[0]])

    def calculate_frames(self):
        for child in self.fixed_children:
            child["body"].frame.set_parent_frame(self.frame, child["pose"], MotionVector(frame=self.frame))
            child["body"].calculate_frames()
        for child in self.free_children:
            child["body"].frame.set_parent_frame(self.frame, child["pose"], child["motion"])
            child["body"].calculate_frames()
        for child in self.articulated_children:
            joint_rot = Versor(child["joint_axis"], child["joint_position"])
            child_pose_joint_rel = child[""]
            #child["body"].root_pose =

    def get_rigid_body_inertia(self):
        raise NotImplementedError()

    def build_local_inertia(self):
        self.local_inertia = self.get_rigid_body_inertia()
        for child in self.free_children + self.fixed_children + self.articulated_children:
            child["body"].build_local_inertia()
        for child in self.fixed_children:
            self.local_inertia += child["body"].local_inertia

    def get_inverse_dynamics(self, accel_vector, root_accel):
        """
        Recursively compute the inverse dynamics problem using RNEA given the current body's pose and motion, and the
        provided acceleration vector.
        :param accel_vector: A theano tensor corresponding to the acceleration values to calculate applied force from.
        These acceleration values correspond to velocity variables given in the same order from get_def_motion_vector().
        :param local_accel: The acceleration of this body in root-relative coordinates.
        :return: A vector of the same size as accel_vector, but with force values corresponding to the acceleration values.
        """
        # First thing is to build motion vectors from accel_vector
        i = 0
        child_force_vectors = []
        for child in self.fixed_children:
            def_motion_vector = child["body"].get_def_motion_vector()
            child_force_vectors.append(child["body"].get_inverse_dynamics(accel_vector[i, i+def_motion_vector], root_accel))




    def build_inertia_and_bias(self):
        self.articulated_inertia = self.get_rigid_body_inertia()
        for child in self.free_children + self.fixed_children + self.articulated_children:
            child["body"].build_inertia_and_bias()
        for child in self.fixed_children:
            self.articulated_inertia += child["body"].articulated_inertia


     #   moment_sum = XYVector(0, 0)
     #   self.total_mass = self.body_mass
     #   self.moment_inertia = self.get_local_inertia()
     #   for child in self.fixed_children + self.free_children:
     #       child["body"].build_inertia()
     #       self.total_mass += child["body"].total_mass
     #       child_pos, child_rot = child["pos"], child["rot"]
     #       child_com = child_rot.rotate_xyvector(child["body"].total_com).add(child_pos)
     #       child_moment = child_com.multiply_by_scalar(child["body"].total_mass)
     #       self.moment_inertia += child_pos.get_magnitude()**2 * child["body"].total_mass + child["body"].moment_inertia
     #       moment_sum = moment_sum.add(child_moment)
     #   self.total_com = moment_sum.multiply_by_scalar(1/self.total_mass)

    def build_forces(self):
        self.net_force = sum(self.forces)
        for child in self.fixed_children:
            child["body"].build_forces()
            self.net_force += child["pose"].transform_force(child["body"].net_force)
        for child in self.free_children:
            child["body"].build_integration()

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
