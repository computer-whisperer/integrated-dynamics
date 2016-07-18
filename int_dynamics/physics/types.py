import theano
import theano.tensor as T
import numpy as np


class Frame:
    """
    A utility class representing a frame of reference
    """
    assert_frames = True

    parent_frame = None

    relative_pose = None
    relative_motion = None

    def __init__(self, name, root_pose=None, root_motion=None):
        self.name = name
        self.root_pose = root_pose
        self.root_motion = root_motion

    def set_parent_frame(self, frame, relative_pose=None, relative_motion=None):
        self.parent_frame = frame
        if relative_pose is not None:
            if Frame.assert_frames:
                assert frame is relative_pose.frame and frame is relative_motion.frame
            self.relative_pose = relative_pose
            self.relative_motion = relative_motion
            self.root_pose = frame.root_pose.transform_pose(relative_pose)
            self.root_motion = frame.root_pose.transform_motion(relative_motion) + frame.root_motion
        else:
            transpose_pose = frame.root_pose.transpose()
            self.parent_pose = transpose_pose.transform_pose(self.root_pose)
            self.parent_pose = transpose_pose.transform_motion(self.root_motion)
            self.parent_motion = motion
            self.root_pose = frame.root_pose.transform_pose(pose)
            self.root_pose.end_frame = self
            self.root_motion = frame.root_pose.transform_motion(motion) + frame.root_motion
            self.parent_frame = frame

    def set_parent_frame_root(self, frame, pose, motion):
        if Frame.assert_frames:
            assert frame is pose.frame and frame is motion.frame
        self.parent_frame = frame
        self.parent_pose = pose
        self.parent_motion = motion
        self.root_pose = frame.root_pose.transform_pose(pose)
        self.root_pose.end_frame = self
        self.root_motion = frame.root_pose.transform_motion(motion) + frame.root_motion
        self.parent_frame = frame

    def update_root_pose_motion(self, new_parent_frame, root_pose, root_motion):
        """
        Create an updated frame with the direct, updated root pose and motion.
        :param root_pose:
        :param root_motion:
        :return:
        """


class Quaternion:

    def __init__(self, a, b, c, d, variables=""):
        self.variables = variables
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        for var in variables:
            setattr(self, "def_{}".format(var), getattr(self, var))

    def get_components(self):
        return self.a, self.b, self.c, self.d

    def get_def_variables(self):
        return np.array([getattr(self, "def_"+var) for var in self.variables])

    def set_variables(self, vars):
        for i in range(len(self.variables)):
            setattr(self, self.variables[i], vars[i])

    def __add__(self, other):
        return Quaternion(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def __sub__(self, other):
        return Quaternion(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return self.hamilton(other)
        else:
            return Quaternion(self.a*other, self.b*other, self.c*other, self.d*other)

    def hamilton(self, other):
        a = self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d
        b = self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c
        c = self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b
        d = self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a
        return Quaternion(a, b, c, d)

    def dot(self, other):
        a = self.a*other.a
        b = self.b*other.b
        c = self.c*other.c
        d = self.d*other.d
        return a+b+c+d

    def cross(self, other):
        b = self.c*other.d - self.d*other.c
        c = self.d*other.b - self.b*other.d
        d = self.b*other.c - self.c*other.b
        return Quaternion(0, b, c, d)

    def elementwise_mul(self, other):
        a = self.a * other.a
        b = self.b * other.b
        c = self.c * other.c
        d = self.d * other.d
        return Quaternion(a, b, c, d)

    def elementwise_div(self, other):
        a = self.a / other.a
        b = self.b / other.b
        c = self.c / other.c
        d = self.d / other.d
        return Quaternion(a, b, c, d)

    def sandwich_mul(self, other):
        return self.hamilton(other).hamilton(self.transpose())

    def transpose(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def get_magnitude(self):
        return T.sqrt(T.sum(*[comp**2 for comp in self.get_components()]))

    def get_array(self):
        return T.stack(self.get_components())

    def get_ndarray(self):
        return self.get_array().eval()


def XYZVector(x=0, y=0, z=0, variable=False):
    return Quaternion(0, x, y, z, variables="bcd" if variable else "")


def Versor(v=None, theta=0, variable=False):
    if v is None:
        v = XYZVector()
    if isinstance(theta, T.TensorType):
        sin = T.sin(theta / 2)
        cos = T.cos(theta / 2)
    else:
        sin = np.sin(theta / 2)
        cos = np.cos(theta / 2)
    return Quaternion(cos, sin*v.b, sin*v.c, sin*v.d, variables="abcd" if variable else "")


class SpatialVector:

    def __init__(self, linear_component, angular_component, frame=None):
        self.frame = frame
        self.linear_component = linear_component
        self.angular_component = angular_component

    def __add__(self, other):
        if Frame.assert_frames:
            assert self.frame is other.frame and self.frame is not None
        return self.__class__(self.linear_component+other.linear_component, self.angular_component+other.angular_component, frame=self.frame)

    def __radd__(self, other):
        if Frame.assert_frames:
            assert self.frame is other.frame and self.frame is not None
        return self.__class__(other.linear_component+self.linear_component, other.angular_component+self.angular_component, frame=self.frame)

    def __sub__(self, other):
        if Frame.assert_frames:
            assert self.frame is other.frame and self.frame is not None
        return self.__class__(self.linear_component-other.linear_component, self.angular_component-other.angular_component, frame=self.frame)

    def get_def_variables(self):
        return np.concatenate([self.linear_component.get_def_variables(), self.angular_component.get_def_variables()])

    def set_variables(self, vars=None):
        linear_var_count = self.linear_component.get_def_variables().shape[0]
        self.linear_component.set_variables(vars)
        self.angular_component.set_variables(vars[linear_var_count:])

    def get_array(self):
        return T.concatenate([self.linear_component.get_array(), self.angular_component.get_array()])

    def get_ndarray(self):
        return self.get_array().eval()


class PoseVector(SpatialVector):

    def __init__(self, linear_component=None, angular_component=None, variable=False, frame=None, end_frame=None):
        if linear_component is None:
            linear_component = XYZVector(variable=variable)
        if angular_component is None:
            angular_component = Versor(variable=variable)
        self.end_frame = end_frame
        SpatialVector.__init__(self, linear_component, angular_component, frame=frame)

    def transform_pose(self, pose):
        """
        :param pose: A pose relative to this pose's endpoint
        :return: pose in coordinates relative to this pose's root
        """
        if Frame.assert_frames:
            assert self.end_frame is pose.frame
        linear_component = self.angular_component.sandwich_mul(pose.linear_component) + self.linear_component
        angular_component = self.angular_component*pose.angular_component
        return PoseVector(linear_component, angular_component, frame=self.frame, end_frame=pose.end_frame)

    def transform_motion(self, motion):
        """
        :param motion: A motion relative to this pose's endpoint
        :return: motion in coordinates relative to this pose's root
        """
        if Frame.assert_frames:
            assert self.end_frame is motion.frame
        angular_component = self.angular_component.sandwich_mul(motion.angular_component)
        linear_component = self.angular_component.sandwich_mul(motion.linear_component) + \
            self.linear_component.cross(angular_component)
        return MotionVector(linear_component, angular_component, frame=self.frame)

    def transform_force(self, force):
        """
        :param force: A force relative to this pose's endpoint
        :return: force in coordinates relative to this pose's root
        """
        if Frame.assert_frames:
            assert self.end_frame is force.frame
        linear_component = self.angular_component.sandwich_mul(force.linear_component)
        angular_component = self.angular_component.sandwich_mul(force.angular_component) + \
            self.linear_component.cross(linear_component)
        return MotionVector(linear_component, angular_component, frame=self.frame)

    def transform_inertia(self, inertia):
        if Frame.assert_frames:
            assert self.end_frame is inertia.frame
        mat_1 = Matrix3X3(
            self.angular_component.sandwich_mul(inertia.inertia_matrix.col_1),
            self.angular_component.sandwich_mul(inertia.inertia_matrix.col_2),
            self.angular_component.sandwich_mul(inertia.inertia_matrix.col_3)
        )
        mat_2 = Matrix3X3(
            self.angular_component.sandwich_mul(mat_1.row_1),
            self.angular_component.sandwich_mul(mat_1.row_2),
            self.angular_component.sandwich_mul(mat_1.row_3)
        )
        new_com = self.angular_component.sandwich_mul(inertia.com) + self.linear_component
        return InertiaMoment(mat_2, new_com, inertia.mass, self.frame)

    def transpose(self):
        return self.__class__(self.linear_component.transpose(), self.angular_component.transpose(), frame=self.end_frame, end_frame=self.frame)


class MotionVector(SpatialVector):

    def __init__(self, linear_component=None, angular_component=None, variable=False, frame=None):
        if linear_component is None:
            linear_component = XYZVector(variable=variable)
        if angular_component is None:
            angular_component = XYZVector(variable=variable)
        SpatialVector.__init__(self, linear_component, angular_component, frame)


class ForceVector(SpatialVector):

    def __init__(self, linear_component=None, angular_component=None, variable=False, frame=None):
        if linear_component is None:
            linear_component = XYZVector(variable=variable)
        if angular_component is None:
            angular_component = XYZVector(variable=variable)
        SpatialVector.__init__(self, linear_component, angular_component, frame)


class UndefinedMatrix:

    def __init__(self, columns, shape=None):
        self.columns = columns
        if shape is None:
            shape = (len(columns), len(columns[0]))
        self.shape = shape
        for column in columns:
            assert len(column) == self.shape[1]


    def elementwise_op(self, other, operation):
        assert self.shape == other.shape
        new_columns = []
        for x in range(self.shape[0]):
            new_column = []
            for y in range(self.shape[1]):
                new_column.append(operation(self.get(x, y), other.get(x, y)))
            new_columns.append(new_column)
        return UndefinedMatrix(new_columns, self.shape)

    def __add__(self, other):
        return self.elementwise_op(other, lambda a, b: a+b)

    def __sub__(self, other):
        return self.elementwise_op(other, lambda a, b: a-b)

    def transpose(self):
        new_columns = []
        for x in range(self.shape[0]):
            new_column = []
            for y in range(self.shape[1]):
                new_column.append(operation(self.get(x, y), other.get(x, y)))
            new_columns.append(new_column)
        return UndefinedMatrix(new_columns, self.shape)

    def get(self, x, y):
        return self.columns[x][y]



class Matrix3X3:

    def __init__(self, col_1, col_2, col_3):
        self.col_1 = col_1
        self.col_2 = col_2
        self.col_3 = col_3
        self.row_1 = XYZVector(col_1.b, col_2.b, col_3.b)
        self.row_2 = XYZVector(col_1.c, col_2.c, col_3.c)
        self.row_3 = XYZVector(col_1.d, col_2.d, col_3.d)

    def __add__(self, other):
        return Matrix3X3(self.col_1+other.col_1, self.col_2+other.col_2, self.col_3+other.col_3)

    def __sub__(self, other):
        return Matrix3X3(self.col_1-other.col_1, self.col_2-other.col_2, self.col_3-other.col_3)

    def transpose(self):
        return Matrix3X3(self.row_1, self.row_2, self.row_3)

    def determinant(self):
        determinant_pt_1 = self.col_1.b*self.col_2.c*self.col_3.d + \
                           self.col_2.b*self.col_3.c*self.col_1.d + \
                           self.col_3.b*self.col_1.c*self.col_2.d
        determinant_pt_2 = self.col_3.b*self.col_1.c*self.col_2.d + \
                           self.col_1.b*self.col_3.c*self.col_2.d + \
                           self.col_2.b*self.col_1.c*self.col_3.d
        return determinant_pt_1 - determinant_pt_2

    def dot(self, other):
        x = self.row_1.dot(other)
        y = self.row_2.dot(other)
        z = self.row_3.dot(other)
        return XYZVector(x, y, z)

    def solve(self, vector):
        """
        Solve the equation other = self.dot(return value)
        :param vector:
        :return: the solution to the equation
        """
        x_mat = Matrix3X3(vector, self.col_2, self.col_3)
        y_mat = Matrix3X3(self.col_1, vector, self.col_3)
        z_mat = Matrix3X3(self.col_1, self.col_2, vector)

        det = self.determinant()

        x = x_mat.determinant()/det
        y = y_mat.determinant()/det
        z = z_mat.determinant()/det

        return XYZVector(x, y, z)


def SymmetricMatrix3X3(m_xx, m_xy, m_xz, m_yy, m_yz, m_zz):
    x_col = XYZVector(m_xx, m_xy, m_xz)
    y_col = XYZVector(m_xy, m_yy, m_yz)
    z_col = XYZVector(m_xz, m_yz, m_zz)
    return Matrix3X3(x_col, y_col, z_col)


def DiagonalMatrix3X3(m_xx, m_yy, m_zz):
    return SymmetricMatrix3X3(m_xx, 0, 0, m_yy, 0, m_zz)


class InertiaMoment:

    def __init__(self, inertia_matrix, com, mass, frame):
        self.inertia_matrix = inertia_matrix
        self.mass = mass
        self.com = com
        self.frame = frame

    def motion_dot(self, motion_vector):
        """
        Multiply the inertia tensor by the motion vector
        :param motion_vector: The motion vector to right-multiply by
        :return: force_vector = dot(I, motion_vector)
        """
        if Frame.assert_frames:
            assert motion_vector.frame is self.frame
        omega_cross = motion_vector.angular_component.cross(self.com)
        angular_component = self.inertia_matrix.dot(motion_vector.angular_component) \
                            - self.com.cross(omega_cross)*self.mass \
                            + self.com.cross(motion_vector.linear_component)*self.mass
        linear_component = (motion_vector.linear_component + omega_cross)*self.mass
        return ForceVector(linear_component, angular_component, frame=self.frame)

    def force_dot(self, force_vector):
        """
        Multiply the inverse of the inertia tensor by the force vector
        :param force_vector: The force vector to right-multiply by
        :return: motion_vector such that force_vector = dot(I, motion_vector),
         or motion_vector = dot(I^(-1), force_vector)
        """
        if Frame.assert_frames:
            assert force_vector.frame is self.frame
        angular_component = self.inertia_matrix.solve(force_vector.angular_component + force_vector.linear_component.cross(self.com))
        linear_component = force_vector/self.mass - angular_component.cross(self.com)
        return MotionVector(angular_component, linear_component, frame=self.frame)

    def __add__(self, other):
        if Frame.assert_frames:
            assert self.frame is other.frame
        inertia_matrix = self.inertia_matrix + other.inertia_matrix
        mass = self.mass + other.mass
        com = (self.com*self.mass + other.com*other.mass)/mass
        return InertiaMoment(inertia_matrix, com, mass, self.frame)

    def __sub__(self, other):
        if Frame.assert_frames:
            assert self.frame is other.frame
        inertia_matrix = self.inertia_matrix - other.inertia_matrix
        mass = self.mass - other.mass
        com = (self.com*self.mass - other.com*other.mass)/mass
        return InertiaMoment(inertia_matrix, com, mass, self.frame)
