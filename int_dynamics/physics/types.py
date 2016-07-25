import numpy as np
from int_dynamics.physics import symbolic_types


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

    def set_parent_frame(self, frame, relative_pose, relative_motion=None):
        self.parent_frame = frame
        if relative_motion is None:
            relative_motion = MotionVector(frame=frame)
        if Frame.assert_frames:
            assert frame is relative_pose.frame and frame is relative_motion.frame
        self.relative_pose = relative_pose
        self.relative_motion = relative_motion
        self.root_pose = frame.root_pose.transform_pose(relative_pose)
        self.root_motion = frame.root_pose.transform_motion(relative_motion) + frame.root_motion


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
            setattr(self, self.variables[i], vars.get(0, i))

    def get_variables(self):
        return ExplicitMatrix([[getattr(self, var_name) for var_name in self.variables]])

    def __add__(self, other):
        return Quaternion(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def __sub__(self, other):
        return Quaternion(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return self.hamilton(other)
        else:
            return Quaternion(self.a*other, self.b*other, self.c*other, self.d*other)

    def __rmul__(self, other):
        if isinstance(other, Quaternion):
            return other.hamilton(self)
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
        assert isinstance(other, Quaternion)
        return self.hamilton(other).hamilton(self.transpose())

    def transpose(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def get_magnitude(self):
        return symbolic_types.sqrt(sum([comp**2 for comp in self.get_components()]))

    def get_array(self):
        return symbolic_types.stack(self.get_components())

    def get_ndarray(self):
        return self.get_array().eval()

    def as_explicit_matrix(self, values="abcd"):
        return ExplicitMatrix([[getattr(self, var_name) for var_name in values]])


def XYZVector(x=0, y=0, z=0, variable=False):
    return Quaternion(0, x, y, z, variables="bcd" if variable else "")


def XYVector(x=0, y=0, variable=False):
    return Quaternion(0, x, y, 0, variables="bc" if variable else "")


def Angle(theta=0, variable=False, use_constant=False):
    if use_constant:
        sin = symbolic_types.sin(theta / 2)
        cos = symbolic_types.cos(theta / 2)
        return Quaternion(cos, 0, 0, sin, variables="ad" if variable else "")
    else:
        return Quaternion(0, 0, 0, theta, variables="d" if variable else "")


def Versor(v=None, theta=0, variable=False):
    if v is None:
        v = XYZVector()
    sin = symbolic_types.sin(theta / 2)
    cos = symbolic_types.cos(theta / 2)
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
        if other == 0:
            return self
        if Frame.assert_frames:
            assert self.frame is other.frame and self.frame is not None
        return self.__class__(other.linear_component+self.linear_component, other.angular_component+self.angular_component, frame=self.frame)

    def __sub__(self, other):
        if Frame.assert_frames:
            assert self.frame is other.frame and self.frame is not None
        return self.__class__(self.linear_component-other.linear_component, self.angular_component-other.angular_component, frame=self.frame)

    def get_def_variables(self):
        return np.concatenate([self.linear_component.get_def_variables(), self.angular_component.get_def_variables()])

    def get_variables(self):
        return ExplicitMatrix.vstack([
            self.linear_component.get_variables(),
            self.angular_component.get_variables()
        ])

    def get_variables_from(self, spatial_vector):
        linear_vars = spatial_vector.linear_component.variables
        angular_vars = spatial_vector.angular_component.variables
        self.linear_component.variables = linear_vars
        self.angular_component.variables = angular_vars
        return ExplicitMatrix.vstack([
            self.linear_component.get_variables(),
            self.angular_component.get_variables()
        ])

    def set_variables(self, vars):
        linear_vars, angular_vars = vars.vsplit(self.linear_component.get_def_variables().shape[0])
        self.linear_component.set_variables(linear_vars)
        self.angular_component.set_variables(angular_vars)

    def set_variables_from(self, spatial_vector, vars):
        linear_vars = spatial_vector.linear_component.variables
        angular_vars = spatial_vector.angular_component.variables
        linear_vals, angular_vals = vars.vsplit(len(linear_vars))
        self.linear_component.variables = linear_vars
        self.angular_component.variables = angular_vars
        self.linear_component.set_variables(linear_vals)
        self.angular_component.set_variables(angular_vals)

    def get_array(self):
        return symbolic_types.concatenate([self.linear_component.get_array(), self.angular_component.get_array()])

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
        vectors = [
            self.angular_component.sandwich_mul(XYZVector(*inertia.inertia_matrix.columns[0])),
            self.angular_component.sandwich_mul(XYZVector(*inertia.inertia_matrix.columns[1])),
            self.angular_component.sandwich_mul(XYZVector(*inertia.inertia_matrix.columns[2]))
        ]
        mat_1 = ExplicitMatrix([[vector.b, vector.c, vector.d] for vector in vectors])
        vectors = [
            self.angular_component.sandwich_mul(XYZVector(*mat_1.columns[0])),
            self.angular_component.sandwich_mul(XYZVector(*mat_1.columns[1])),
            self.angular_component.sandwich_mul(XYZVector(*mat_1.columns[2]))
        ]
        mat_2 = ExplicitMatrix([[vector.b, vector.c, vector.d] for vector in vectors])
        new_com = self.angular_component.sandwich_mul(inertia.com) + self.linear_component
        return InertiaMoment(mat_2, new_com, inertia.mass, self.frame)

    def transpose(self):
        return self.__class__(self.linear_component.transpose(), self.angular_component.transpose(), frame=self.end_frame, end_frame=self.frame)

    def inverse(self):
        inverse_angular = self.angular_component.transpose()
        inverse_linear = inverse_angular.sandwich_mul(self.linear_component)*-1
        return PoseVector(inverse_linear, inverse_angular, frame=self.end_frame, end_frame=self.frame)


class MotionVector(SpatialVector):

    def __init__(self, linear_component=None, angular_component=None, variable=False, frame=None):
        if linear_component is None:
            linear_component = XYZVector(variable=variable)
        if angular_component is None:
            angular_component = XYZVector(variable=variable)
        SpatialVector.__init__(self, linear_component, angular_component, frame)

    def cross(self, other):
        assert self.frame is other.frame
        if isinstance(other, MotionVector):
            angular_component = self.angular_component.cross(other.angular_component)
            linear_component = self.angular_component.cross(other.linear_component) + \
                self.linear_component.cross(other.angular_component)
            return MotionVector(linear_component, angular_component, frame=self.frame)
        else:
            angular_component = self.angular_component.cross(other.angular_component) + \
                self.linear_component.cross(other.linear_component)
            linear_component = self.angular_component.cross(other.linear_component)
            return ForceVector(linear_component, angular_component, frame=self.frame)


class ForceVector(SpatialVector):

    def __init__(self, linear_component=None, angular_component=None, variable=False, frame=None):
        if linear_component is None:
            linear_component = XYZVector(variable=variable)
        if angular_component is None:
            angular_component = XYZVector(variable=variable)
        SpatialVector.__init__(self, linear_component, angular_component, frame)





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
        angular_component = self.inertia_matrix.dot(motion_vector.angular_component.as_explicit_matrix(values="cbd")).as_quaternion() \
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


class ExplicitMatrix:

    def __init__(self, columns, shape=None):
        self.columns = columns
        if shape is None:
            shape = (len(columns), len(columns[0]))
        self.shape = shape

    def elementwise_op(self, other, operation):
        assert self.shape == other.shape
        new_columns = []
        for x in range(self.shape[0]):
            new_column = []
            for y in range(self.shape[1]):
                new_column.append(operation(self.get(x, y), other.get(x, y)))
            new_columns.append(new_column)
        return ExplicitMatrix(new_columns, self.shape)

    def __add__(self, other):
        return self.elementwise_op(other, lambda a, b: a+b)

    def __sub__(self, other):
        return self.elementwise_op(other, lambda a, b: a-b)

    def __mul__(self, other):
        if isinstance(other, ExplicitMatrix):
            return self.elementwise_op(other, lambda a, b: a*b)
        else:
            return self.elementwise_op(self, lambda a, b: a*other)

    def transpose(self):
        new_columns = []
        for x in range(self.shape[0]):
            new_column = []
            for y in range(self.shape[1]):
                new_column.append(self.get(y, x))
            new_columns.append(new_column)
        return ExplicitMatrix(new_columns, self.shape)

    def determinant(self):
        assert self.shape[0] == self.shape[1]
        if self.shape[0] == 2:
            return self.columns[0][0]*self.columns[1][1] - self.columns[1][0]*self.columns[0][1]
        multiplier = 1
        determinant = 0
        for x in range(self.shape[0]):
            if self.get(x, 0) != 0:
                new_cols = []
                for col in range(self.shape[0]):
                    if col != x:
                        new_cols.append(self.columns[col][1:])
                minor = ExplicitMatrix(new_cols)
                determinant += minor.determinant()*self.get(x, 0)*multiplier
            multiplier *= -1
        return determinant

    def dot(self, other):
        assert other.shape[1] == self.shape[0]
        new_shape = (other.shape[0], self.shape[1])
        new_columns = []
        for x in range(new_shape[0]):
            new_column = []
            for y in range(new_shape[1]):
                cell = 0
                for i in range(self.shape[0]):
                    cell += self.get(i, y) * other.get(x, i)
                new_column.append(cell)
            new_columns.append(new_column)
        return ExplicitMatrix(new_columns, new_shape)

    def solve(self, vector):
        """
        Solve the equation other = self.dot(return value)
        :param vector:
        :return: the solution to the equation
        """
        det = self.determinant()
        result = []
        for x in range(self.shape[0]):
            new_columns = []
            for x2 in range(self.shape[0]):
                if x == x2:
                    new_columns.append(vector.columns[0])
                else:
                    new_columns.append(self.columns[x2])
            minor_det = ExplicitMatrix(new_columns).determinant()
            result.append(minor_det/det)

        return ExplicitMatrix([result])

    def get(self, x, y):
        return self.columns[x][y]

    @staticmethod
    def hstack(explicit_matrices):
        shape = explicit_matrices[0].shape
        columns = []
        for mat in explicit_matrices:
            assert mat.shape[1] == shape[1]
            columns.extend(mat.columns)
        return ExplicitMatrix(columns)

    @staticmethod
    def vstack(explicit_matrices):
        shape = explicit_matrices[0].shape
        columns = [[] for _ in range(shape[0])]
        for mat in explicit_matrices:
            assert mat.shape[0] == shape[0]
            for x in range(shape[0]):
                columns[x].extend(mat.columns[x])
        return ExplicitMatrix(columns, (shape[0], len(columns[0])))

    def vsplit(self, index):
        a_columns = []
        b_columns = []
        for x in range(self.shape[0]):
            a_columns.append(self.columns[x][:index])
            b_columns.append(self.columns[x][index:])
        return ExplicitMatrix(a_columns, (self.shape[0], index)), \
               ExplicitMatrix(b_columns, (self.shape[0], self.shape[1]-index))

    def hsplit(self, index):
        return ExplicitMatrix(self.columns[:index], (index, self.shape[0])), \
               ExplicitMatrix(self.columns[index:], (self.shape[0]-index, self.shape[1]))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.columns[item]

    def as_quaternion(self):
        assert self.shape[0] == 1
        if self.shape[1] == 3:
            return Quaternion(0, self.get(0, 0), self.get(0, 1), self.get(0, 2))
        elif self.shape[1] == 4:
            return Quaternion(self.get(0, 0), self.get(0, 1), self.get(0, 2), self.get(0, 3))
        else:
            assert False

    def get_symbolic_array(self):
        return symbolic_types.stack([symbolic_types.stack(column) for column in self.columns], axis=0)

    def get_ndarray(self):
        return self.get_symbolic_array().eval()


def SymmetricMatrix3X3(m_xx, m_xy, m_xz, m_yy, m_yz, m_zz):
    return ExplicitMatrix([[m_xx, m_xy, m_xz],
                           [m_xy, m_yy, m_xz],
                           [m_xz, m_yz, m_zz]])


def DiagonalMatrix3X3(m_xx, m_yy, m_zz):
    return SymmetricMatrix3X3(m_xx, 0, 0, m_yy, 0, m_zz)