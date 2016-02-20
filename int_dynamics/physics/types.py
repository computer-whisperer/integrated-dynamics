import sympy
from sympy.matrices import Matrix, eye, zeros
import numpy # Required for workaround https://stackoverflow.com/questions/38040846/error-with-sympy-lambdify-for-piecewise-functions-and-numpy-module
from . import integrators


class Frame:
    """
    A utility class representing a frame of reference
    """
    auto_substiute_poses = False

    parent_frame = None

    relative_pose = None
    relative_motion = None

    def __init__(self, name, root_pose=None, root_motion=None):
        self.name = name
        self.root_pose = root_pose
        self.root_motion = root_motion
        self.substitutions = {}

    def set_parent_frame(self, frame, relative_pose, relative_motion=None):
        self.parent_frame = frame
        if relative_motion is None:
            relative_motion = MotionVector(frame=frame)
        frame.check_identity(relative_pose.frame)
        frame.check_identity(relative_motion.frame)
        self.relative_pose = relative_pose
        self.relative_motion = relative_motion
        self.root_pose = frame.root_pose.transform_pose(relative_pose)
        if self.auto_substiute_poses:
            self.substitutions.update(self.root_pose.symbol_substitute("{}_root_pose".format(self.name)))
        self.root_motion = frame.root_pose.transform_motion(relative_motion) + frame.root_motion

    def check_identity(self, other):
        if other is not self:
            raise ValueError("Frame of reference error: frames not equal: {} and {}".format(self.name, other.name))

    def get_substitutions(self):
        return self.substitutions

class Quaternion:

    def __init__(self, a, b, c, d, symbol_components=""):
        self.symbol_components = symbol_components
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def init_symbols(self, symbol_prefix=""):
        for component in self.symbol_components:
            setattr(self, "def_{}".format(component), getattr(self, component))
            setattr(self, component, sympy.symbols("_".join([symbol_prefix, component])))

    @classmethod
    def from_matrix(cls, matrix, components="abcd"):
        args = []
        for val in "abcd":
            if val in components:
                args.append(matrix[components.index(val), 0])
            else:
                args.append(0)
        return cls(*args)

    @classmethod
    def from_list(cls, matrix, components="abcd"):
        args = []
        for val in "abcd":
            if val in components:
                args.append(matrix[components.index(val)])
            else:
                args.append(0)
        return cls(*args)

    def get_def_symbol_values(self):
        return [getattr(self, "def_"+var) for var in self.symbol_components]

    def get_symbol_components(self):
        return self.symbol_components

    def get_symbols(self):
        return self.get_values(self.symbol_components)

    def get_values(self, components="abcd"):
        return [getattr(self, var_name) for var_name in components]

    def set_values(self, values, components="abcd"):
        for i in range(len(components)):
            setattr(self, components[i], values[i])

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

    def to_rot_matrix(self):
        return Matrix([
            [1 - 2*self.c*self.c - 2*self.d*self.d, 2*self.b*self.c - 2*self.d*self.a, 2*self.b*self.d + 2*self.c*self.a],
            [2*self.b*self.c + 2*self.d*self.a, 1 - 2*self.b*self.b - 2*self.d*self.d, 2*self.c*self.d - 2*self.b*self.a],
            [2*self.b*self.d - 2*self.c*self.a, 2*self.c*self.d + 2*self.b*self.a, 1 - 2*self.b*self.b - 2*self.c*self.c]
        ])

    def to_cross_matrix(self):
        return Matrix([
            [0, -self.d, self.c],
            [self.d, 0, -self.b],
            [-self.c, self.b, 0]
        ])

    def transpose(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def get_magnitude(self):
        return (self.a**2 + self.b**2 + self.c**2 + self.d**2)**0.5

    def as_matrix(self, values="abcd"):
        return Matrix([[getattr(self, var_name)] for var_name in values])

    def get_ndarray(self, subs=None):
        import numpy
        return numpy.array(self.as_matrix().evalf(subs=subs)).astype(numpy.float64)[:, 0]

    def symbol_substitute(self, symbol_prefix):
        substitutions = {}
        for component in "abcd":
            value = getattr(self, component)
            if not sympy.sympify(value).is_constant():
                sub_symbol = sympy.symbols("_".join([symbol_prefix, component]))
                setattr(self, component, sub_symbol)
                substitutions[sub_symbol] = value
        return substitutions


def XYZVector(x=0, y=0, z=0, symbols=False):
    return Quaternion(0, x, y, z, symbol_components="bcd" if symbols else "")


def XYVector(x=0, y=0, symbols=False):
    return Quaternion(0, x, y, 0, symbol_components="bc" if symbols else "")


def Angle(theta=0, symbols=False, use_constant=False):
    if use_constant:
        sin = sympy.sin(theta / 2)
        cos = sympy.cos(theta / 2)
        return Quaternion(cos, 0, 0, sin, symbol_components="ad" if symbols else "")
    else:
        return Quaternion(0, 0, 0, theta, symbol_components="d" if symbols else "")


def Versor(v=None, theta=0, symbols=False):
    if v is None:
        v = XYZVector()
    sin = sympy.sin(theta / 2)
    cos = sympy.cos(theta / 2)
    return Quaternion(cos, sin * v.b, sin * v.c, sin * v.d, symbol_components="abcd" if symbols else "")


class SpatialVector:

    def __init__(self, linear_component, angular_component, frame=None):
        self.frame = frame
        self.linear_component = linear_component
        self.angular_component = angular_component

    @classmethod
    def from_matrix(cls, matrix, frame=None):
        linear_component = Quaternion.from_list(matrix[:3], "bcd")
        angular_component = Quaternion.from_list(matrix[3:], "bcd")
        return cls(linear_component, angular_component, frame=frame)

    @classmethod
    def from_spatial_mat(cls, matrix, frame=None):
        angular_component = Quaternion.from_list(matrix[:3], "bcd")
        linear_component = Quaternion.from_list(matrix[3:], "bcd")
        return cls(linear_component, angular_component, frame=frame)

    def init_symbols(self, symbol_prefix=""):
        self.linear_component.init_symbols("_".join([symbol_prefix, "lin"]))
        self.angular_component.init_symbols("_".join([symbol_prefix, "ang"]))

    def get_symbol_components(self):
        symbols = []
        for symbol in self.linear_component.get_symbol_components():
            symbols.append("_".join(["lin", symbol]))
        for symbol in self.angular_component.get_symbol_components():
            symbols.append("_".join(["ang", symbol]))
        return symbols

    def get_def_symbol_values(self):
        return self.linear_component.get_def_symbol_values() + self.angular_component.get_def_symbol_values()

    def get_symbols(self):
        return self.get_values(components=self.get_symbol_components())

    def get_values(self, components=("lin_a", "lin_b", "lin_c", "lin_d", "ang_a", "ang_b", "ang_c", "ang_d")):
        linear_components = ""
        angular_components = ""
        for comp in components:
            if comp.startswith("lin_"):
                linear_components += (comp[4])
            if comp.startswith("ang_"):
                angular_components += (comp[4])
        return self.linear_component.get_values(linear_components) + self.angular_component.get_values(angular_components)

    def set_values(self, values, components=("lin_a", "lin_b", "lin_c", "lin_d", "ang_a", "ang_b", "ang_c", "ang_d")):
        linear_components = ""
        angular_components = ""
        for comp in components:
            if comp.startswith("lin_"):
                linear_components += (comp[4])
            if comp.startswith("ang_"):
                angular_components += (comp[4])
        linear_values, angular_values = split_list(values, [len(linear_components)])
        self.linear_component.set_values(linear_values, linear_components)
        self.angular_component.set_values(angular_values, angular_components)

    def __add__(self, other):
        self.frame.check_identity(other.frame)
        return self.__class__(self.linear_component+other.linear_component, self.angular_component+other.angular_component, frame=self.frame)

    def __radd__(self, other):
        if other == 0:
            return self
        self.frame.check_identity(other.frame)
        return self.__class__(other.linear_component+self.linear_component, other.angular_component+self.angular_component, frame=self.frame)

    def __sub__(self, other):
        self.frame.check_identity(other.frame)
        return self.__class__(self.linear_component-other.linear_component, self.angular_component-other.angular_component, frame=self.frame)

    def symbol_substitute(self, symbol_prefix):
        substitutions = {}
        substitutions.update(self.linear_component.symbol_substitute(symbol_prefix + "_lin"))
        substitutions.update(self.angular_component.symbol_substitute(symbol_prefix + "_ang"))
        return substitutions

    def as_matrix(self):
        return self.linear_component.as_matrix().col_join(self.angular_component.as_matrix())

    def get_ndarray(self, subs=None):
        import numpy
        return numpy.array(self.as_matrix().evalf(subs=subs)).astype(numpy.float64)[:, 0]

    def to_spatial_mat(self):
        return Matrix([
            self.angular_component.get_values("bcd") + self.linear_component.get_values("bcd")
        ]).T

class PoseVector(SpatialVector):

    def __init__(self, linear_component=None, angular_component=None, variable=False, frame=None, end_frame=None):
        if linear_component is None:
            linear_component = XYZVector(symbols=variable)
        if angular_component is None:
            angular_component = Versor(symbols=variable)
        self.end_frame = end_frame
        SpatialVector.__init__(self, linear_component, angular_component, frame=frame)

        self.transform_matrix = None

    def get_transform_matrix(self):
        if self.transform_matrix is None:
            rot_mat = self.angular_component.to_rot_matrix()
            trans_cross = self.linear_component.to_cross_matrix().T
            r1 = rot_mat.row_join(zeros(3))
            r2 = (rot_mat*trans_cross).row_join(rot_mat)
            self.transform_matrix = r1.col_join(r2)
        return self.transform_matrix

    def transform_vector(self, vector):
        """
        :param vector: An XYZVector relative to this pose's endpoint
        :return: vector in coordinates relative to this pose's root
        """
        return self.angular_component.sandwich_mul(vector) + self.linear_component

    def transform_pose(self, pose):
        """
        :param pose: A pose relative to this pose's endpoint
        :return: pose in coordinates relative to this pose's root
        """
        self.end_frame.check_identity(pose.frame)
        linear_component = self.angular_component.sandwich_mul(pose.linear_component) + self.linear_component
        angular_component = self.angular_component*pose.angular_component
        return PoseVector(linear_component, angular_component, frame=self.frame, end_frame=pose.end_frame)

    def transform_motion(self, motion):
        """
        :param motion: A motion relative to this pose's endpoint
        :return: motion in coordinates relative to this pose's root
        """
        self.end_frame.check_identity(motion.frame)
        angular_component = self.angular_component.sandwich_mul(motion.angular_component)
        linear_component = self.angular_component.sandwich_mul(motion.linear_component) + \
            self.linear_component.cross(angular_component)
        return MotionVector(linear_component, angular_component, frame=self.frame)

    def transform_force(self, force):
        """
        :param force: A force relative to this pose's endpoint
        :return: force in coordinates relative to this pose's root
        """
        self.end_frame.check_identity(force.frame)
        #linear_component = self.angular_component.sandwich_mul(force.linear_component)
        #angular_component = self.angular_component.sandwich_mul(force.angular_component) + \
        #    self.linear_component.cross(linear_component)
        return ForceVector.from_spatial_mat(
            self.inverse().get_transform_matrix().T*force.to_spatial_mat(),
            frame=self.frame
        )

    def transform_inertia(self, inertia):
        self.end_frame.check_identity(inertia.frame)
        #rot_matrix = self.angular_component.to_rot_matrix()
        #result_mat = rot_matrix * inertia.inertia_matrix * rot_matrix.T + \
        #    rot_matrix*self.linear_component.to_cross_matrix().T*inertia.mass*inertia.com.to_cross_matrix().T
        #new_com = self.angular_component.sandwich_mul(inertia.com) + self.linear_component
        #return InertiaMoment(result_mat, new_com, inertia.mass, self.frame)
        trans_mat = self.inverse().get_transform_matrix()
        return InertiaMoment(trans_mat.T*inertia.matrix*trans_mat, self.frame)

    def transpose(self):
        return self.__class__(self.linear_component.transpose(), self.angular_component.transpose(), frame=self.end_frame, end_frame=self.frame)

    def inverse(self):
        inverse_angular = self.angular_component.transpose()
        inverse_linear = inverse_angular.sandwich_mul(self.linear_component)*-1
        return PoseVector(inverse_linear, inverse_angular, frame=self.end_frame, end_frame=self.frame)

    def integrate_motion(self, motionvector, dt):
        linear_component = self.linear_component + motionvector.linear_component * dt
        angular_magnitude = motionvector.angular_component.get_magnitude()
        #print(angular_magnitude)
        angular_normal = motionvector.angular_component * (1/sympy.Piecewise((angular_magnitude, angular_magnitude > 0.01), (1, True)))
        angular_delta = Versor(angular_normal, angular_magnitude*dt)
        #print(angular_delta.get_ndarray(subs=integrators.substitute_symbols))
        angular_component = angular_delta.hamilton(self.angular_component)
        linear_component.symbol_components = self.linear_component.symbol_components
        angular_component.symbol_components = self.angular_component.symbol_components
        return PoseVector(linear_component, angular_component, frame=self.frame)


class MotionVector(SpatialVector):

    def __init__(self, linear_component=None, angular_component=None, variable=False, frame=None):
        if linear_component is None:
            linear_component = XYZVector(symbols=variable)
        if angular_component is None:
            angular_component = XYZVector(symbols=variable)
        SpatialVector.__init__(self, linear_component, angular_component, frame)

    def cross(self, other):
        self.frame.check_identity(other.frame)
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
            linear_component = XYZVector(symbols=variable)
        if angular_component is None:
            angular_component = XYZVector(symbols=variable)
        SpatialVector.__init__(self, linear_component, angular_component, frame)


class InertiaMoment:

    def __init__(self, matrix, frame):
        self.matrix = matrix
        self.frame = frame

    @classmethod
    def from_comps(cls, inertia_matrix, com, mass, frame):
        c_tilde = com.to_cross_matrix()
        r1 = (inertia_matrix - mass*c_tilde*c_tilde).row_join(mass*c_tilde)
        r2 = (mass*c_tilde.T).row_join(mass*eye(3))
        matrix = r1.col_join(r2)
        return cls(matrix, frame)

    def dot(self, motion_vector):
        """
        Multiply the inertia tensor by the motion vector
        :param motion_vector: The motion vector to right-multiply by
        :return: force_vector = dot(I, motion_vector)
        """
        self.frame.check_identity(motion_vector.frame)
        #angular_component = Quaternion.from_matrix(self.inertia_matrix * motion_vector.angular_component.as_matrix("bcd"), "bcd") \
        #                    + self.com.cross(motion_vector.linear_component)*self.mass
        #linear_component = (motion_vector.linear_component + motion_vector.angular_component.cross(self.com))*self.mass
        result_mat = self.matrix*motion_vector.to_spatial_mat()
        ang_component = Quaternion(0, result_mat[0, 0], result_mat[1, 0], result_mat[2, 0])
        lin_component = Quaternion(0, result_mat[3, 0], result_mat[4, 0], result_mat[5, 0])
        return ForceVector(lin_component, ang_component, frame=self.frame)

    def __add__(self, other):
        self.frame.check_identity(other.frame)
        inertia_matrix = self.matrix + other.matrix
        return InertiaMoment(inertia_matrix, self.frame)

    def __sub__(self, other):
        self.frame.check_identity(other.frame)
        inertia_matrix = self.matrix - other.matrix
        return InertiaMoment(inertia_matrix, self.frame)


def split_list(list, sizes):
    i = 0
    result = []
    for size in sizes:
        result.append(list[:][i: i + size])
        i += size
    result.append(list[:][i:])
    return result
