from .types import *
from .utilities import *
import sympy
from sympy.physics import vector, mechanics
import time
import dill
sympy.init_printing()


def count_ops(mat):
    count = 0
    shape = mat.shape
    for x in range(shape[0]):
        for y in range(shape[1]):
            count += mat[x, y].count_ops()
    return count



class EulerIntegrator:

    def __init__(self, name=None, math_library="numpy"):
        self.name = name or self.__class__.__name__
        self.state_names = None
        self.state_symbols = None
        self.current_state = None
        self.default_state = None
        self.pose_symbols = None
        self.motion_symbols = None
        self.root_body = None

        self.dt_symbol = None
        self.t_symbol = None
        self.dt_value = 0.1
        self.time = 0

        self.new_state = None
        self.forward_dynamics_func = None

        self.math_library = math_library

        self.vertices = None
        self.edge_matrix = None
        self.edge_matrix_components = None
        self.edge_func = None
        self.symbol_substitutions = {}

    def init_symbols(self, root_body):
        global substitute_symbols
        self.pose_symbols = root_body.get_pose_symbols()
        self.motion_symbols = root_body.get_motion_symbols()
        self.state_symbols = self.pose_symbols + self.motion_symbols
        self.state_names = root_body.get_pose_symbol_components() + root_body.get_motion_symbol_components()
        self.root_body = root_body
        self.default_state = root_body.get_def_pose_symbol_values() + root_body.get_def_motion_symbol_values()
        self.current_state = self.default_state[:]
        self.dt_symbol = sympy.symbols("dt")
        self.t_symbol = sympy.symbols('t')
        substitute_symbols = self.build_state_substitutions()

        self.derivative_substitutions = root_body.get_derivative_substitutions()
        self.differential_equations = []
        for d_symbol in self.derivative_substitutions:
            symbol = self.derivative_substitutions[d_symbol]
            self.differential_equations.append(symbol - d_symbol)
        self.sympy_rigid_bodies = self.root_body.get_sympy_rigid_bodies()
        self.force_tuples = self.root_body.get_force_tuples()

        self.vertices = root_body.get_edges()


    def build_simulation_expressions(self, root_body, root_accel=None, autocache=False):
        self.init_symbols(root_body)
        component_symbols_gen = sympy.numbered_symbols()

        print("Building lists for Kane's Method")
        self.kanes_method = mechanics.KanesMethod(self.root_body.frame, q_ind=self.pose_symbols, u_ind=self.motion_symbols, kd_eqs=self.differential_equations)

        (fr, frstar) = self.kanes_method.kanes_equations(self.force_tuples, self.sympy_rigid_bodies)

        mass_matrix = self.kanes_method.mass_matrix_full
        print("Attempting to invert matrix")
        mass_matrix_inv = mass_matrix.inv()
        forcing = self.kanes_method.forcing_full
        self.new_state = (mass_matrix_inv * forcing) * self.dt_symbol + Matrix(self.state_symbols)
        print("Finished simulation expression build.")

        print("Building vertex expressions")
        new_values = (vert_1.T.tolist()[0] + vert_2.T.tolist()[0] for vert_1, vert_2 in self.vertices)
        self.edge_matrix = Matrix(new_values)
        #components, reduced_exprs = sympy.cse(edge_matrix, symbols=component_symbols_gen, order='none')
        #self.edge_matrix = reduced_exprs[0]
        #self.symbol_substitutions.update(dict(components))

        if autocache:
            self.cache_integrator("{}.pkl".format(self.name))
            self.cache_integrator("autocache.pkl".format(self.name))

    def build_simulation_function(self):
        self.forward_dynamics_func = build_function(
            self.new_state,
            self.symbol_substitutions,
            self.state_symbols + [self.dt_symbol, self.t_symbol],
            self.math_library
        )

    def build_rendering_function(self):
        self.edge_func = build_function(
            self.edge_matrix,
            self.symbol_substitutions,
            self.state_symbols + [self.dt_symbol, self.t_symbol],
            self.math_library
        )

    def build_state_substitutions(self):
        subs = {
            self.dt_symbol: self.dt_value
        }
        for i in range(len(self.state_symbols)):
            subs[self.state_symbols[i]] = self.current_state[i]
        return subs

    def get_edges(self):
        edge_mat = self.edge_func(*(self.current_state + [self.dt_value]))
        return [(edge_mat[edge, :3].tolist(), edge_mat[edge, 3:].tolist()) for edge in range(edge_mat.shape[0])]

    def opengl_draw(self):
        from OpenGL import GL
        GL.glBegin(GL.GL_LINES)
        for vert_1, vert_2 in self.get_edges():
            GL.glVertex3fv(vert_1)
            GL.glVertex3fv(vert_2)
        GL.glEnd()

    def step_time(self, dt=None):
        if dt is not None:
            self.dt_value = dt
        new_state = self.forward_dynamics_func(*(self.current_state + [self.dt_value, self.time]))[:,0].tolist()
        fixed_new_state = [float(state) for state in new_state]
        self.current_state = fixed_new_state
        self.time += self.dt_value

    def get_time(self):
        return self.time

    def reset_simulation(self):
        self.time = 0
        self.current_state = self.default_state[:]

    def cache_integrator(self, path):
        start_time = time.time()
        print("starting integrator cache to {}".format(path))
        with open(path, 'wb') as f:
            dill.settings["recurse"] = True
            dill.dump(self, f, -1)
        print("integrator cache took {} seconds".format(time.time() - start_time))
