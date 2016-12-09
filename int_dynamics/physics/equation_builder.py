from .types import *
from .utilities import *
import sympy
from sympy.physics import vector, mechanics
import time
import pickle
sympy.init_printing()


def count_ops(mat):
    count = 0
    shape = mat.shape
    for x in range(shape[0]):
        for y in range(shape[1]):
            count += mat[x, y].count_ops()
    return count


class EquationBuilder:

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

        self.state_names = None

        self.state_symbols = None
        self.control_symbols = None

        self.current_state = None
        self.default_state = None

        self.root_body = None

        self.dt_symbol = None
        self.t_symbol = None
        self.dt_value = 0.05
        self.time = 0

        self.state_diff = None
        self.forward_dynamics_func = None

        self.vertices = None
        self.edge_matrix = None
        self.edge_matrix_components = None
        self.edge_func = None
        self.cse_symbol_substitutions = []
        self.controllers = []

    def add_controllers(self, controllers):
        self.controllers.extend(controllers)

    def build_simulation_expressions(self, root_body, autocache=True, integrator='rk4', cse=True):
        global substitute_symbols

        task_timer = TaskTimer()

        pose_dynamic_symbols, motion_dynamic_symbols = root_body.get_dynamic_symbols()
        state_dynamic_symbols = pose_dynamic_symbols + motion_dynamic_symbols
        pose_symbol_names, motion_symbol_names = root_body.get_symbol_names()

        self.state_names = pose_symbol_names + motion_symbol_names
        self.state_symbols = sympy.symbols(self.state_names)

        self.root_body = root_body
        pose_def_state, motion_def_state = root_body.get_def_symbol_values()
        self.default_state = pose_def_state + motion_def_state

        self.current_state = self.default_state[:]
        self.dt_symbol = sympy.symbols("dt")
        self.t_symbol = sympy.symbols("t")
        substitute_symbols = self.build_state_substitutions()

        self.control_symbols = []
        for controller in self.controllers:
            self.control_symbols.extend(controller.get_symbols())

        derivative_substitutions = root_body.get_derivative_substitutions()
        differential_equations = []
        for d_symbol in derivative_substitutions:
            symbol = derivative_substitutions[d_symbol]
            differential_equations.append(symbol - d_symbol)
        sympy_rigid_bodies = self.root_body.get_sympy_rigid_bodies()
        force_tuples = self.root_body.get_force_tuples()

        self.vertices = root_body.get_edges()

        component_symbols_gen = sympy.numbered_symbols()

        task_timer.next_task("building Kane's method equations")
        kanes_method = mechanics.KanesMethod(self.root_body.frame, q_ind=pose_dynamic_symbols, u_ind=motion_dynamic_symbols, kd_eqs=differential_equations)

        (fr, frstar) = kanes_method.kanes_equations(force_tuples, sympy_rigid_bodies)

        mass_matrix = kanes_method.mass_matrix_full.subs(dict(zip(state_dynamic_symbols, self.state_symbols)))
        forcing = kanes_method.forcing_full.subs(dict(zip(state_dynamic_symbols, self.state_symbols)))

        #task_timer.next_task("simplify mass_matrix")
        #mass_matrix = sympy.simplify(mass_matrix)

        task_timer.next_task("solving {} matrix".format(mass_matrix.shape))
        self.state_diff = mass_matrix.LUsolve(forcing)

        if cse:
            task_timer.next_task("running cse() on state_diff")
            components, reduced_exprs = sympy.cse(self.state_diff, symbols=component_symbols_gen, order='none')
            self.cse_symbol_substitutions.extend(components)
            self.state_diff = reduced_exprs[0]
        # print("Doing integration")
        # if integrator == 'euler':
        #     self.new_state = delta_state * self.dt_symbol + Matrix(self.state_symbols)
        # elif integrator == 'rk4':
        #     k1 = delta_state
        #     next_state = Matrix(self.state_symbols) + self.dt_symbol/2*k1
        #     k2 = delta_state.subs(
        #         [(a, b) for a, b in zip(self.state_symbols, next_state.T.tolist()[0])])
        #     next_state = Matrix(self.state_symbols) + self.dt_symbol / 2 * k2
        #     k3 = delta_state.subs(
        #         [(a, b) for a, b in zip(self.state_symbols, next_state.T.tolist()[0])])
        #     next_state = Matrix(self.state_symbols) + self.dt_symbol * k3
        #     k4 = delta_state.subs(
        #         [(a, b) for a, b in zip(self.state_symbols, next_state.T.tolist()[0])])
        #     self.new_state = Matrix(self.state_symbols) + self.dt_symbol/6*(k1 + 2*k2 + 2*k3 + k4)

        task_timer.next_task("building vertex expressions")
        self.edge_matrix = Matrix([vert_1.T.tolist()[0] + vert_2.T.tolist()[0] for vert_1, vert_2 in self.vertices])
        self.edge_matrix = self.edge_matrix.subs(dict(zip(state_dynamic_symbols, self.state_symbols)))
        #components, reduced_exprs = sympy.cse(edge_matrix, symbols=component_symbols_gen, order='none')
        #self.edge_matrix = reduced_exprs[0]
        #self.symbol_substitutions.update(dict(components))

        if autocache:
            task_timer.next_task("caching equations")
            self.cache_equations("{}.pkl".format(self.name))
            self.cache_equations("autocache.pkl".format(self.name))

        task_timer.finish_task()

    def build_simulation_function(self, math_library="numpy"):
        self.forward_dynamics_func = build_function(
            self.state_diff,
            self.cse_symbol_substitutions,
            self.state_symbols + self.control_symbols,
            math_library
        )

    def build_rendering_function(self, math_library="numpy"):
        self.edge_func = build_function(
            self.edge_matrix,
            self.cse_symbol_substitutions,
            self.state_symbols,
            math_library
        )

    def build_state_substitutions(self):
        subs = {
            self.dt_symbol: self.dt_value
        }
        for i in range(len(self.state_symbols)):
            subs[self.state_symbols[i]] = self.current_state[i]
        return subs

    def get_edges(self):
        edge_mat = self.edge_func(*(self.current_state))
        return [(edge_mat[edge, :3].tolist(), edge_mat[edge, 3:].tolist()) for edge in range(edge_mat.shape[0])]

    def opengl_draw(self):
        from OpenGL import GL
        GL.glBegin(GL.GL_LINES)
        for vert_1, vert_2 in self.get_edges():
            GL.glVertex3fv(vert_1)
            GL.glVertex3fv(vert_2)
        GL.glEnd()

    def get_controller_values(self):
        values = {}
        for controller in self.controllers:
            values.update(controller.get_values())
        return [values[symbol] for symbol in self.control_symbols]

    def step_time(self, dt=None, integrator='rk4'):
        if dt is not None:
            self.dt_value = dt
        controller_values = self.get_controller_values()
        if integrator == 'euler':
            state_diff = self.forward_dynamics_func(*(self.current_state + controller_values))[:,0].tolist()
            self.current_state = [float(diff)*self.dt_value + last_state for diff, last_state in zip(state_diff, self.current_state)]
        if integrator == 'rk4':
            k1 = self.forward_dynamics_func(*(self.current_state+controller_values))[:, 0].tolist()
            new_state = [curr + self.dt_value/2*diff for curr, diff in zip(self.current_state, k1)]
            k2 = self.forward_dynamics_func(*(new_state+controller_values))[:, 0].tolist()
            new_state = [curr + self.dt_value / 2 * diff for curr, diff in zip(self.current_state, k2)]
            k3 = self.forward_dynamics_func(*(new_state+controller_values))[:, 0].tolist()
            new_state = [curr + self.dt_value * diff for curr, diff in zip(self.current_state, k3)]
            k4 = self.forward_dynamics_func(*(new_state+controller_values))[:, 0].tolist()
            self.current_state = [curr + self.dt_value/6*(k1_comp + 2*k2_comp + 2*k3_comp + k4_comp) for curr, k1_comp, k2_comp, k3_comp, k4_comp in zip(self.current_state, k1, k2, k3, k4)]
        self.time += self.dt_value

    def get_time(self):
        return self.time

    def reset_simulation(self):
        self.time = 0
        self.current_state = self.default_state[:]

    def clean_equations(self):
        self.root_body = None
        self.vertices = None

    def cache_equations(self, path):
        self.clean_equations()
        task_timer = TaskTimer()
        task_timer.next_task("caching equation builder to {}".format(path))
        with open(path, 'wb') as f:
            pickle.dump(self, f, -1)
        task_timer.finish_task()
