from .types import *
from .utilities import *
import sympy
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
        self.dt_value = 0.1
        self.time = 0

        self.new_state = None
        self.forward_dynamics_func = None

        self.math_library = math_library

        self.vertices = None
        self.edge_matrix = None
        self.edge_matrix_components = None
        self.edge_func = None
        self.symbol_substitutions = None

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
        substitute_symbols = self.build_state_substitutions()
        self.vertices = root_body.get_edges()
        self.symbol_substitutions = self.root_body.get_substitutions()

    def build_simulation_expressions(self, root_body, root_accel=None, autocache=True):
        self.init_symbols(root_body)
        component_symbols_gen = sympy.numbered_symbols()
        print("Building crb_C")
        # Compute the forward dynamics problem with the Composite-Rigid-Body algorithm
        # Get C (the force vector required for zero acceleration) from the RNEA inverse dynamics solver
        crb_C = Matrix([self.root_body.get_inverse_dynamics([0 for _ in self.motion_symbols], root_accel)[0]]).T
        #print(crb_C.evalf(subs=self.build_state_substitutions()))
#        components, reduced_exprs = sympy.cse(crb_C, symbols=component_symbols_gen)
#        self.new_state_components.extend(components)
#        crb_C = reduced_exprs[0]
        #print(count_ops(crb_C))
        print("Building crb_H")
        # Get H by CRB
        columns = [None for _ in range(len(self.motion_symbols))]
        for x in reversed(range(len(self.motion_symbols))):
            column = self.root_body.get_inverse_dynamics([1 if x == i else 0 for i in range(len(self.motion_symbols))])[0]
            while len(column) < len(self.motion_symbols):
                column.append(0)
            columns[x] = column
        for y in range(len(self.motion_symbols)):
            for x in range(x+1, len(self.motion_symbols)):
                columns[x][y] = columns[y][x]
        crb_H = Matrix(columns).T
        #print(crb_H.evalf(subs=self.build_state_substitutions()))
        if True:
            components, reduced_exprs = sympy.cse(crb_H, symbols=component_symbols_gen, order="none")
            self.symbol_substitutions.update(components)
            crb_H = reduced_exprs[0]
        forces = Matrix([self.root_body.get_total_forces()]).T
        print("Begin solve")
        joint_accel = crb_H.LUsolve(forces-crb_C)
        #print(joint_accel.evalf(subs=self.build_state_substitutions()))
        if True:
            start_time = time.time()
            print("Begin cse")
            components, reduced_exprs = sympy.cse(joint_accel, symbols=component_symbols_gen, order="none")
            self.symbol_substitutions.update(dict(components))
            joint_accel = reduced_exprs[0]
            print("cse took {} seconds".format(time.time()-start_time))
        print("symbol break")
        joint_accel_list = []
        for i in range(len(self.motion_symbols)):
            value = joint_accel[i, 0]
            new_symbol = next(component_symbols_gen)
            self.symbol_substitutions[new_symbol] = value
            joint_accel_list.append(new_symbol)
        print("Begin integration")
        new_joint_motion = [self.motion_symbols[i] + joint_accel_list[i]*self.dt_symbol for i in range(len(self.motion_symbols))]
        new_joint_pose = self.root_body.integrate_motion(new_joint_motion, self.dt_symbol)
        self.new_state = new_joint_pose + new_joint_motion
        self.new_state = Matrix([self.new_state])
        print("Finished simulation expression build.")

        print("Building vertex expressions")
        edge_matrix = Matrix([vert_1.get_values('bcd') + vert_2.get_values('bcd') for vert_1, vert_2 in self.vertices])
        components, reduced_exprs = sympy.cse(edge_matrix, symbols=component_symbols_gen, order='none')
        self.edge_matrix = reduced_exprs[0]
        self.symbol_substitutions.update(dict(components))

        if autocache:
            self.cache_integrator("{}.pkl".format(self.name))
            self.cache_integrator("autocache.pkl".format(self.name))

    def build_simulation_function(self):
        self.forward_dynamics_func = build_function(
            self.new_state,
            self.symbol_substitutions,
            self.state_symbols + [self.dt_symbol],
            self.math_library
        )

    def build_rendering_function(self):
        self.edge_func = build_function(
            self.edge_matrix,
            self.symbol_substitutions,
            self.state_symbols + [self.dt_symbol],
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
        new_state = self.forward_dynamics_func(*(self.current_state + [self.dt_value]))[0].tolist()
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
            pickle.dump(self, f, -1)
        print("integrator cache took {} seconds".format(time.time() - start_time))
