from .types import *
import sympy
import time
sympy.init_printing()


def count_ops(mat):
    count = 0
    shape = mat.shape
    for x in range(shape[0]):
        for y in range(shape[1]):
            count += mat[x, y].count_ops()
    return count



class EulerIntegrator:

    def __init__(self, math_library="numpy"):
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
        self.new_state_components = []
        self.forward_dynamics_func = None

        self.math_library = math_library


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

    def build_simulation_expressions(self, root_body, root_accel=None):
        self.init_symbols(root_body)
        print("begin c computation")
        # Compute the forward dynamics problem with the Composite-Rigid-Body algorithm
        # Get C (the force vector required for zero acceleration) from the RNEA inverse dynamics solver
        crb_C = Matrix([self.root_body.get_inverse_dynamics([0 for _ in self.motion_symbols], root_accel)[0]]).T
#        components, reduced_exprs = sympy.cse(crb_C)
#        self.new_state_components.extend(components)
#        crb_C = reduced_exprs[0]
        #print(count_ops(crb_C))
        print("finished with c, beginning H computation")
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
#        components, reduced_exprs = sympy.cse(crb_H)
#        self.new_state_components.extend(components)
#        crb_H = reduced_exprs[0]
        #print(count_ops(crb_H))
        #print(crb_H.evalf(subs=self.build_state_substitutions()))
        print("Finished with H")
        forces = Matrix([self.root_body.get_total_forces()]).T
        print("begin solve")
        joint_accel = crb_H.LUsolve(forces-crb_C)
        #print(count_ops(joint_accel))
        print("end solve")

        start_time = time.time()
        print("begin cse")
        self.new_state_components, reduced_exprs = sympy.cse(joint_accel)
        joint_accel = reduced_exprs[0]
        print("cse took {} seconds".format(time.time()-start_time))
        print(count_ops(reduced_exprs[0]))
        print(len(self.new_state_components))

        joint_dv = joint_accel*self.dt_symbol
        new_joint_motion = [self.motion_symbols[x] + joint_dv[x, 0] for x in range(len(joint_dv))]
        print("begin motion integration")
        new_joint_pose = self.root_body.integrate_motion(new_joint_motion, self.dt_symbol)

        print("end motion integration")
        self.new_state = new_joint_pose + new_joint_motion
        self.new_state = Matrix([self.new_state])


    def build_simulation_functions(self):
        print("begin function compile")
        start_time = time.time()
        if self.math_library == "theano":
            from sympy.printing.theanocode import theano_function
            print("building theano function")
            #raw_new_state = self.new_state.subs(self.new_state_replacements)
            self.forward_dynamics_func = theano_function(self.state_symbols + [self.dt_symbol], raw_new_state)
            print("finished building theano function")
        else:
            print("begin lambdification")
            pre_funcs = []
            for symbol, expr in self.new_state_components:
                needed_symbols = expr.atoms(sympy.Symbol)
                pre_funcs.append((symbol, needed_symbols, sympy.lambdify(needed_symbols, expr, modules=self.math_library, dummify=False)))
            final_needed_symbols = self.new_state.atoms(sympy.Symbol)
            final_func = sympy.lambdify(
                final_needed_symbols,
                self.new_state,
                modules=self.math_library,
                dummify=False)

            def sympy_dynamics_func(*args_in):
                values = self.build_state_substitutions()
                for symbol, needed_symbols, func in pre_funcs:
                    args = [numpy.asarray(float(values[s])) for s in needed_symbols]
                    result = func(*args)
                    values[symbol] = result
                args = [values[s] for s in final_needed_symbols]
                return final_func(*args)

            self.forward_dynamics_func = sympy_dynamics_func
            print("finish lambdification")
        print("finished function build in {} seconds".format(time.time()-start_time))

    def build_state_substitutions(self):
        subs = {
            self.dt_symbol: self.dt_value
        }
        for i in range(len(self.state_symbols)):
            subs[self.state_symbols[i]] = self.current_state[i]
        return subs

    def step_time(self, dt=None):
        if dt is not None:
            self.dt_value = dt
        new_state = self.forward_dynamics_func(*(self.current_state + [self.dt_value]))[0].tolist()
        fixed_new_state = [float(state) for state in new_state]
        self.current_state = fixed_new_state
        self.time += self.dt_value

    def get_time(self):
        return self.time

