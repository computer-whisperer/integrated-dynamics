from .types import *
import sympy
#sympy.init_printing()


class EulerIntegrator:

    def __init__(self, math_library="math"):
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

        self.forward_dynamics_func = None
        self.sympy_functions = None

        self.math_library = math_library

    def build_simulation_expressions(self, root_body):
        self.pose_symbols = root_body.get_pose_symbols()
        self.motion_symbols = root_body.get_motion_symbols()
        self.state_symbols = self.pose_symbols + self.motion_symbols
        self.state_names = root_body.get_pose_symbol_components() + root_body.get_motion_symbol_components()
        self.root_body = root_body
        self.default_state = root_body.get_def_pose_symbol_values() + root_body.get_def_motion_symbol_values()
        self.current_state = self.default_state[:]
        print("begin c computation")
        # Compute the forward dynamics problem with the Composite-Rigid-Body algorithm
        # Get C (the force vector required for zero acceleration) from the RNEA inverse dynamics solver
        crb_C = Matrix([self.root_body.get_inverse_dynamics([0 for _ in self.motion_symbols])[0]]).T
        #sympy.pprint(sympy.simplify(crb_C))
        print("finished with c, beginning H computation")
        # Get H by successive calls to RNEA
        columns = []
        for x in range(len(self.motion_symbols)):
            columns.append(self.root_body.get_inverse_dynamics([1 if x == i else 0 for i in range(len(self.motion_symbols))])[0])
        crb_H = Matrix(columns).T
        print("Finished with H")
        forces = Matrix([self.root_body.get_total_forces()]).T
        print("begin solve")
        joint_accel = crb_H.LUsolve(forces-crb_C)
        print("end solve")
        self.dt_symbol = sympy.symbols("dt")
        joint_dv = joint_accel*self.dt_symbol
        new_joint_motion = [self.motion_symbols[x] + joint_dv[x, 0] for x in range(len(joint_dv))]
        print("begin motion integration")
        new_joint_pose = self.root_body.integrate_motion(self.motion_symbols, self.dt_symbol)
        print("end motion integration")
        self.new_state = new_joint_pose + new_joint_motion

    def build_simulation_functions(self):
        if self.math_library == "theano":
            from sympy.printing.theanocode import theano_function
            self.forward_dynamics_func = theano_function(self.state_symbols + [self.dt_symbol], self.new_state)
        else:
            print("begin lambdification")
            self.sympy_functions = []
            for name, expression in zip(self.state_names, self.new_state):
                print("simplifying {}".format(name))
                new_expr = sympy.simplify(expression)
                print("lambdifying {}".format(name))
                self.sympy_functions.append(sympy.lambdify(self.state_symbols + [self.dt_symbol], new_expr))
            print("finish lambdification")
            self.forward_dynamics_func = lambda args: [sympy_fun(args) for sympy_fun in self.sympy_functions]

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
        new_state = self.forward_dynamics_func(self.current_state + [self.dt_value])
        self.current_state = new_state
        self.time += self.dt_value

    def get_time(self):
        return self.time

