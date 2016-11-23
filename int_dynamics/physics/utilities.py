import time
import sympy
import numpy
import logging

def build_function(expression, components, arguments, math_library):
    print("begin function compile")
    start_time = time.time()
    if math_library == "theano":
        from sympy.printing.theanocode import theano_function
        print("building theano function")
        raw_expression = expression.subs(components)
        result_func = theano_function(arguments, raw_expression)
        print("finished building theano function")
    else:
        print("begin lambdification")
        symbol_funcs = {}
        for symbol in components:
            needed_symbols = list(components[symbol].atoms(sympy.Symbol))
            symbol_funcs[symbol] = (needed_symbols, sympy.lambdify(needed_symbols, components[symbol], modules=math_library, dummify=False))
        final_needed_symbols = list(expression.atoms(sympy.Symbol))
        final_func = sympy.lambdify(
            final_needed_symbols,
            expression)

        def resolve_substitute(symbol, values):
            needed_symbols, func = symbol_funcs[symbol]
            args = []
            for sub_symbol in needed_symbols:
                if sub_symbol in values:
                    val = values[sub_symbol]
                else:
                    val = float(resolve_substitute(sub_symbol, values))
                args.append(val)
            value = func(*args)
            values[symbol] = value
            return value

        def sympy_dynamics_func(*arg_values):
            values = {}
            for arg, val in zip(arguments, arg_values):
                values[arg] = numpy.asarray(float(val))
            args = []
            for symbol in final_needed_symbols:
                if symbol not in values:
                    resolve_substitute(symbol, values)
                args.append(values[symbol])
            return final_func(*args)
        result_func = sympy_dynamics_func
        print("finish lambdification")
    print("finished function build in {} seconds".format(time.time()-start_time))
    return result_func


def list_to_vector(frame, value):
    return value[0]*frame.x + value[1]*frame.y + value[2]*frame.z


class TaskTimer:

    current_task = None
    task_start_time = time.time()

    def next_task(self, task_name):
        self.finish_task()
        self.new_task(task_name)

    def new_task(self, task_name):
        print("Started {}".format(task_name))
        self.current_task = task_name
        self.task_start_time = time.time()

    def finish_task(self):
        if self.current_task is None:
            return
        print("Completed {} in {} seconds.".format(self.current_task, time.time() - self.task_start_time))
        self.current_task = None
