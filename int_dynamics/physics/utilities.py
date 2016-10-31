import time
import sympy
from sympy import physics
import numpy


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
            needed_symbols = components[symbol].atoms(sympy.Symbol)
            symbol_funcs[symbol] = (needed_symbols, sympy.lambdify(needed_symbols, components[symbol], modules=math_library, dummify=False))
        #final_needed_symbols = expression.atoms(sympy.Symbol)
        final_func = sympy.lambdify(
            arguments,
            expression)

        def resolve_substitute(symbol, values):
            needed_symbols, func = symbol_funcs[symbol]
            args = []
            for sub_symbol in needed_symbols:
                if sub_symbol in values:
                    val = values[sub_symbol]
                else:
                    val = numpy.asarray(float(resolve_substitute(sub_symbol, values)))
                args.append(val)
            value = func(*args)
            values[symbol] = value
            return value

        def sympy_dynamics_func(*arg_values):
            values = {}
            for arg, val in zip(arguments, arg_values):
                values[arg] = numpy.asarray(float(val))
            args = []
            for symbol in arguments:
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