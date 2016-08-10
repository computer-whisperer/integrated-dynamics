import time
import sympy
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
        pre_funcs = []
        for symbol, expr in components:
            needed_symbols = expr.atoms(sympy.Symbol)
            pre_funcs.append((symbol, needed_symbols, sympy.lambdify(needed_symbols, expr, modules=math_library, dummify=False)))
        final_needed_symbols = expression.atoms(sympy.Symbol)
        final_func = sympy.lambdify(
            final_needed_symbols,
            expression,
            modules=math_library,
            dummify=False)
        def sympy_dynamics_func(*arg_values):
            values = dict(zip(arguments, arg_values))
            for symbol, needed_symbols, func in pre_funcs:
                args = [numpy.asarray(float(values[s])) for s in needed_symbols]
                result = func(*args)
                values[symbol] = result
            args = [numpy.asarray(float(values[s])) for s in final_needed_symbols]
            return final_func(*args)
        result_func = sympy_dynamics_func
        print("finish lambdification")
    print("finished function build in {} seconds".format(time.time()-start_time))
    return result_func