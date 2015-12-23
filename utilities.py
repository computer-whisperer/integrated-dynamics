import math
import numpy as np
import theano
from theano import tensor as T
from theano.ifelse import ifelse


def rot_matrix(theta):
    if isinstance(theta, float) or isinstance(theta, int):
        sin = math.sin(theta)
        cos = math.cos(theta)
    else:
        sin = T.sin(theta)
        cos = T.cos(theta)
    coss = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])*cos
    sins = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])*sin
    tensor = coss + sins +\
            np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])
    return tensor


def integrate_via_ode(expression, integral, wrt, init_val):
    """
    Integrates :param expression by linearizing :param expression: around :param integral: and solving the resulting
    ODE of the form
    x'(wrt) = Ax(wrt) + b
    """

    if expression.ndim == 0:
        # Matrix exponential ODE solution
        A = theano.grad(expression, integral)[0]
        b = expression - A*integral
        x_star = -T.dot(T.inv(A), b)
        special_e = T.exp(A*wrt)
        result = x_star - T.dot(special_e, (init_val - x_star))
        return ifelse(T.isnan(x_star), init_val + expression*wrt, result)

    else:
        # Taylor series ODE solution
        A = theano.gradient.jacobian(expression, integral)

        def series_advance(i, last_term, A, wrt):
            next_term = T.dot(A, last_term)*wrt/i
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-7))

        init_term = wrt*T.identity_like(A)
        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 500)],
                               non_sequences=[A, wrt],
                               outputs_info=init_term,
                               )
        return init_val + T.dot(T.sum(terms, axis=0) + init_term, expression)