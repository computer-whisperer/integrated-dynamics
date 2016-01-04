import math
import numpy as np
import theano
from theano import tensor as T
from theano.tensor import slinalg
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


def integrate_via_ode(expression, integral, wrt, init_val, consider_constant=None):
    """
    Integrates :param expression by linearizing :param expression: around :param integral: and solving the resulting
    ODE of the form
    x'(wrt) = Ax(wrt) + b
    """

    if expression.ndim == 0:
        A = theano.grad(expression, integral, consider_constant=consider_constant)
    else:
        A = theano.gradient.jacobian(expression, integral, consider_constant=consider_constant)
    b = expression - T.dot(integral, A)

    # Equation given by http://math.stackexchange.com/questions/1567784/matrix-differential-equation-xt-axtb-solution-defined-for-non-invertible/1567806?noredirect=1#comment3192556_1567806

    # Matrix exponentiation method
    eat = slinalg.expm(A*wrt)

    # Two methods to calculate the integral:

    # e^(at) method, given by http://wolfr.am/9mNgcOgM
    eat_integral = T.dot(eat-1, T.inv(A))

    # Taylor series method
    def series_advance(i, last_term, A, wrt):
        next_term = T.dot(last_term, A)*wrt/i
        return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-5))
    if expression.ndim == 0:
        init_term = wrt
    else:
        init_term = wrt*T.identity_like(A)
    terms, _ = theano.scan(series_advance,
                           sequences=[T.arange(2, 100)],
                           non_sequences=[A, wrt],
                           outputs_info=init_term,
                           )
    taylor_integral = T.sum(terms, axis=0) + init_term

    # Decide which integral to use, preferring the eat method when it works
    integral = ifelse(T.any(T.isnan(eat_integral)), taylor_integral, eat_integral)

    return T.dot(eat, init_val) + T.dot(integral, b)
