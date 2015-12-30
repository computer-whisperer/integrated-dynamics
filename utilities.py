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


def integrate_via_ode(expression, integral, wrt, init_val, consider_constant=None):
    """
    Integrates :param expression by linearizing :param expression: around :param integral: and solving the resulting
    ODE of the form
    x'(wrt) = Ax(wrt) + b
    """

    if expression.ndim == 0:
        A = theano.grad(expression, integral, consider_constant=consider_constant)
    else:
        A = theano.printing.Print("A")(theano.gradient.jacobian(expression, integral, consider_constant=consider_constant))
    b = expression - T.dot(integral, A)

    # Matrix exponentiation method
    x_star = theano.printing.Print("x star")(-T.dot(T.inv(A), b))
    special_e = T.exp(A*wrt)
    meth1_res = x_star - T.dot(special_e, (init_val - x_star))
    #meth1_res = ifelse(T.isnan(x_star), init_val + expression*wrt, result)

    # Taylor series ODE solution

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
    meth2_res = init_val + T.dot(T.sum(terms, axis=0) + init_term, T.dot(init_val, A) + b)

    return ifelse(T.any(T.isnan(x_star)), meth2_res, meth1_res)