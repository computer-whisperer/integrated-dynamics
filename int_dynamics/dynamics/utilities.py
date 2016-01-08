import math
import numpy as np
import theano
from theano import tensor as T


def ensure_column(tensor):
    if tensor.ndim == 0:
        return tensor.dimshuffle('x', 'x')
    elif tensor.ndim == 1:
        return tensor.dimshuffle(0, 'x')
    else:
        return tensor


def ensure_row(tensor):
    if tensor.ndim == 0:
        return tensor.dimshuffle('x', 'x')
    elif tensor.ndim == 1:
        return tensor.dimshuffle('x', 0)
    else:
        return tensor


def get_list_derivative(expressions, wrt_list):
    if not isinstance(expressions, list):
        expressions = [expressions]
    corrected_expressions = []
    derivative_rows = []
    for expression in expressions:
        blocks = theano.gradient.jacobian(expression, wrt_list, disconnected_inputs='ignore')

        # Correct dimensions
        for i in range(len(blocks)):
            if expression.ndim == 0:
                blocks[i] = ensure_row(blocks[i])
            else:
                blocks[i] = ensure_column(blocks[i])
        derivative_rows.append(T.concatenate(blocks, axis=1))
        corrected_expressions.append(ensure_column(expression))
    return T.concatenate(corrected_expressions), T.concatenate(derivative_rows)


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
