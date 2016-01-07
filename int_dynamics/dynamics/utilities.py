import math
import numpy as np
from theano import tensor as T


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
