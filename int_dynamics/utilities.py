import math
import numpy as np
import theano
from theano import tensor as T
from threading import Lock
from int_dynamics.version import __version__
from os.path import join, exists, dirname
from os import makedirs
import pickle
import hashlib
import sys


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

build_lock = Lock()


def cache_object(object_class, file_path, pickle_dir=".pickle_cache"):
    sys.setrecursionlimit(10000)
    with open(file_path, 'rb') as f:
        m = hashlib.md5()
        while True:
            data = f.read(8192)
            if not data:
                break
            m.update(data)
        file_hash = m.hexdigest()
    with build_lock:
        cache_fname = "cached_object--integrated_dynamics-{}--file_hash-{}.pickle".format(__version__, file_hash)
        cache_dir = join(dirname(file_path), pickle_dir)
        if not exists(cache_dir):
            makedirs(cache_dir)
        cache_path = join(cache_dir, cache_fname)
        if exists(cache_path):
            print("Loading pickled object {}".format(cache_fname))
            obj = pickle.load(open(cache_path, 'rb'))
        else:
            obj = object_class()
            print("Pickling {}".format(obj))
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f, -1)
    return obj

