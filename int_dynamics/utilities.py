import math
import numpy as np
import theano
import warnings
import sys
from theano import tensor as T
from theano.tensor import slinalg, basic
from theano.tensor.shared_randomstreams import RandomStreams
from threading import Lock


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


def get_covariance_matrix_from_object_dict(mean_vector, object_dict, extra_sources=None, debugger=None):
    source_derivatives = {}
    for key in object_dict:
        variance_data = object_dict[key].get_variance_sources()
        for variance_source in variance_data:
            variance_derivative = theano.gradient.jacobian(mean_vector, variance_source, disconnected_inputs='ignore').dimshuffle(0, 'x')
            #variance_derivative = replace_nans(variance_derivative)
            if debugger is not None:
                #debugger.add_tensor(variance_data[variance_source], "{} variance data".format(key), 2)
                debugger.add_tensor(variance_derivative, "{} variance derivative".format(key), 2)
            source_derivatives[variance_derivative] = variance_data[variance_source]
    if extra_sources is not None:
        source_derivatives.update(extra_sources)
    return get_covariance_matrix(source_derivatives)


def get_covariance_matrix(covariance_sources):
    components = []
    for covariance_derivative in covariance_sources:
        p1 = T.dot(covariance_sources[covariance_derivative], covariance_derivative.T)
        p2 = T.dot(covariance_derivative, p1)
        components.append(p2)
    return sum(components)


def sample_covariance_theano(mean, covariance):
    # http://scicomp.stackexchange.com/q/22111/19265
    srng = RandomStreams(seed=481)
    random = srng.normal(mean.shape)
    decomp = slinalg.cholesky(covariance)
    return T.dot(decomp, random) + mean


def sample_covariance_numpy(mean, covariance):
    nonzero_elements = abs(np.diag(covariance)) > 10**-10
    trimmed_covariance = covariance[nonzero_elements, :][:, nonzero_elements]
    sample_mean = mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if any(nonzero_elements):
            sample_mean[nonzero_elements] = np.random.multivariate_normal(mean[nonzero_elements], trimmed_covariance)
    return sample_mean


def replace_nans(tensor, value=0):
    return T.set_subtensor(tensor[T.isnan(tensor).nonzero()], value)


class DebugTensorLogger:

    def __init__(self, verbosity_level=0):
        self.verbosity_level = verbosity_level
        self.tensors = []
        self.updates = []
        self.last_values = []

    def add_tensor(self, tensor, name=None, verbosity=1):
        if verbosity > self.verbosity_level:
            return
        default_val = 0
        while np.ndim(default_val) < tensor.ndim:
            default_val = np.array([default_val], dtype=tensor.dtype)
        shared_var = theano.shared(default_val, name=name, strict=False)
        self.tensors.append((shared_var, name))
        self.updates.append((shared_var, T.unbroadcast(tensor, *range(tensor.ndim))))

    def get_updates(self):
        return self.updates

    def do_checkup(self, check_nans=True, max_magnitude=10**10, raise_exception=True):
        current_values = [(shared_var.get_value(), name) for shared_var, name in self.tensors]
        for value, name in current_values:
            if check_nans and (np.isnan(value)).any():
                print("Warning: Nan encountered in debug tensor '{}'.".format(name))
                break
            if max_magnitude > 0 and (abs(value) > max_magnitude).any():
                print("Warning: Too big of a value encountered in debug tensor '{}'.".format(name))
                break
        else:
            self.last_values = current_values
            return
        np.set_printoptions(precision=2, suppress=True)
        bad_name = name
        for value, name in current_values:
            print("Current value of {}: \n {}".format(name, value))
        for value, name in self.last_values:
            print("Last value of {}: \n {}".format(name, value))
        if raise_exception:
            raise ValueError("Invalid value encountered in debug tensor '{}'.".format(bad_name))
