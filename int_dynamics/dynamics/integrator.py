__author__ = 'christian'
import numpy as np
from scipy.linalg import block_diag
import theano
import theano.tensor as T
from theano import ifelse
from theano.printing import Print
from theano.tensor import slinalg
from . import utilities


class Integrator:

    def __init__(self):
        self.updates = []
        self.ekf_updates = []
        self.state_list = []
        self.state_vector = None
        self.state_vector_shared = theano.shared(np.array([0.0]), theano.config.floatX)
        self.state_derivative = theano.shared(np.array([[0.0]]), theano.config.floatX)

        self.control_list = []
        self.control_derivative = theano.shared(np.array([[0.0]]), theano.config.floatX)

        self.dt = theano.shared(0.0, theano.config.floatX)

    def add_control_tensor(self, tensor):
        self.control_list.append(tensor)

    def add_ode_update(self, state_derivatives):
        # Trim away any states that are not shared variables

        trimmed_state_derivatives = {}
        for state in state_derivatives:
            if hasattr(state, "get_value"):
                trimmed_state_derivatives[state] = state_derivatives[state]
        state_derivatives = trimmed_state_derivatives

        # build two-dimensional states and derivatives
        derivatives = []
        twodim_states = []
        for state in state_derivatives:
            # save state to a list for reference
            self.state_list.append(state)
            derivatives.append(state_derivatives[state])

            # Correct state dimensions
            state = utilities.ensure_column(state)
            twodim_states.append(state)

        # Build complete state vector
        self.state_vector = T.concatenate(twodim_states)

        derivative_matrix, A = utilities.get_list_derivative(derivatives, self.state_list)
        b = derivative_matrix - T.dot(A, self.state_vector)

        # Equation given by http://math.stackexchange.com/questions/1567784/matrix-differential-equation-xt-axtb-solution-defined-for-non-invertible/1567806?noredirect=1#comment3192556_1567806
        # Taylor series method
        init_term = T.identity_like(A)*self.dt

        def series_advance(i, last_term, A, wrt):
            next_term = T.dot(last_term, A)*wrt/i
            next_term = T.unbroadcast(next_term, 0, 1)
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-6))
        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 200)],
                               non_sequences=[A, self.dt],
                               outputs_info=init_term,
                               )
        integral = T.sum(terms, axis=0) + init_term

        new_state = (T.dot(slinalg.expm(A*self.dt), self.state_vector) + T.dot(integral, b)).flatten()

        self.updates.append(self.build_updater(new_state))

    def build_ekf_updater(self, sensor_data):
        if not isinstance(sensor_data, list):
            sensor_data = [sensor_data]
        sensor_states_blocks = []
        sensor_updates_blocks = []
        sensor_error_blocks = []
        for sensor in sensor_data:
            for state in sensor:
                sensor_states_blocks.append(utilities.ensure_column(state))
                sensor_updates_blocks.append(sensor[state]["update"])
                sensor_error_blocks.append(sensor[state]["covariance"])
        sensor_state = T.concatenate(sensor_states_blocks)

        sensor_update, sensor_update_derivative = utilities.get_list_derivative(sensor_updates_blocks, self.state_list)
        sensor_error = block_diag(sensor_error_blocks)

        # Initialize state covariance
        self.state_covariance = theano.shared(np.array([[np.nan]]), theano.config.floatX)

        last_covariance = ifelse.ifelse(T.any(T.isnan(self.state_covariance)), T.identity_like(self.state_derivative), self.state_covariance)

        # predict covariance
        covarience_prediction = T.dot(self.state_derivative, last_covariance) + T.dot(last_covariance, self.state_derivative.T)

        # EKF update
        kalman_denominator = T.dot(sensor_update_derivative, T.dot(covarience_prediction, sensor_update_derivative.T)) + sensor_error
        kalman_numerator = T.dot(covarience_prediction, sensor_update_derivative.T)
        kalman = T.dot(kalman_numerator, T.inv(kalman_denominator))
        new_state = self.state_vector + T.dot(kalman, sensor_state - sensor_update)
        new_covariance = T.dot(T.identity_like(self.state_covariance) - T.dot(kalman, sensor_update_derivative), covarience_prediction)

        self.ekf_updates.append(self.build_updater(new_state, extra_updates=[(self.state_covariance, new_covariance)]))

    def add_sensor_update(self, sensor_data):
        if not isinstance(sensor_data, list):
            sensor_data = [sensor_data]
        updates = []
        for sensor in sensor_data:
            for state in sensor:
                updates.append((state, sensor[state]["update"]))
        self.updates.append(theano.function([], [], updates=updates))

    def build_updater(self, new_state_in, extra_updates=[]):
        new_state, state_derivative = utilities.get_list_derivative(new_state_in, self.state_list)
        index = 0
        updates = [(self.state_derivative, state_derivative)]

        if len(self.control_list) > 0:
            _, control_derivative = utilities.get_list_derivative(new_state_in, self.control_list)
            updates.append((self.control_derivative, control_derivative))
        updates.extend(extra_updates)
        for state in self.state_list:
            if state.ndim == 0:
                state_len = 1
            else:
                state_len = T.shape(state)[0]
            state_update = new_state[index:index+state_len]
            state_update = state_update.reshape(T.shape(state))
            updates.append((state, state_update))
            index += state_len
        return theano.function([], [], updates=updates)

    def update_physics(self, dt=None, do_ekf=False):
        if dt is not None:
            self.dt.set_value(dt)
        for update in self.updates:
            update()
        if do_ekf:
            for update in self.ekf_updates:
                update()
