import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
import logging
from int_dynamics import utilities
from scipy.linalg import block_diag
from theano import ifelse
from theano.tensor import slinalg
import time


class DynamicsEngine:

    # Enable parts of the engine as needed
    SIMULATION = True
    PREDICTION = False
    SIMULATION_NOISE = False

    # State data debug sinks
    STATE_DATA_TO_NT = True

    def __init__(self):

        self.loads = {}
        self.sensors = {}
        self.controllers = {}

        self.simulation_func = None
        self.simulation_sensor_func = None
        self.prediction_func = None

        self.state_list = None
        self.state = theano.shared(np.array([[0.0]]), theano.config.floatX)
        self.state_derivative = theano.shared(np.array([[0.0]]), theano.config.floatX)

        self.dt = theano.shared(0.0, theano.config.floatX)
        self.tic_time = 0

        self.sd = None


        self.build_functions()

    def build_functions(self):
        """
        Build all of the Theano tensors and functions for the dynamics engine.
        """
        self.build_loads()
        # Build sensor data
        sensor_data = {}
        for sensor in self.sensors:
            sensor_data.update(self.sensors[sensor].get_sensor_data())
        # Build controls data

        # Build state data
        self.state_list, state_derivative_list, state_vector = self._build_states_and_derivatives(state_order=self.state_list)

        if self.SIMULATION:
            print("Building the simulation functions, this may take some time depending on how complex your model is.")
            # State update function
            new_state = self._build_state_ode_update(self.state_list, state_derivative_list, state_vector)
            self.simulation_func = self._build_state_update_func(new_state)
            # Sensor value update function
            updates = [(state, sensor_data[state]["update"]) for state in sensor_data]
            self.simulation_sensor_func = theano.function([], [], updates=updates)

        if self.PREDICTION:
            print("Building the prediction function, this may take a bit, though not as long as the simulation build.")
            # EKF state update function
            new_state, new_covariance = self._build_state_ekf_update(sensor_data)
            self.prediction_func = self._build_state_update_func(new_state, [(self.state_covariance, new_covariance)])

    def _build_states_and_derivatives(self, state_order=None):
        """
        Get state derivative dictionaries from all components, build state and state derivative lists,
        and build complete state vectors and derivative matrices.
        """
        state_derivatives = {}
        for component in self.loads:
            state_derivatives.update(self.loads[component].get_state_derivatives())

        # Trim away any states that are not shared variables
        trimmed_state_derivatives = {}
        for state in state_derivatives:
            if hasattr(state, "get_value"):
                trimmed_state_derivatives[state] = state_derivatives[state]
        state_derivatives = trimmed_state_derivatives

        if state_order is not None:
            state_derivatives = OrderedDict(sorted(state_derivatives, key=lambda t: state_order.index(t)))

        # build two-dimensional states and derivatives
        twodim_states = []
        state_list = []
        state_derivative_list = []
        for state in state_derivatives:
            # save state to a list for reference
            state_list.append(state)
            state_derivative_list.append(state_derivatives[state])

            # Correct state dimensions
            twodim_states.append(utilities.ensure_column(state))

        # Build complete state vector
        state_vector = T.concatenate(twodim_states)

        return state_list, state_derivative_list, state_vector

    def _build_state_ode_update(self, state_list, state_derivative_list, state_vector):
        """
        Build the updated state vector with the ode integration equation
        """
        derivative_matrix, A = utilities.get_list_derivative(state_derivative_list, state_list)
        b = derivative_matrix - T.dot(A, state_vector)

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

        return (T.dot(slinalg.expm(A*self.dt), state_vector) + T.dot(integral, b)).flatten()

    def _build_state_ekf_update(self, sensor_data):
        """
        Build the predicted state vector with an extended kalman filter
        """
        sensor_states_blocks = []
        sensor_updates_blocks = []
        sensor_error_blocks = []
        for state in sensor_data:
            sensor_states_blocks.append(utilities.ensure_column(state))
            sensor_updates_blocks.append(sensor_data[state]["update"])
            sensor_error_blocks.append(sensor_data[state]["covariance"])
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
        new_state = (self.state + T.dot(kalman, sensor_state - sensor_update)).flatten()
        new_covariance = T.dot(T.identity_like(self.state_covariance) - T.dot(kalman, sensor_update_derivative), covarience_prediction)
        return new_state, new_covariance

    def _build_state_update_func(self, new_state_in, extra_updates=None):
        new_state, state_derivative = utilities.get_list_derivative(new_state_in, self.state_list)
        index = 0
        updates = [(self.state_derivative, state_derivative),
                   (self.state, T.unbroadcast(new_state, 0, 1))]
        if extra_updates is not None:
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

    def build_loads(self):
        """
        This will be overridden by the user
        """
        pass

    def sink_state_data(self):
        state_data = {
            "sensors": {},
            "controllers": {},
            "loads": {},
            "tic_time": self.tic_time
        }
        for load in self.loads:
            state_data["loads"][load] = self.loads[load].get_state()
        for controller in self.controllers:
            state_data["controllers"][controller] = self.controllers[controller].get_state()
        for sensor in self.sensors:
            state_data["sensors"][sensor] = self.sensors[sensor].get_state()
        if self.STATE_DATA_TO_NT:
            self._publish_dictionary_to_nt(state_data, "integrated_dynamics")

    def _publish_dictionary_to_nt(self, dictionary, nt_prefix):
        if self.sd is None:
            from networktables import NetworkTable
            self.sd = NetworkTable.getTable('SmartDashboard')
        for dict_key in dictionary:
            if isinstance(dictionary[dict_key], dict):
                self._publish_dictionary_to_nt(dictionary[dict_key], "/".join((nt_prefix, dict_key)))
            elif hasattr(dictionary[dict_key], 'shape') and dictionary[dict_key].size > 1:
                i = 0
                for value in np.nditer(dictionary[dict_key]):
                    self.sd.putNumber("/".join((nt_prefix, dict_key, str(i))), value)
                    i += 1
            else:
                self.sd.putNumber("/".join((nt_prefix, dict_key)), dictionary[dict_key])


    def simulation_update(self, dt, hal_data=None):
        start_time = time.time()
        for controller in self.controllers:
            self.controllers[controller].set_from_hal_data(hal_data, dt, add_noise=self.SIMULATION_NOISE)
        self.dt.set_value(dt)
        self.simulation_func()
        self.simulation_sensor_func()
        for sensor in self.sensors:
            self.sensors[sensor].update_hal_data(hal_data, dt, add_noise=self.SIMULATION_NOISE)
        self.tic_time = time.time() - start_time
        self.sink_state_data()

    def prediction_update(self, dt):
        self.dt.set_value(dt)
        self.simulation_func()
        self.prediction_func()

    def init_wpilib_devices(self):
        for controller in self.controllers:
            self.controllers[controller].init_device()
        for sensor in self.sensors:
            self.sensors[sensor].init_device()

    def get_state(self):
        state = {}
        for component in self.loads:
            state[component] = self.loads[component].get_state()
        return state
