import time
from collections import OrderedDict
import sys
import numpy as np
import theano
import theano.tensor as T
from theano import ifelse
from theano.tensor import slinalg

from int_dynamics import utilities


class DynamicsEngine:

    SINK_IN_SIMULATION = False
    SINK_TO_NT = True

    def __init__(self, mode="simulation"):
        self.mode = mode
        self.loads = {}
        self.sensors = {}
        self.controllers = {}

        self.add_noise = theano.shared(0)

        self.simulation_func = None
        self.simulation_sensor_func = None
        self.prediction_func = None

        self.state_list = None
        self.state_mean = theano.shared(np.array([[0.0]]), theano.config.floatX)
        self.state_covariance = theano.shared(np.array([[0.0]]), theano.config.floatX)

        self.sensor_value_list = []

        self.dt = theano.shared(0.0, theano.config.floatX)
        self.tic_time = 0

        self.sd = None

        self.build_functions()

    @classmethod
    def cached_init(cls, mode):
        filename = sys.modules[cls.__module__].__file__
        return utilities.cache_object(filename, cls, mode)

    def build_functions(self):
        """
        Build all of the Theano tensors and functions for the dynamics engine.
        """
        print("Building the dynamics functions for run mode {}, "
              "this may take some time depending on how complex your model is.".format(self.mode))
        self.build_loads()

        ###################
        # Calculate some data, then build simulation function if asked.
        ####################

        # Build state data
        self.state_list, state_derivative_list, state_vector = self._build_states_and_derivatives(state_order=self.state_list)

        # Build state prediction mean, derivative, and covariance
        state_prediction_mean = self._build_state_ode_update(self.state_list, state_derivative_list, state_vector)
        _,  state_prediction_derivative = utilities.get_list_derivative(state_prediction_mean, self.state_list)
        state_covariance = ifelse.ifelse(T.eq(self.state_covariance.shape[0], 1), T.zeros_like(state_prediction_derivative), self.state_covariance)
        state_prediction_covariance = utilities.get_covariance_matrix_from_object_dict(
            state_prediction_mean, self.loads, {state_prediction_derivative: state_covariance}
        )

        # Build sensor data
        sensor_prediction_data = {}
        for sensor in self.sensors:
            value_predictions = self.sensors[sensor].get_value_prediction()
            for sensor_value in value_predictions:
                self.sensor_value_list.append(sensor_value)
                sensor_prediction_data[sensor_value] = value_predictions[sensor_value]
            sensor_prediction_data.update(self.sensors[sensor].get_value_prediction())

        # Function builds for simulation mode
        if self.mode == "simulation":

            # Simulation update function
            state_updates = self._build_state_updates(state_prediction_mean)
            state_updates.extend([
                (self.state_mean, T.unbroadcast(state_prediction_mean.dimshuffle(0, 'x'), 1)),
                (self.state_covariance, state_prediction_covariance)
            ])
            self.simulation_func = theano.function([], [], updates=state_updates)

            # Sensor value update function
            sensor_value_updates = [(value, sensor_prediction_data[value]) for value in self.sensor_value_list]
            self.simulation_sensor_func = theano.function([], [], updates=sensor_value_updates)
            return

        ####################################################################
        # We want more than just a wimpy little simulation if we get here. #
        # Calculate more data, then build estimation function if asked.    #
        ####################################################################

        # Calculate read sensor values
        read_sensor_values = T.concatenate(self.sensor_value_list)

        prev_sensor_prediction = T.concatenate([sensor_prediction_data[value] for value in self.sensor_value_list])

        # The sensor predictions we get from sensor_prediction_data are all based on shared variables, which currently
        # contain the state values from last tick. We need to update these to base upon the new predicted state. We do
        # this by getting the first derivative and multiplying
        _, sensor_state_derivative = utilities.get_list_derivative(
            [sensor_prediction_data[value] for value in self.sensor_value_list],
            self.state_list
        )

        # Get new sensor prediction mean
        sensor_prediction_mean = T.dot(sensor_state_derivative, state_prediction_mean)
        sensor_prediction_covariance = utilities.get_covariance_matrix_from_object_dict(
            prev_sensor_prediction, self.sensors, {sensor_state_derivative: state_prediction_covariance}
        )

        # From the state prediction and the sensor data, we get the state estimation via a kalman filter
        kalman = T.dot(T.dot(state_prediction_covariance, sensor_state_derivative.T), T.inv(sensor_prediction_covariance))
        state_estimation_mean = state_prediction_mean + T.dot(kalman, read_sensor_values - sensor_prediction_mean)
        state_estimation_covariance = T.dot(T.identity_like(self.state_covariance) - T.dot(kalman, sensor_state_derivative), state_prediction_covariance)

        if self.mode == "estimation":

            # Estimation update function
            state_updates = self._build_state_updates(state_estimation_mean)
            state_updates.extend([
                (self.state_mean, state_estimation_mean),
                (self.state_covariance, state_estimation_covariance)
            ])
            self.estimation_func = theano.function([], [], updates=state_updates)
            return

        ##############################################################
        # We want more than just a trivial estimator if we get here. #
        # Calculate more data, then build the optimizer.             #
        ##############################################################

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
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-3))
        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 100)],
                               non_sequences=[A, self.dt],
                               outputs_info=init_term,
                               )
        integral = T.sum(terms, axis=0) + init_term

        return (T.dot(slinalg.expm(A*self.dt), state_vector) + T.dot(integral, b)).flatten()

    def _build_state_updates(self, new_state):
        index = 0
        updates = []
        for state in self.state_list:
            if state.ndim == 0:
                state_len = 1
            else:
                state_len = T.shape(state)[0]
            state_update = new_state[index:index+state_len]
            state_update = state_update.reshape(T.shape(state))
            updates.append((state, state_update))
            index += state_len
        return updates

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
        if self.SINK_TO_NT:
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
            self.controllers[controller].set_from_hal_data(hal_data, dt)
        self.dt.set_value(dt)
        self.simulation_func()
        self.simulation_sensor_func()
        for sensor in self.sensors:
            self.sensors[sensor].update_hal_data(hal_data, dt)
        self.tic_time = time.time() - start_time
        if self.SINK_IN_SIMULATION:
            self.sink_state_data()

    def estimation_update(self, dt):
        start_time = time.time()
        self.dt.set_value(dt)
        self.estimation_func()
        self.tic_time = time.time() - start_time
        self.sink_state_data()

    def init_wpilib_devices(self):
        for controller in self.controllers:
            self.controllers[controller].init_device()
        for sensor in self.sensors:
            self.sensors[sensor].init_device()

    def poll_sensors(self):
        for sensor in self.sensors:
            sensor.poll_sensor()

    def get_state(self):
        state = {}
        for component in self.loads:
            state[component] = self.loads[component].get_state()
        return state
