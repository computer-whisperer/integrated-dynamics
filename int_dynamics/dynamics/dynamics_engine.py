import time
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from theano import ifelse
from int_dynamics.version import __version__
from os.path import join, exists, dirname
from os import makedirs
import pickle
import threading
import hashlib
import sys
from theano.tensor import slinalg

from int_dynamics import utilities

try:
    import simplestreamer
except ImportError:
    pass


class DynamicsEngine:

    SINK_IN_SIMULATION = False
    SINK_TO_NT = False
    SINK_TO_SIMPLESTREAMER = False
    SS_SIM_PORT = 5803
    SS_EST_PORT = 5804

    RAM_CLEAN = True

    DEBUG_VERBOSITY = 0

    def __init__(self, mode="simulation"):
        self.mode = mode
        self.loads = {}
        self.sensors = {}
        self.controllers = {}
        self.costs = {}

        self.add_noise = theano.shared(0)

        self.simulation_func = None

        self.estimation_func = None
        self.state_flush_func = None
        self.sensor_flush_func = None

        self.state_list = None
        self.state_mean = theano.shared(np.array([[0.0]]), theano.config.floatX)
        self.state_covariance = theano.shared(np.array([[0.0]]), theano.config.floatX)
        self.state_derivative = theano.shared(np.array([[0.0]]), theano.config.floatX)
        self.control_derivative = theano.shared(np.array([[0.0]]), theano.config.floatX)

        # Shared variable updates
        self.state_prediction_mean_update = None
        self.state_prediction_covariance_update = None
        self.state_prediction_derivative_update = None
        self.state_prediction_control_derivative_update = None
        self.state_prediction_debugger = None

        self.state_prediction_derivatives = []
        self.state_prediction_A_derivatives = []
        self.state_prediction_A = None
        self.state_prediction_b_derivatives = []
        self.state_prediction_b = None

        self.state_estimation_mean_update = None
        self.state_estimation_covariance_update = None

        self.state_flush_updates = []
        self.sensor_flush_updates = []

        self.feedback_gains_clock = theano.shared(np.array([0.0]), theano.config.floatX)
        self.feedback_gains = theano.shared(np.array([[[0.0]]]), theano.config.floatX)
        self.feedback_gains_progress_update = None
        self.feedback_gains_progression_func = None

        self.sensor_value_list = []
        self.sensor_prediction_list = []
        self.controller_list = []
        self.control_tensors =[]

        self.dt = theano.shared(0.0, theano.config.floatX)
        self.tic_time = 0

        self.sd = None
        self.streamer = None

        self.rebuild_functions()

    @classmethod
    def cached_init(cls, mode):
        filename = sys.modules[cls.__module__].__file__
        sys.setrecursionlimit(100000)
        with open(filename, 'rb') as f:
            m = hashlib.md5()
            while True:
                data = f.read(8192)
                if not data:
                    break
                m.update(data)
            file_hash = m.hexdigest()
        build_lock = getattr(cls, "build_lock", None)
        if build_lock is None:
            build_lock = threading.Lock()
        with build_lock:
            cache_fname = "cached_object--{}--{}.pickle".format(__version__, file_hash)
            cache_dir = join(dirname(filename), ".pickle_cache")
            if not exists(cache_dir):
                makedirs(cache_dir)
            cache_path = join(cache_dir, cache_fname)
            if exists(cache_path):
                print("Loading cached dynamics engine instance.")
                obj = pickle.load(open(cache_path, 'rb'))
                obj.mode = mode
                # Run the function rebuild, and re-cache if any were recompiled.
                rebuild_count = obj.rebuild_functions()
                if rebuild_count > 0:
                    print("Rebuilt {} functions from cached dynamics engine instance".format(rebuild_count))
                else:
                    if obj.RAM_CLEAN:
                        obj.clean_build_memory()
                    return obj
            else:
                obj = cls(mode)
            print("Caching dynamics engine instance.")
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f, -1)
        if obj.RAM_CLEAN:
            obj.clean_build_memory()
        return obj

    def rebuild_functions(self):
        rebuild_count = 0
        if self.mode in ["simulation", "estimation"] and self.simulation_func is None:
            self.build_simulation_function()
            rebuild_count += 1
        if self.mode in ["simulation"] and self.sensor_flush_func is None:
                self.build_sensor_flush_function()
                rebuild_count += 1
        if self.mode in ["estimation"] and self.estimation_func is None:
                self.build_estimation_function()
                rebuild_count += 1
        if self.state_flush_func is None:
            self.build_state_flush_function()
            rebuild_count += 1
        return rebuild_count

    def clean_build_memory(self):
        del self.state_prediction_mean_update
        del self.state_prediction_covariance_update
        del self.state_prediction_derivative_update

        del self.state_estimation_mean_update
        del self.state_estimation_covariance_update

        del self.state_flush_updates
        del self.sensor_flush_updates

    def build_simulation_function(self):
        if self.state_prediction_mean_update is None:
            print("Building dynamics engine simulation updates. This may take a while depending on how complex your model is.")
            self.build_loads()
            debugger = utilities.DebugTensorLogger(self.DEBUG_VERBOSITY)
            self.state_prediction_debugger = debugger

            self.control_tensors = [self.controllers[controller].get_control_tensor() for controller in
                                    self.controllers]

            # Build state data
            self.state_list, state_derivative_list, state_vector = self._build_states_and_derivatives(state_order=self.state_list)
            debugger.add_tensor(state_vector, "input state prediction mean", 1)
            previous_state_covariance = ifelse.ifelse(T.eq(self.state_covariance.shape[0], 1), T.zeros((state_vector.shape[0],state_vector.shape[0])), self.state_covariance)
            debugger.add_tensor(previous_state_covariance, "input state prediction covariance", 1)

            # Run state prediction
            self.state_prediction_mean_update, \
            self.state_prediction_derivative_update, \
            self.state_prediction_covariance_update,\
            self.state_prediction_control_derivative_update = self._build_prediction(
                state_derivative_list,
                state_vector,
                previous_state_covariance,
                debugger=debugger
            )
            debugger.add_tensor(self.state_prediction_mean_update, "state prediction mean", 1)
            debugger.add_tensor(self.state_prediction_derivative_update, "state prediction derivative", 2)
            debugger.add_tensor(self.state_prediction_covariance_update, "state prediction covariance", 1)
            debugger.add_tensor(self.state_prediction_control_derivative_update, "state prediction control derivative", 1)

        print("Building dynamics engine simulation function. This may take a while depending on how complex your model is.")
        # Simulation update function
        state_updates = ([
            (self.state_mean, T.unbroadcast(self.state_prediction_mean_update.dimshuffle(0, 'x'), 1)),
            (self.state_derivative, self.state_prediction_derivative_update),
            (self.state_covariance, self.state_prediction_covariance_update),
            (self.control_derivative, self.state_prediction_control_derivative_update)
        ])
        for shared_var, update in state_updates:
            if shared_var.type != update.type:
                print("we have a problem")
        state_updates.extend(self.state_prediction_debugger.get_updates())
        self.simulation_func = theano.function([], [], updates=state_updates)

    def build_estimation_function(self):
        if self.state_estimation_mean_update is None:
            print("Building dynamics engine estimation updates. This may take a bit depending on how complex your model is.")
            if len(self.sensor_prediction_list) == 0:
                self.state_estimation_covariance_update = self.state_covariance
                self.state_estimation_mean_update = T.addbroadcast(self.state_mean, 1)
            else:
                # What we think the sensor values should be
                sensor_prediction = T.stack(self.sensor_prediction_list)

                # Derivative of the sensor prediction with respect to the state prediction
                _, sensor_state_derivative = utilities.get_list_derivative(self.sensor_prediction_list, self.state_list)
                # Covariance of the predicted sensor values
                sensor_covariance = utilities.get_covariance_matrix_from_object_dict(
                    sensor_prediction, self.sensors, {sensor_state_derivative: self.state_covariance}
                )
                sensor_values = T.stack(self.sensor_value_list)

                # Run state prediction
                self.state_estimation_mean_update, self.state_estimation_covariance_update = self._build_estimation(
                    sensor_prediction,
                    sensor_values,
                    sensor_covariance,
                    sensor_state_derivative,
                    self.state_mean,
                    self.state_covariance
                )
        print("Building dynamics engine estimation function. This may take a bit depending on how complex your model is.")
        # Estimation update function
        state_updates = ([
            (self.state_mean, T.unbroadcast(self.state_estimation_mean_update.dimshuffle(0, 'x'), 1)),
            (self.state_covariance, self.state_estimation_covariance_update)
        ])
        self.estimation_func = theano.function([], [], updates=state_updates)

    def build_optimization_function(self):
        if self.feedback_gains_progress_update is None:
            # Index our controllers into a list
            for controller in self.controllers:
                self.controller_list.append(controller)
            # Step forward in time from the current state over self.feedback_gains_clock
            result, updates = theano.scan(
                    fn=self._simulation_scan_func,
                    sequences=[
                        dict(input=self.feedback_gains_clock, taps=[-1, 0]),
                        self.feedback_gains
                    ],
            )

    # This inner scan function steps the simulation forward in time from prev_time to new_time each iteration
    def _simulation_scan_func(self, prev_time, new_time, feedback_gains, state_mean_covariance, state_covariance_covariance):
        # Hard-set dt
        self.dt = new_time - prev_time

        feedback_state = T.concatenate((self.state_estimation_mean_update, self.state_estimation_covariance_update.flatten()))

        # Hard-set controllers to respond to feedback gains
        for i in range(len(self.controller_list)):
            self.controller_list[i].percent_vbus = T.dot(feedback_gains[i], feedback_state)

        self.state_list, state_derivative_list, state_vector = self._build_states_and_derivatives(state_order=self.state_list)

        # Get new predicted state
        self.state_prediction_mean_update, \
        self.state_prediction_derivative_update, \
        self.state_prediction_covariance_update,\
        self.state_prediction_control_derivative_update = self._build_prediction(state_derivative_list, state_vector, self.state_covariance)

        # Get predicted sensor values
        for sensor in self.sensors:
            value_predictions = self.sensors[sensor].get_value_prediction()
            for sensor_value in value_predictions:
                self.sensor_value_list.append(sensor_value)
                self.sensor_prediction_list.append(value_predictions[sensor_value])

        # What we think the sensor values should be
        sensor_prediction = T.stack(self.sensor_prediction_list)

        # Derivative of the sensor prediction with respect to the state prediction
        _, sensor_state_derivative = utilities.get_list_derivative(self.sensor_prediction_list, self.state_list)
        # Covariance of the predicted sensor values
        sensor_covariance = utilities.get_covariance_matrix_from_object_dict(
            sensor_prediction, self.sensors, {sensor_state_derivative: self.state_prediction_covariance_update}
        )
        sensor_values = sensor_prediction + 0

        # Run state estimation, but with substituting the sensor prediction for the sensor value
        # The reason for this is to simulate the best case scenario, adding in covariance later
        self.state_estimation_mean_update, self.state_estimation_covariance_update = self._build_estimation(
            sensor_prediction,
            sensor_values,
            sensor_covariance,
            sensor_state_derivative,
            self.state_prediction_mean_update,
            self.state_prediction_covariance_update
        )

        _, state_estimation_mean_sensor_value_derivative = utilities.get_list_derivative(self.state_estimation_mean_update, sensor_values)
        _, state_estimation_covariance_sensor_value_derivative = utilities.get_list_derivative(self.state_estimation_covariance_update_update, sensor_values)

        state_mean_covariance = utilities.get_covariance_matrix({
            state_estimation_mean_sensor_value_derivative: sensor_covariance,
        })
        state_covariance_covariance = utilities.get_covariance_matrix({
            state_estimation_covariance_sensor_value_derivative: sensor_covariance,
        })

        # Now we update the worst-case scenario, calculating the covariance of both state mean and covariance
        pesssimistic_state_covariance = utilities.get_covariance_matrix()

        updates = self._build_state_updates(self.state_estimation_mean_update)
        updates.extend([
            (self.state_mean, self.state_estimation_mean_update),
            (self.state_covariance, self.state_estimation_covariance_update)
        ])



        previous_cost = T.sum([cost.get_cost() for cost in self.costs])
        return [self.state_estimation_mean_update, self.state_estimation_covariance_update, previous_cost, feedback_state], updates

    def build_state_flush_function(self):
        if len(self.state_flush_updates) == 0:
            self.state_flush_updates = self._build_state_updates(self.state_mean)
        self.state_flush_func = theano.function([], [], updates=self.state_flush_updates)

    def build_sensor_flush_function(self):
        if len(self.sensor_flush_updates) < 0:
            for sensor in self.sensors:
                value_predictions = self.sensors[sensor].get_value_prediction()
                for sensor_value in value_predictions:
                    self.sensor_value_list.append(sensor_value)
                    self.sensor_prediction_list.append(value_predictions[sensor_value])
            # Sensor value update function
            self.sensor_flush_updates = [(value, prediction) for value, prediction in zip(self.sensor_value_list, self.sensor_prediction_list)]
        self.sensor_flush_func = theano.function([], [], updates=self.sensor_flush_updates)

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
            state_derivatives = OrderedDict(sorted(state_derivatives, key=lambda t: state_order.index(t) if t in state_order else 100))

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

    def _build_prediction(self, state_derivative_list, state_vector, state_covariance, debugger=None):
        """
        Build the state prediction for self.dt seconds into the future
        :param state_derivative_list: A list of derivatives of the various state values with respect to time.
        :param state_vector: A vector of the current predicted state.
        :param state_covariance: The current covariance of the state vector.

        :return The new predicted state vector.
        :return The new derivative of the predicted state with respect to the last state.
        :return The new covariance of the state vector.
        """

        derivative_matrix, A = utilities.get_list_derivative(state_derivative_list, self.state_list)
        b = derivative_matrix - T.dot(A, state_vector)
        self.state_prediction_A = A
        self.state_prediction_b = b
        if False and debugger is not None:
            _, A_deriv = utilities.get_list_derivative(A.flatten(), self.state_list)
            debugger.add_tensor(A, "ODE A matrix", 2)
            debugger.add_tensor(A_deriv, "ODE A matrix derivative", 2)
            _, b_deriv = utilities.get_list_derivative(b.flatten(), self.state_list)
            debugger.add_tensor(b, "ODE b matrix", 2)
            debugger.add_tensor(b_deriv, "ODE b matrix derivative", 2)

        # Equation given by http://math.stackexchange.com/a/1567806/294141
        # Taylor series method
        init_term = T.identity_like(A)*self.dt

        def series_advance(i, last_term, A, dt):
            next_term = T.dot(last_term, A)*dt/i
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-6))
        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 100)],
                               non_sequences=[A, self.dt],
                               outputs_info=init_term,
                               )
        integral = T.sum(terms, axis=0) + init_term

        # The mean prediction of the new state
        prediction_mean = (T.dot(slinalg.expm(A*self.dt), state_vector) + T.dot(integral, b)).flatten()
        self.state_prediction_mean_update = prediction_mean

        # Derivative of the new state with respect to last state
        prediction_derivative = self.build_prediction_derivative(prediction_mean, self.state_list)
        # prediction_derivative = utilities.replace_nans(prediction_derivative)

        control_derivative = self.build_prediction_derivative(prediction_mean, self.control_tensors)

        # Covariance of the new state
        source_derivatives = {prediction_derivative: state_covariance}
        for load in self.loads:
            variance_data = self.loads[load].get_variance_sources()
            for variance_source in variance_data:
                variance_derivative = self.build_prediction_derivative(prediction_mean, variance_source)#.dimshuffle(0, 'x')
                # variance_derivative = replace_nans(variance_derivative)
                if debugger is not None:
                    # debugger.add_tensor(variance_data[variance_source], "{} variance data".format(key), 2)
                    debugger.add_tensor(variance_derivative, "load {} variance derivative".format(load), 2)
                source_derivatives[variance_derivative] = variance_data[variance_source]
        prediction_covariance = utilities.get_covariance_matrix(source_derivatives)

        return prediction_mean, prediction_derivative, prediction_covariance, control_derivative

    def build_prediction_derivative(self, expressions, wrt):
        print("getting derivative!")
        if expressions is self.state_prediction_mean_update:
            dexp_dstate = 1
        else:
            print("computing dexp/dstate")
            _, dexp_dstate = utilities.get_list_derivative(expressions, self.state_prediction_mean_update)
        for cached_wrt, dstate_dwrt in self.state_prediction_derivatives:
            if wrt is cached_wrt:
                break
        else:
            for cached_wrt, dstate_dA in self.state_prediction_derivatives:
                if self.state_prediction_A is cached_wrt:
                    break
            else:
                print("computing dstate/dA")
                dstate_dA = theano.gradient.jacobian(self.state_prediction_mean_update, self.state_prediction_A, disconnected_inputs='ignore').flatten(ndim=2)
                self.state_prediction_derivatives.append((self.state_prediction_A, dstate_dA))
            for cached_wrt, dstate_db in self.state_prediction_derivatives:
                if self.state_prediction_b is cached_wrt:
                    break
            else:
                print("computing dstate/db")
                dstate_db = theano.gradient.jacobian(self.state_prediction_mean_update, self.state_prediction_b, disconnected_inputs='ignore').flatten(ndim=2)
                self.state_prediction_derivatives.append((self.state_prediction_b, dstate_db))
            for cached_wrt, dA_dwrt in self.state_prediction_A_derivatives:
                if wrt is cached_wrt:
                    break
            else:
                print("computing dA/dwrt")
                dA_dwrt, _ = utilities.get_list_derivative(self.state_prediction_A.flatten(), wrt)
                self.state_prediction_A_derivatives.append((wrt, dA_dwrt))
            for cached_wrt, db_dwrt in self.state_prediction_b_derivatives:
                if wrt is cached_wrt:
                    break
            else:
                print("computing db/dwrt")
                db_dwrt, _ = utilities.get_list_derivative(self.state_prediction_b.flatten(), wrt)
                self.state_prediction_b_derivatives.append((wrt, db_dwrt))
            dstate_dwrt_A = T.dot(dstate_dA, dA_dwrt)
            dstate_dwrt_b = T.dot(dstate_db, db_dwrt)
            dstate_dwrt = dstate_dwrt_A + dstate_dwrt_b
            self.state_prediction_derivatives.append((wrt, dstate_dwrt))
        return T.dot(dexp_dstate, dstate_dwrt)

    def _build_estimation(self, sensor_prediction, sensor_values, sensor_covariance, sensor_derivative, state_mean, state_covariance):
        # From the state prediction and the sensor data, we get the state estimation via a kalman filter
        kalman = T.dot(T.dot(state_covariance, sensor_derivative.T), T.inv(sensor_covariance))
        estimation_mean = state_mean + T.dot(kalman, sensor_values - sensor_prediction)
        estimation_covariance = \
            T.dot(
                T.identity_like(state_covariance) - T.dot(kalman, sensor_derivative),
                state_covariance
            )

        return estimation_mean, estimation_covariance

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
        if self.SINK_TO_SIMPLESTREAMER:
            if self.streamer is None:
                if self.mode == "simulation":
                    self.streamer = simplestreamer.SimpleStreamer(self.SS_SIM_PORT)
                elif self.mode == "estimation":
                    self.streamer = simplestreamer.SimpleStreamer(self.SS_EST_PORT)
                else:
                    self.streamer = simplestreamer.SimpleStreamer(5801)
            self.streamer.send_data(state_data)

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

    def simulation_update(self, dt, hal_data=None, resolve_error=False):
        start_time = time.time()
        if hal_data is not None:
            for controller in self.controllers:
                self.controllers[controller].set_from_hal_data(hal_data, dt)
        self.dt.set_value(dt)
        self.simulation_func()
        self.state_prediction_debugger.do_checkup(max_magnitude=-1)
        if resolve_error:
            covariance = self.state_covariance.get_value()
            new_state = utilities.sample_covariance_numpy(self.state_mean.get_value()[:, 0], covariance)
            self.state_mean.set_value(new_state[:, None])
            self.state_covariance.set_value(np.zeros_like(covariance))
        self.state_flush_func()
        self.sensor_flush_func()
        if hal_data is not None:
            for sensor in self.sensors:
                self.sensors[sensor].update_hal_data(hal_data, dt)
        self.tic_time = time.time() - start_time
        if self.SINK_IN_SIMULATION:
            self.sink_state_data()

    def estimation_update(self, dt):
        start_time = time.time()
        self.dt.set_value(dt)
        self.simulation_func()

        self.poll_sensors()
        self.estimation_func()
        self.state_flush_func()
        self.update_controllers()

        self.tic_time = time.time() - start_time
        self.sink_state_data()

    def optimization_update(self):
        start_time = time.time()

        self.feedback_gain_progression_func()

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

    def update_controllers(self):
        for controller in self.controllers:
            self.controllers[controller].update_device()

    def set_controllers(self, control):
        for i in range(len(self.control_tensors)):
            self.control_tensors[i].set_value(control[i])

    def get_state(self):
        state = {}
        for component in self.loads:
            state[component] = self.loads[component].get_state()
        return state

    def __sizeof__(self):
        return object.__sizeof__(self) + \
            sum(sys.getsizeof(v) for v in self.__dict__.values())
