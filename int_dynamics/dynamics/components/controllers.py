import theano
import theano.tensor as T
import numpy as np


class SpeedController:
    """
    Simulates the dynamics of a standard speed controller
    """
    def __init__(self, motor, noise=0.0001):
        self.percent_vbus = theano.shared(0.0, theano.config.floatX)
        self.voltage_out = self.percent_vbus*12
        motor.voltage_in = self.voltage_out
        self.noise = noise
        self.feedback_state_vector = None
        self.device = False

    def set_percent_vbus(self, value, add_noise=False):
        if add_noise:
            value += np.random.normal(0, self.noise)
        self.percent_vbus.set_value(max(-1, min(1, value)))

    def set_from_hal_data(self, hal_data, dt, add_noise=False):
        pass

    def set_feedback_state_vector(self, state_vector):
        self.feedback_state_vector = state_vector

    def init_device(self):
        pass

    def get_state(self):
        return {
            "percentVbus": self.percent_vbus.get_value()
        }

    def update_device(self):
        pass


class PWMSpeedController(SpeedController):
    """
    Simulates the dynamics of a standard speed controller
    """
    def __init__(self, motor, channel, noise=0.0001):
        self.channel = channel
        self.wpilib_type = None
        super().__init__(motor, noise)

    def set_from_hal_data(self, hal_data, dt, add_noise=False):
        if hal_data['control']['enabled']:
            percent_vbus = hal_data['pwm'][self.channel]['value']
        else:
            percent_vbus = 0
        self.set_percent_vbus(percent_vbus)

    def init_device(self):
        if self.wpilib_type is None:
            import wpilib
            self.wpilib_type = wpilib.Talon
        self.device = self.wpilib_type(self.channel)

    def update_device(self):
        self.device.set(self.percent_vbus.get_value())


class CANTalonSpeedController(SpeedController):
    """
    Simulates the dynamics of a Talon SRX speed controller

    This simulation currently covers the following modes: percentVbus, voltage, position, and velocity.
    """
    def __init__(self, motor, can_id, noise=0.0001):
        self.can_id = can_id
        self.mode = 'percent_vbus'
        self.last_vel = 0
        self.sensor = False

        super().__init__(motor, noise)

    def add_encoder(self, encoder):
        self.sensor = encoder
        encoder.set_can_id(self.can_id)

    def set_from_hal_data(self, hal_data, dt, add_noise=False):
        if self.can_id not in hal_data['CAN']:
            return
        import hal
        class Handle:
            id = self.can_id
        sensor_pos = hal.TalonSRX_GetSensorPosition(Handle())
        sensor_vel = hal.TalonSRX_GetSensorVelocity(Handle())
        talon = hal_data['CAN'][self.can_id]
        percent_vbus = 0
        if hal_data.get('profile_slot_select', 0) == 0:
            p_gain = talon['profile0_p']
            i_gain = talon['profile0_i']
            d_gain = talon['profile0_d']
            f_gain = talon['profile0_f']
            izone = talon['profile0_izone']
            close_loop_ramp_rate = talon['profile0_closeloopramprate']
        else:
            p_gain = talon['profile1_p']
            i_gain = talon['profile1_i']
            d_gain = talon['profile1_d']
            f_gain = talon['profile1_f']
            izone = talon['profile1_izone']
            close_loop_ramp_rate = talon['profile1_closeloopramprate']
        if hal_data['control']['enabled']:
            if talon['mode_select'] == 0: # Percent vbus mode
                percent_vbus = talon['value']
            elif talon['mode_select'] == 1: # Position PID mode
                if 'pid_iaccum' not in talon:
                    talon['pid_iaccum'] = 0
                talon['closeloop_err'] = talon['value'] - sensor_pos
                talon['pid_iaccum'] += dt*talon['closeloop_err']
                if izone != 0 and abs(talon['pid_iaccum']) > izone:
                    talon['pid_iaccum'] = 0
                output = p_gain*talon['closeloop_err']\
                         + i_gain*talon['pid_iaccum']\
                         - d_gain*sensor_vel \
                         + f_gain*talon['value']
                # Output is -1023 to 1023
                percent_vbus = output/1023
            elif talon['mode_select'] == 2:
                if 'pid_accum' not in talon:
                    talon['pid_iaccum'] = 0
                talon['closeloop_err'] = talon['value'] - sensor_vel
                talon['pid_iaccum'] += dt*talon['closeloop_err']
                if izone != 0 and abs(['pid_iaccum']) > izone:
                    talon['pid_iaccum'] = 0
                output = p_gain*talon['closeloop_err']\
                         + i_gain*talon['pid_iaccum']\
                         - d_gain*(sensor_vel - self.last_vel)/dt \
                         + f_gain*talon['value']
                # Output is -1023 to 1023
                percent_vbus = output/1023
            elif talon['mode_select'] == 4:
                percent_vbus = talon['value']/12
        self.set_percent_vbus(percent_vbus, add_noise)
        self.last_vel = sensor_vel

    def init_device(self):
        import wpilib
        self.device = wpilib.CANTalon(self.can_id)

    def update_device(self):
        self.device.set(self.percent_vbus.get_value())


#class CANTalonSpeedControllerFeedback(CANTalonSpeedController):
#
#    def __init__(self, motor, can_id, noise=0.0001):
#        self.feedback = 0
#        self.feedforward = 0
#        self.state = theano.shared(np.array([[0]]), theano.config.floatX)
#        super().__init__(motor, can_id, noise)
#
#    def set_percent_vbus(self, value, add_noise=False):
#        raise ValueError("Cannot set percent vbus on feedback controller.")
#
#    def set_controller_gains(self, feedback, feedforward):
#        self.feedback = feedback
#        self.feedforward = feedforward
#        self.voltage_out = self.feedforward + T.sum(self.state*self.feedback)
#
#    def set_state_vector(self, state):
#        self.state = state
#
#    def update_device(self):
#        self.percent_vbus.set_value(self.feedforward + np.sum(self.state.get_value()*self.feedback))
#        super().update_device()
