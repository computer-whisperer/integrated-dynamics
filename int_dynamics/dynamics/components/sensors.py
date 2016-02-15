import theano
import math
import numpy as np


class Sensor:

    def __init__(self):
        self.device = None

    def get_state(self, add_noise=False):
        pass

    def update_hal_data(self, hal_data, dt, add_noise=False):
        """
        Updates hal_data with the latest encoder data.
        """
        pass

    def set_device_object(self, device):
        """
        Set the wpilib device reference for this sensor
        """
        self.device = device

    def poll_sensor(self):
        pass

    def get_value_prediction(self):
        return {}

    def get_variance(self):
        return {}


class Encoder(Sensor):

    def __init__(self, gearbox, a_channel=None, b_channel=None, tics_per_rev=360, variance=.001):
        self.gearbox = gearbox
        self.position = theano.shared(0.0, theano.config.floatX)
        self.velocity = theano.shared(0.0, theano.config.floatX)
        self.a_channel = a_channel
        self.b_channel = b_channel
        self.tics_per_rev = tics_per_rev
        self.variance = variance
        super().__init__()

    def get_state(self, add_noise=False):
        """
        :returns a dictionary of state values from the sensor.
        State values are "position" and "velocity" in units of rotations and rotations/second
        """
        pos = self.position.get_value()
        vel = self.velocity.get_value()
        if add_noise:
            pos += np.random.normal(0, math.sqrt(self.variance))
            vel += np.random.normal(0, math.sqrt(self.variance))
        return {
            "position": pos,
            "velocity": vel
        }

    def update_hal_data(self, hal_data, dt, add_noise=False):
        state = self.get_state(add_noise)
        for encoder in hal_data['encoder']:
            if encoder['config'].get('ASource_Channel', None) == self.a_channel:
                encoder['count'] = state["position"]*self.tics_per_rev
                return

    def poll_sensor(self):
        self.position.set_value(self.device.getDistance())
        self.position.set_value(self.device.getRate())

    def get_value_prediction(self):
        return {
            self.position: self.gearbox.position,
            self.velocity: self.gearbox.velocity
        }

    def get_variance(self):
        return {
            self.position: self.variance
        }


class CANTalonEncoder(Encoder):

    def set_can_id(self, can_id):
        self.can_id = can_id

    def update_hal_data(self, hal_data, dt, add_noise=False):
        state = self.get_state(add_noise)
        if self.can_id in hal_data['CAN']:
            hal_data['CAN'][self.can_id]['enc_position'] = state["position"]*self.tics_per_rev
            hal_data['CAN'][self.can_id]['enc_velocity'] = state["velocity"]*self.tics_per_rev
            hal_data['CAN'][self.can_id]['sensor_position'] = state["position"]*self.tics_per_rev
            hal_data['CAN'][self.can_id]['sensor_velocity'] = state["velocity"]*self.tics_per_rev

    def poll_sensor(self):
        self.position.set_value(self.device.getPosition()/self.tics_per_rev)
        self.velocity.set_value(self.device.getVelocity()/self.tics_per_rev)


class AnalogGyro(Sensor):

    def __init__(self, twodimensionalload, analog_channel, variance=.001):
        self.load = twodimensionalload
        self.angle = theano.shared(0.0, theano.config.floatX)
        self.channel = analog_channel
        self.variance = variance
        super().__init__()

    def update_hal_data(self, hal_data, dt, add_noise=False):
        angle = self.angle.get_value()
        if add_noise:
            angle += np.random.normal(0, math.sqrt(self.variance))
        hal_data['analog_in'][self.channel]['accumulator_value'] = math.degrees(angle) / 2.7901785714285715e-12

    def init_device(self):
        import wpilib
        self.device = wpilib.AnalogGyro(self.channel)

    def poll_sensor(self):
        self.angle.set_value(math.radians(self.device.getAngle()))

    def get_value_prediction(self):
        return {
            self.angle: self.load.position[2]
        }

    def get_variance(self):
        return {
            self.angle: self.variance
        }


class NavX(Sensor):
    def __init__(self, twodimensionalload, do_accel=False, gyro_variance=0.001, accel_variance=0.01):
        self.load = twodimensionalload
        self.angle = theano.shared(0.0, theano.config.floatX)
        self.accel_x = theano.shared(0.0, theano.config.floatX)
        self.accel_y = theano.shared(0.0, theano.config.floatX)
        self.gyro_variance = gyro_variance
        self.accel_variance = accel_variance
        self.do_accel = do_accel
        super().__init__()

    def update_hal_data(self, hal_data, dt, add_noise=False):
        angle = self.angle.get_value()
        if add_noise:
            angle += np.random.normal(0, math.sqrt(self.gyro_variance))
        hal_data['robot']['navxmxp_i2c_1_angle'] = math.degrees(self.angle.get_value())

    def init_sensor(self):
        from robotpy_ext.common_drivers import navx
        self.device = navx.AHRS.create_spi()

    def poll_sensor(self):
        self.angle.set_value(math.radians(self.device.getYaw()))
        if self.do_accel:
            self.accel_x.set_value(self.device.getRawAccelX() * 32)
            self.accel_y.set_value(self.device.getRawAccelY() * 32)

    def get_value_prediction(self):
        data = {self.angle: self.load.position[2]}
        if self.do_accel:
            data[self.accel_x] = self.load.local_accel[0]
            data[self.accel_y] = self.load.local_accel[1]
        return data

    def get_variance(self):
        data = {self.angle: self.gyro_variance}
        if self.do_accel:
            data[self.accel_x] = self.accel_variance
            data[self.accel_y] = self.accel_variance
        return data
