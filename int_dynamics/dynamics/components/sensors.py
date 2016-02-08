import theano
import math
import numpy as np


class Sensor:

    def __init__(self):
        self.device = None

    def get_state(self):
        pass

    def update_hal_data(self, hal_data, dt, add_noise=False):
        """
        Updates hal_data with the latest encoder data.
        """
        pass

    def init_device(self):
        """
        Initialize the wpilib device for this sensor
        """
        pass

    def poll_sensor(self):
        pass

    def get_sensor_data(self):
        return {}


class Encoder(Sensor):

    def __init__(self, gearbox, a_channel=None, b_channel=None, tics_per_rev=360, noise=.1):
        self.gearbox = gearbox
        self.position = theano.shared(0.0, theano.config.floatX)
        self.velocity = theano.shared(0.0, theano.config.floatX)
        self.a_channel = a_channel
        self.b_channel = b_channel
        self.tics_per_rev = tics_per_rev
        self.noise = noise
        super().__init__()

    def get_state(self, add_noise=False):
        """
        :returns a dictionary of state values from the sensor.
        State values are "position" and "velocity" in units of rotations and rotations/second
        """
        pos = self.position.get_value()
        vel = self.velocity.get_value()
        if add_noise:
            pos += np.random.normal(0, self.noise)
            vel += np.random.normal(0, self.noise)
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

    def init_device(self):
        import wpilib
        self.device = wpilib.Encoder(self.a_channel, self.b_channel)
        self.device.setDistancePerPulse(1/self.tics_per_rev)

    def poll_sensor(self):
        self.position.set_value(self.device.getDistance())
        self.position.set_value(self.device.getRate())

    def get_sensor_data(self):
        return {
            self.position: {"update": self.gearbox.position, "covariance": 0},
            self.velocity: {"update": self.gearbox.velocity, "covariance": 0}
        }


class CANTalonEncoder(Encoder):

    def set_talon(self, talon, can_id):
        self.talon = talon
        self.can_id = can_id

    def update_hal_data(self, hal_data, dt, add_noise=False):
        state = self.get_state(add_noise)
        hal_data['CAN'][self.can_id]['enc_position'] = state["position"]*self.tics_per_rev
        hal_data['CAN'][self.can_id]['enc_velocity'] = state["velocity"]*self.tics_per_rev
        hal_data['CAN'][self.can_id]['sensor_position'] = state["position"]*self.tics_per_rev
        hal_data['CAN'][self.can_id]['sensor_velocity'] = state["velocity"]*self.tics_per_rev

    def init_device(self):
        pass

    def poll_sensor(self):
        self.position.set_value(self.talon.getPosition()/self.tics_per_rev)
        self.velocity.set_value(self.talon.getVelocity()/self.tics_per_rev)


class AnalogGyro(Sensor):

    def __init__(self, twodimensionalload, analog_channel, noise=.1):
        self.load = twodimensionalload
        self.angle = theano.shared(0.0, theano.config.floatX)
        self.channel = analog_channel
        self.noise = noise
        super().__init__()

    def update_hal_data(self, hal_data, dt, add_noise=False):
        angle = self.angle.get_value()
        if add_noise:
            angle += np.random.normal(0, self.noise)
        hal_data['analog_in'][self.channel]['accumulator_value'] = math.degrees(angle) / 2.7901785714285715e-12

    def init_device(self):
        import wpilib
        self.device = wpilib.AnalogGyro(self.channel)

    def poll_sensor(self):
        self.angle.set_value(math.radians(self.device.getAngle()))

    def get_sensor_data(self):
        return {
            self.angle: {"update": self.load.position[2], "covariance": self.noise}
        }


class NavX(Sensor):
    def __init__(self, twodimensionalload, do_accel=False, gyro_noise=0.1, accel_noise=0.1):
        self.load = twodimensionalload
        self.angle = theano.shared(0.0, theano.config.floatX)
        self.accel_x = theano.shared(0.0, theano.config.floatX)
        self.accel_y = theano.shared(0.0, theano.config.floatX)
        self.gyro_noise = gyro_noise
        self.accel_noise = accel_noise
        self.do_accel = do_accel
        super().__init__()

    def update_hal_data(self, hal_data, dt, add_noise=False):
        angle = self.angle.get_value()
        if add_noise:
            angle += np.random.normal(0, self.gyro_noise)
        hal_data['robot']['navxmxp_i2c_1_angle'] = math.degrees(self.angle.get_value())

    def init_sensor(self):
        from robotpy_ext.common_drivers import navx
        self.device = navx.AHRS.create_spi()

    def poll_sensor(self):
        self.angle.set_value(math.radians(self.device.getYaw()))
        if self.do_accel:
            self.accel_x.set_value(self.device.getRawAccelX() * 32)
            self.accel_y.set_value(self.device.getRawAccelY() * 32)

    def get_sensor_data(self):
        data = {self.angle: {"update": self.load.position[2], "covariance": self.gyro_noise}}
        if self.do_accel:
            data[self.accel_x] = {"update": self.load.local_accel[0], "covariance": self.accel_noise}
            data[self.accel_y] = {"update": self.load.local_accel[1], "covariance": self.accel_noise}
        return data
