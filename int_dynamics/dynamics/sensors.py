import theano


class Encoder:

    def __init__(self, gearbox):
        self.gearbox = gearbox
        self.position = theano.shared(0.0, theano.config.floatX)
        self.velocity = theano.shared(0.0, theano.config.floatX)

    def get_sensor_data(self):
        return {
            self.position: {"update": self.gearbox.position, "covariance": 0},
            self.velocity: {"update": self.gearbox.velocity, "covariance": 0}
        }


class Gyro:

    def __init__(self, twodimensionalload):
        self.load = twodimensionalload
        self.angle = theano.shared(0.0, theano.config.floatX)

    def get_sensor_data(self):
        return {
            self.angle: {"update": self.load.position[2], "covariance": .05}
        }


class NavX:
    def __init__(self, twodimensionalload):
        self.load = twodimensionalload
        self.angle = theano.shared(0.0, theano.config.floatX)
        self.accel_x = theano.shared(0.0, theano.config.floatX)
        self.accel_y = theano.shared(0.0, theano.config.floatX)

    def get_sensor_data(self):
        return {
            self.angle: {"update": self.load.position[2], "covariance": .05},
            self.accel_x: {"update": self.load.local_accel[0], "covariance": .05},
            self.accel_y: {"update": self.load.local_accel[1], "covariance": .05}
        }
