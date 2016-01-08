import theano


class Encoder:

    def __init__(self, gearbox):
        self.gearbox = gearbox
        self.position = theano.shared(0.0, theano.config.floatX)
        self.velocity = theano.shared(0.0, theano.config.floatX)

    def get_sensor_data(self):
        return {
            self.position: {"update": self.gearbox.position},
            self.velocity: {"update": self.gearbox.velocity}
        }


class Gyro:

    def __init__(self, twodimensionalload):
        self.load = twodimensionalload
        self.angle = theano.shared(0.0, theano.config.floatX)

    def get_sensor_data(self):
        return {
            self.angle: {"update": self.load.position[2]}
        }