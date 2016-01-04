import theano


class SpeedController:
    """
    Simulates the dynamics of a standard speed controller
    """
    def __init__(self):
        self.percent_vbus = theano.shared(0.0, theano.config.floatX)
        self.voltage_out = self.percent_vbus*12

    def set_value(self, value):
        self.percent_vbus.set_value(max(-1, min(1, value)))
