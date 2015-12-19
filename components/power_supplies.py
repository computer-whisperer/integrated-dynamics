import theano
from components import DynamicsComponent

class SpeedController(DynamicsComponent):
    """
    Simulates the dynamics of a standard speed controller
    """
    def __init__(self):
        self.percent_vbus = theano.shared(0.0, theano.config.floatX)
        super().__init__()

    def set_value(self, value):
        self.percent_vbus.set_value(max(-1, min(1, value)))

    def get_force_tensor(self):
        return self.percent_vbus*12
