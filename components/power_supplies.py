import theano

class SpeedController:
    """
    Simulates the dynamics of a standard speed controller
    """
    value = theano.shared(0.0)

    def set_value(self, value):
        self.value.set_value(max(-1, min(1, value)))

    def get_tensors(self, tensors_in):
        return {
            "voltage": self.value*12
        }
