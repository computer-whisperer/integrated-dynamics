import theano


class GearBox:
    """
    Simulates the dynamics of a gearbox with one or more motors attached
    """

    def __init__(self, motors, gear_ratio=10, dynamic_friction=10):
        self.friction = dynamic_friction
        self.gear_ratio = gear_ratio
        self.motors = motors
        self.position = theano.shared(0.0, theano.config.floatX)
        self.velocity = theano.shared(0.0, theano.config.floatX)

    def get_state_derivatives(self, load_moment):
        for motor in self.motors:
            motor.velocity = self.velocity*self.gear_ratio
        state_derivatives = {
            self.velocity: 0,
            self.position: self.velocity
        }
        for motor in self.motors:
            state_derivatives.update(motor.get_state_derivatives(load_moment))
            state_derivatives[self.velocity] += state_derivatives[motor.velocity]*self.gear_ratio
        return state_derivatives
