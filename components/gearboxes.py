from components import DynamicsComponent
import theano


class GearBox(DynamicsComponent):
    """
    Simulates the dynamics of a gearbox with one or more motors attached
    """

    def __init__(self, motors, gear_ratio=10, dynamic_friction=10):
        self.friction = dynamic_friction
        self.gear_ratio = gear_ratio
        super().__init__(motors)
        self.state = {
            "position": theano.shared(0.0, theano.config.floatX),
            "velocity": theano.shared(0.0, theano.config.floatX)
        }

    def get_force_tensor(self, load_state):
        self.load_state = load_state
        torque_in = self.get_input_force_tensor(load_state*self.gear_ratio)
        return torque_in*self.gear_ratio

    def build_state_updates(self):
        self.state_derivatives = {
            "position": self.load_state[1],
        }
        self.state_updates = {
            "velocity": self.load_state[1]
        }
        super().build_state_updates()
