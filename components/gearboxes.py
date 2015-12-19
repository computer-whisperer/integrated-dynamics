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

    def get_force_tensor(self):
        torque_in = self.get_input_force_tensor()
        return torque_in*self.gear_ratio

    def build_state_tensors(self, travel, velocity):
        self.state_tensors = {
            "position": self.state["position"] + travel,
            "velocity": velocity
        }
        self.build_input_state_tensors(travel*self.gear_ratio, velocity*self.gear_ratio)
