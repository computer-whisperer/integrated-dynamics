from components import DynamicsComponent
import theano
import theano.tensor as T
from theano.ifelse import ifelse

class Motor(DynamicsComponent):
    """
    Simulates the dynamics of any number of motors connected to a common power supply
    """

    def __init__(self, power_supply, free_rps_per_volt, stall_torque_per_volt):
        self.free_rps_per_volt = free_rps_per_volt
        self.stall_torque_per_volt = stall_torque_per_volt
        super().__init__([power_supply])
        self.state = {
            "velocity": theano.shared(0.0, theano.config.floatX)
        }

    def get_force_tensor(self):
        voltage_in = self.get_input_force_tensor()
        torque_out = self.stall_torque_per_volt*voltage_in + \
                     self.state_tensors["velocity"] * -self.stall_torque_per_volt/self.free_rps_per_volt
        return torque_out

    def build_state_tensors(self, travel, velocity, dt):
        self.state_tensors["velocity"] = velocity

class CIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 7.37, 1.75)

class DualCIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 14.73, 3.5)
