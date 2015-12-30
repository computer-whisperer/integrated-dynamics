from components import DynamicsComponent


class Motor(DynamicsComponent):
    """
    Simulates the dynamics of any number of motors connected to a common power supply
    """

    def __init__(self, power_supply, free_rps_per_volt, stall_torque_per_volt):
        self.free_rps_per_volt = free_rps_per_volt
        self.stall_torque_per_volt = stall_torque_per_volt
        super().__init__([power_supply])

    def get_force_tensor(self, load_state):
        voltage_in = self.get_input_force_tensor(load_state)
        torque_out = self.stall_torque_per_volt*voltage_in + \
                     load_state[1] * -self.stall_torque_per_volt/self.free_rps_per_volt
        return torque_out

class CIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 7.37, 1.75)

class DualCIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 14.73, 3.5)
