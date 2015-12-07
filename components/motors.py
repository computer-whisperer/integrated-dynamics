import theano


class Motor:
    """
    Simulates the dynamics of any number of motors connected to a common power supply
    """

    def __init__(self, power_supply, free_rps_per_volt, stall_torque_per_volt):
        self.power_supply = power_supply
        self.free_rps_per_volt = free_rps_per_volt
        self.stall_torque_per_volt = stall_torque_per_volt
        self.state_functions = {}

    def get_tensors(self, tensors_in):
        self.state_functions = {
            "rps": theano.function([], tensors_in["rps"])
        }
        voltage_in = self.power_supply.get_tensors({})["voltage"]
        free_rps = self.free_rps_per_volt*voltage_in
        stall_torque = self.stall_torque_per_volt*voltage_in
        torque = stall_torque + tensors_in["rps"] * -stall_torque/free_rps
        return {
            "torque": torque
        }

    def update_state(self):
        pass

class CIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 7.37, 1.75)

class DualCIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 14.73, 3.5)
