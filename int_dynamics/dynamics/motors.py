import theano


class Motor:
    """
    Simulates the dynamics of any number of motors connected to a common power supply
    """

    def __init__(self, power_supply, free_rps_per_volt, stall_torque_per_volt):
        self.free_rps_per_volt = free_rps_per_volt
        self.stall_torque_per_volt = stall_torque_per_volt
        self.power_supply = power_supply
        self.velocity = theano.shared(0.0, theano.config.floatX)

    def get_state_derivatives(self, load_mass):
        voltage_in = self.power_supply.voltage_out
        stall_torque = self.stall_torque_per_volt*voltage_in
        torque = stall_torque - self.velocity * self.stall_torque_per_volt/self.free_rps_per_volt
        return {
            self.velocity: torque/load_mass
        }


class CIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 7.37, 1.79)


class DualCIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 7.37, 3.58)
