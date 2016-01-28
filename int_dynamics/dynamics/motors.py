import theano
import math


class Motor:
    """
    Simulates the dynamics of any number of motors connected to a common power supply
    """

    def __init__(self, power_supply, free_rps_per_volt, stall_torque_per_volt):
        self.free_rps_per_volt = free_rps_per_volt
        self.stall_torque_per_volt = stall_torque_per_volt
        self.power_supply = power_supply
        self.velocity = theano.shared(0.0, theano.config.floatX)

    def get_state_derivatives(self, load_moment):
        voltage_in = self.power_supply.voltage_out
        stall_torque = self.stall_torque_per_volt*voltage_in
        torque = stall_torque - self.velocity * self.stall_torque_per_volt/self.free_rps_per_volt
        # rot/sec^2 = (T/mom)/(2*math.pi)
        return {
            self.velocity: torque/load_moment/(2*math.pi)
        }


class CIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 7.37, .149)


class MiniCIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 8.61, .086)


class DualCIMMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 7.37, .298)


class BAGMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 19.4, .0243)


class ThrottleMotor(Motor):
    def __init__(self, power_supply):
        Motor.__init__(self, power_supply, 7.37, .00798)
