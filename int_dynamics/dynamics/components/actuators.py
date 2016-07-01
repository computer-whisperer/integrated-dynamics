import math

import theano
from .. import base

class Motor(base.Actuator):
    """
    Simulates the dynamics of any number of motors connected to a common power supply
    """

    def __init__(self, free_rps_per_volt, stall_torque_per_volt):
        self.free_rps_per_volt = free_rps_per_volt
        self.stall_torque_per_volt = stall_torque_per_volt
        self.voltage_in = theano.shared(0.0, theano.config.floatX)
        self.velocity = theano.shared(0.0, theano.config.floatX)
        super().__init__([])

    def get_state_derivatives(self, load_moment):
        stall_torque = self.stall_torque_per_volt*self.voltage_in
        torque = stall_torque - self.velocity * self.stall_torque_per_volt/self.free_rps_per_volt
        # rot/sec^2 = (T/mom)/(2*math.pi)
        return {
            self.velocity: torque/load_moment/(2*math.pi)
        }


class CIMMotor(Motor):
    def __init__(self):
        Motor.__init__(self, 7.37, .149)


class MiniCIMMotor(Motor):
    def __init__(self):
        Motor.__init__(self, 8.61, .086)


class DualCIMMotor(Motor):
    def __init__(self):
        Motor.__init__(self, 7.37, .298)


class BAGMotor(Motor):
    def __init__(self):
        Motor.__init__(self, 19.4, .0243)


class RS775Motor(Motor):
    def __init__(self):
        Motor.__init__(self, 95, .0152)


class ThrottleMotor(Motor):
    def __init__(self):
        Motor.__init__(self, 7.37, .00798)