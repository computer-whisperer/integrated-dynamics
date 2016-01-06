import math
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import slinalg
from theano.ifelse import ifelse
from .utilities import integrate_via_ode


class SimpleWheels:
    """
    Simulates the dynamics of a wheel without friction calculations
    """

    def __init__(self, gearbox, diameter):
        self.diameter = diameter/12
        self.gearbox = gearbox
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)

    def get_state_derivatives(self, load_mass):
        circumference = (math.pi * self.diameter)
        self.gearbox.velocity = self.velocity[1]/circumference

        state_derivatives = self.gearbox.get_state_derivatives(load_mass)
        state_derivatives[self.velocity] = np.array([0, 1])*state_derivatives[self.gearbox.velocity]*circumference
        return state_derivatives


class SolidWheels:
    """
    Simulates the dynamics of a wheel with friction calculations
    """

    def __init__(self, gearbox, count, diameter, static_cof, dynamic_cof, normal_force):
        self.diameter = diameter/12
        self.gearbox = gearbox
        self.mass = .25*count
        self.total_static_cof = normal_force*static_cof
        self.total_dynamic_cof = normal_force*dynamic_cof
        # Ground velocity
        self.velocity = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)
        # Difference between wheel surface velocity and ground velocity
        self.slip = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)

    def get_state_derivatives(self, load_mass):
        circumference = (math.pi * self.diameter)
        self.gearbox.velocity = (self.velocity[1] + self.slip[1])/circumference
        state_derivatives = self.gearbox.get_state_derivatives(load_mass)

        force_in = state_derivatives[self.gearbox.velocity]*circumference*load_mass

        force_out = T.clip(force_in, -self.total_static_cof, self.total_static_cof) + self.slip
        state_derivatives[self.slip] = (force_in - force_out)/self.mass
        state_derivatives[self.velocity] = force_out/self.mass
        return state_derivatives


class KOPWheels(SolidWheels):
    def __init__(self, gearbox, diameter, count, normal_force):
        SolidWheels.__init__(self, gearbox, count, diameter, .00107, .9, normal_force)

class MecanumWheel:
    """
    Simulates the dynamics of one or more mecanum wheels attached to a single gearbox
    """

    def __init__(self, gearbox, wheel_count, wheel_diameter, static_cof, dynamic_cof, normal_force):
        self.gearbox = gearbox
        self.count = wheel_count
        self.diameter = wheel_diameter/12
        self.circumference = self.diameter*math.pi
        self.total_static_cof = normal_force*static_cof*wheel_count
        self.total_dynamic_cof = normal_force*dynamic_cof*wheel_count
        self.state = {
            "velocity": [0, 0, 0],
            "travel": [0, 0, 0]
        }

    def get_output(self):
        gearbox_torque = self.gearbox.get_output()
        wheel_force = gearbox_torque/(self.diameter/2)
        wheel_edge_speed = self.gearbox.state["rpm"]*self.circumference/60
        slip_rate = wheel_edge_speed - self.state["travel"][1]
        if slip_rate > .1 or abs(wheel_force) > self.total_static_cof:
            wheel_force = max(-self.total_dynamic_cof, min(self.total_dynamic_cof, wheel_force))
        return [wheel_force, wheel_force, 0]

    def update_state(self, dt, new_state):
        gearbox_torque = self.gearbox.get_output()
        wheel_force = gearbox_torque/(self.diameter/2)
        wheel_edge_speed = self.gearbox.state["rpm"]*self.circumference/60
        slip_rate = wheel_edge_speed - self.state["travel"][1]
        if (slip_rate != 0 or abs(wheel_force) > self.total_static_cof) and abs(wheel_force) > self.total_dynamic_cof:
            spin_force = wheel_force - max(-self.total_dynamic_cof, min(self.total_dynamic_cof, wheel_force))
            accel = max(-5, min(spin_force*(self.diameter/2), 5))
            rpm = self.gearbox.state["rpm"] + accel*dt*60
            rotations = self.gearbox.state["rotations"] + rpm*dt/60
        else:
            rpm = new_state["velocity"][1]/self.circumference*60
            rotations = (new_state["travel"][1]-self.state["travel"][1])/self.circumference

        self.gearbox.update_state(dt, {"rpm": rpm, "rotations": rotations})
        self.state = new_state