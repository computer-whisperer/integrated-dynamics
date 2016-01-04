import math

import numpy as np
import theano
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
        mech_advantage = 1/(math.pi * self.diameter)
        self.gearbox.velocity = self.velocity[1]*mech_advantage
        state_derivatives = self.gearbox.get_state_derivatives(load_mass)
        state_derivatives[self.velocity] = np.array([0, 1])*state_derivatives[self.gearbox.velocity]/mech_advantage
        return state_derivatives


class SolidWheels(SimpleWheels):
    """
    Simulates the dynamics of a set of solid wheels attached to a single gearbox
    """

    def __init__(self, components, count, diameter, static_cof, dynamic_cof, normal_force):
        super().__init__(components, diameter)
        self.count = count
        self.mass = .25*count
        self.total_static_cof = normal_force*static_cof
        self.total_dynamic_cof = normal_force*dynamic_cof
        self.state = {
            "velocity": theano.shared(np.array([0.0, 0.0]), theano.config.floatX),
            "slip": theano.shared(np.array([0.0, 0.0]), theano.config.floatX)
        }

    def get_force_tensor(self):
        force = self.get_input_force_tensor()/(self.diameter/2)
        #return force - self.total_static_cof*self.state_tensors["slip"]/T.sum(self.state["slip"]**2)**.5
        return force - self.total_static_cof*self.state_derivatives["slip"]

    def build_state_updates(self, travel, velocity, dt):
        wheel_vel = velocity/(self.diameter*math.pi) + self.state["slip"]
        wheel_travel = travel/(self.diameter*math.pi) + self.state["slip"]*dt
        self.state_derivatives = {
            "velocity": wheel_vel,
            "slip": self.state["slip"]
        }
        self.build_source_state_derivatives(wheel_travel[1], wheel_vel[1], dt)
        #slip_dir = self.state["slip"]/T.sum(self.state["slip"]**2)**.5
        #slip_dir = ifelse(T.any(T.isnan(slip_dir)), np.array([0.0, 0.0]), slip_dir)
        #slip_accel = (self.get_force_tensor() - self.total_dynamic_cof*slip_dir)/self.mass
        slip_accel = (self.get_force_tensor() - self.total_dynamic_cof*self.state["slip"])/self.mass
        new_slip = theano.printing.Print("slip")(integrate_via_ode(slip_accel, self.state["slip"], dt, self.state["slip"]))#\, [velocity, travel]))
        #new_slip = theano.shared(np.array([0.0, 0.0]), theano.config.floatX)

        wheel_vel = velocity/(self.diameter*math.pi) + new_slip
        wheel_travel = travel/(self.diameter*math.pi) + new_slip*dt
        self.state_derivatives = {
            "velocity": wheel_vel,
            "slip": new_slip
        }
        self.build_source_state_derivatives(wheel_travel[1], wheel_vel[1], dt)


class KOPWheels(SolidWheels):
    def __init__(self, gearbox, diameter, count, normal_force):
        SolidWheels.__init__(self, gearbox, count, diameter, 1.07, .9, normal_force)


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