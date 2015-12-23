import math
import numpy as np
from components import DynamicsComponent
import theano
import theano.tensor as T
from utilities import integrate_via_ode

class SimpleWheels(DynamicsComponent):
    """
    Simulates the dynamics of a wheel without friction calculations
    """

    def __init__(self, components, diameter):
        """
        :param components: The list of objects providing torque to drive the wheel(s) from
        :param diameter: The diameter of the wheel(s) in inches
        """
        self.diameter = diameter/12
        super().__init__(components)
        self.state = {
            "velocity": theano.shared(np.array([0.0, 0.0]), theano.config.floatX)
        }

    def get_input_force_tensor(self):
        torque_in = np.array([0.0, 0.0])
        for component in self.input_components:
            torque_in += np.array([0, 1]) *component["component"].get_force_tensor()
        return torque_in

    def build_state_tensors(self, travel, velocity, dt):
        wheel_vel = velocity/(math.pi*self.diameter)
        wheel_travel = travel/(math.pi*self.diameter)
        self.state_tensors = {
            "velocity": wheel_vel
        }
        self.build_input_state_tensors(wheel_travel[1], wheel_vel[1], dt)


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
            "velocity": np.array([0.0, 0.0], theano.config.floatX),
            "travel": np.array([0.0, 0.0], theano.config.floatX)
        }

    def get_force_tensor(self):
        force = self.get_input_force_tensor()/self.diameter/2
        return force + T.min(self.total_static_cof, T.norm(force))/force

    def build_state_tensors(self, travel, velocity, dt):

        self.state_tensors = {
            "velocity": velocity,
            "travel": travel
        }
        self.build_input_state_tensors(travel[1], velocity[1], dt)
        accel = (self.get_input_force_tensor()/self.diameter/2 - self.get_force_tensor())/self.mass
        wheel_vel = velocity + accel*dt
        wheel_travel = travel + (wheel_vel-velocity)*dt
        self.state_tensors = {
            "velocity": wheel_vel,
            "travel": wheel_travel
        }
        self.build_input_state_tensors(wheel_travel[1], wheel_vel[1], dt)

    def update_state(self, dt, new_state):
        power_in = self.source.get_output()
        force = np.array([-self.state["velocity"][0], power_in["torque"]/self.radius])
        force_derivative = np.array([[-1, 0],
                                     [0, power_in["d_torque/d_rps"]/self.circumference/self.radius]])

        slip_rate = np.array([0, self.source.state["rps"]*self.circumference]) - self.state["velocity"]
        slip_magnitude = np.linalg.norm(slip_rate)
        force_magnitude = np.linalg.norm(force)
        if slip_magnitude > .1 or force_magnitude > self.total_static_cof:
            # Equation for new rpm:
            # http://wolfr.am/8BlLGSYm
            #
            # Equation for new position:
            # http://wolfr.am/8BlO6iX9

            special_e = math.e**(force_derivative*self.radius*dt/self.mass)
            F_over_D = (force[1]-math.copysign(self.total_dynamic_cof, slip_rate[1]))/force_derivative
            rps = F_over_D*(special_e-1)+self.source.state["rps"]
            rotations = F_over_D*(self.mass*special_e-force_derivative*self.radius*dt)/force_derivative/self.radius + self.source.state["rps"]*dt + self.source.state["rotations"]
        else:
            rps = new_state["velocity"][1]/self.circumference
            rotations = self.source.state["rotations"] + (new_state["travel"][1]-self.state["travel"][1])/self.circumference

        self.source.update_state(dt, {"rps": rps, "rotations": rotations})
        self.state = new_state

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