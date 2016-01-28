from int_dynamics import dynamics
import math
import numpy as np


class MyRobotDynamics:

    def __init__(self):
        self.drivetrain = dynamics.KOPAssembly(120)
        self.get_state()

    def set_control_vector(self, control):
        self.drivetrain.set_values(control[0], control[1])

    def update_physics(self, dt):
        self.drivetrain.update_physics(dt)

    def get_state_vector(self):
        return self.drivetrain.drivetrain_integrator.state_vector_shared

    def get_state_derivative(self):
        pass

    def get_state(self):
        self.state = {
            "drivetrain": self.drivetrain.get_state()
        }
        return self.state


def get_dynamics():
    return MyRobotDynamics()
