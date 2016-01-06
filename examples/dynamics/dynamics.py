from int_dynamics.dynamics import KOPAssembly, SpeedController, CIMMotor, GearBox, KOPWheels, SimpleWheels, OneDimensionalLoad
from int_dynamics.dynamics.integrator import build_integrator
import math
import numpy as np

class MyRobotDynamics:

    def __init__(self):
        #self.drivetrain = KOPAssembly(120)
        #self.drivetrain.drivetrain_load.state["position"].set_value(np.array([0, 0, 0]))

        self.lift_speed_controller = SpeedController()
        self.lift_motor = CIMMotor(self.lift_speed_controller)
        self.lift_gearbox = GearBox([self.lift_motor], 20, 0)
        self.lift_wheel = KOPWheels(self.lift_gearbox, 1, 1, 30)
        #self.lift_wheel = SimpleWheels(self.lift_gearbox, 3)
        self.lift_load = OneDimensionalLoad([self.lift_wheel], 60/32)
        self.lift_updater = build_integrator(self.lift_load.get_state_derivatives(.3))

    def update_physics(self, dt):
        #self.drivetrain.set_values(1, -1)
        self.lift_speed_controller.set_value(1)
        #drivetrain_state = self.drivetrain.update_state(dt)
        lift_state = self.lift_updater(dt)
        return {
            #"drivetrain": drivetrain_state,
            "lift": lift_state,
        }

    def get_state(self):
        return {
        #   "drivetrain": self.drivetrain.get_state(),
            "lift": self.lift_load.state
        }


def get_dynamics():
    return MyRobotDynamics()
