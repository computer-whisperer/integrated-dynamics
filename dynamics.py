from components import KOPAssembly, SpeedController, CIMMotor, GearBox, KOPWheels, OneDimensionalLoad
import math
import numpy as np

class MyRobotDynamics:

    def __init__(self):
        #self.drivetrain = KOPAssembly(120)
        #self.drivetrain.drivetrain_load.state["position"].set_value(np.array([0, 0, 0]))

        self.lift_speed_controller = SpeedController()
        self.lift_motor = CIMMotor(self.lift_speed_controller)
        self.lift_gearbox = GearBox([self.lift_motor], 20, 0)
        self.lift_wheel = KOPWheels([self.lift_gearbox], 3, 6, 30)
        self.lift_load = OneDimensionalLoad([self.lift_wheel], 60/32)
        self.lift_load.build_functions()

    def update_physics(self, dt):
        #self.drivetrain.set_values(1, -1)
        self.lift_speed_controller.set_value(.1)
        #drivetrain_state = self.drivetrain.update_state(dt)
        lift_state = self.lift_load.update_state(dt)
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
