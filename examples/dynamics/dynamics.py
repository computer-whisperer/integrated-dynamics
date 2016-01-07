from int_dynamics.dynamics import KOPAssembly, SpeedController, CIMMotor, GearBox, KOPWheels, SimpleWheels, OneDimensionalLoad
from int_dynamics.dynamics.integrator import build_integrator
import math
import numpy as np

class MyRobotDynamics:

    def __init__(self):
        self.drivetrain = KOPAssembly(120)
        #self.drivetrain.drivetrain_load.velocity.set_value(np.array([-5, 0, 0]))
        #self.drivetrain.drivetrain_load.position.set_value(np.array([0, 0, math.pi/2]))

        #self.lift_speed_controller = SpeedController()
        #self.lift_motor = CIMMotor(self.lift_speed_controller)
        #self.lift_gearbox = GearBox([self.lift_motor], 20, 0)
        #self.lift_wheel = KOPWheels(self.lift_gearbox, 1, 3, 30)
        #self.lift_wheel = SimpleWheels(self.lift_gearbox, 3)
        #self.lift_load = OneDimensionalLoad([self.lift_wheel], 60/32)
        #self.lift_updater = build_integrator(self.lift_load.get_state_derivatives())

    def update_physics(self, dt):
        self.drivetrain.set_values(.5, -1)
        #self.lift_speed_controller.set_value(1)
        #print(self.drivetrain.drivetrain_load.velocity.get_value())
        print(self.drivetrain.drivetrain_load.position.get_value())
        #print(self.drivetrain.left_gearbox.position.get_value())
        #print(self.drivetrain.right_gearbox.position.get_value())
        self.drivetrain.update_state(dt)
        #lift_state = self.lift_updater(dt)
        #return {}


def get_dynamics():
    return MyRobotDynamics()
