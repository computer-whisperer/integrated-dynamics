from int_dynamics import dynamics, utilities
import math
import os


class MyRobotDynamics(dynamics.DynamicsEngine):

    def build_loads(self):
        # Setup a simple drivetrain

        # Two CIMs
        left_motor = dynamics.CIMMotor()
        right_motor = dynamics.CIMMotor()
        # Two 10:1 gearboxes
        left_gearbox = dynamics.GearBox([left_motor], 10, 1)
        right_gearbox = dynamics.GearBox([right_motor], 10, 1)

        left_wheels = dynamics.KOPWheels(left_gearbox, 3, 6, 60)
        right_wheels = dynamics.KOPWheels(right_gearbox, 3, 6, 60)

        self.loads["drivetrain"] = dynamics.TwoDimensionalLoad(120)
        self.loads["drivetrain"].add_wheel(left_wheels, x_origin=-.5)
        self.loads["drivetrain"].add_wheel(right_wheels, x_origin=.5, r_origin=math.pi)

        # Init sensors
        self.sensors['left_encoder'] = dynamics.Encoder(left_gearbox, 0, 1)
        self.sensors['right_encoder'] = dynamics.Encoder(right_gearbox, 2, 3)

        # Init controller
        self.controllers['left_controller'] = dynamics.PWMSpeedController(left_motor, 0)
        self.controllers['right_controller'] = dynamics.PWMSpeedController(right_motor, 1)


def get_dynamics():
    return utilities.cache_object(MyRobotDynamics, file_path=os.path.abspath(__file__))
