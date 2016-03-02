from int_dynamics import dynamics
import math


class MyRobotDynamics(dynamics.DynamicsEngine):

    SINK_IN_SIMULATION = True
    SINK_TO_NT = False

    def build_loads(self):
        # Setup a simple drivetrain

        # Two CIM
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
        self.controllers['left_cim'] = dynamics.PWMSpeedController(left_motor, 0)
        self.controllers['right_cim'] = dynamics.PWMSpeedController(right_motor, 1)

