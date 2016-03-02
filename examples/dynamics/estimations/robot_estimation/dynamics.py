from int_dynamics import dynamics
import math


class MyRobotDynamics(dynamics.DynamicsEngine):

    SINK_IN_SIMULATION = True
    SINK_TO_SIMPLESTREAMER = True

    def build_loads(self):
        # Init drivetrain components (the assembly does this for us)

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

        # Init drivetrain sensors
        #self.sensors['gyro'] = dynamics.AnalogGyro(self.loads['drivetrain'], 0)
        #self.sensors['left_encoder'] = dynamics.Encoder(left_gearbox, 0, 1)
        #self.sensors['right_encoder'] = dynamics.Encoder(right_gearbox, 2, 3)

        # Set drivetrain controllers
        self.controllers['left_drive'] = dynamics.PWMSpeedController(left_motor, 0)
        self.controllers['right_drive'] = dynamics.PWMSpeedController(right_motor, 1)
