import math

from components import SpeedController, CIMMotor, GearBox, SimpleWheels, TwoDimensionalLoad


class KOPAssembly:
    """
    A simple, no-frills KOP drivetrain assembly. 6wd tank with one cim and toughbox per side.
    """

    def __init__(self, bot_weight=120):
        """
        :param bot_weight: (optional) weight of robot in pounds.
        """
        self.left_speed_controller = SpeedController()
        self.right_speed_controller = SpeedController()

        self.left_motor = CIMMotor(self.left_speed_controller)
        self.right_motor = CIMMotor(self.right_speed_controller)

        self.left_gearbox = GearBox([self.left_motor], 10, 1)
        self.right_gearbox = GearBox([self.right_motor], 10, 1)

        #self.left_wheels = KOPWheels(self.left_gearbox, 3, 6, bot_weight / 2)
        #self.right_wheels = KOPWheels(self.right_gearbox, 3, 6, bot_weight / 2)
        self.left_wheels = SimpleWheels(self.left_gearbox, 6)
        self.right_wheels = SimpleWheels(self.right_gearbox, 6)

        self.drivetrain_load = TwoDimensionalLoad(bot_weight/32)
        self.drivetrain_load.add_input(self.left_wheels, x_origin=-.5)
        self.drivetrain_load.add_input(self.right_wheels, x_origin=.5, r_origin=math.pi)
        self.drivetrain_load.build_functions()

    def set_values(self, left_value, right_value):
        self.left_speed_controller.set_value(left_value)
        self.right_speed_controller.set_value(right_value)

    def update_state(self, dt):
        return self.drivetrain_load.update_state(dt)

    def get_state(self):
        return self.drivetrain_load.state