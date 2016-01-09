import math
from . import power_supplies, motors, gearboxes, wheels, loads, sensors, integrator


class KOPAssembly:
    """
    A simple, no-frills KOP drivetrain assembly. 6wd tank with one cim and toughbox per side.
    """

    def __init__(self, bot_weight=120, simple_wheels=False, calc_ekf=False):
        """
        :param bot_weight: (optional) weight of robot in pounds.
        """
        self.left_speed_controller = power_supplies.SpeedController()
        self.right_speed_controller = power_supplies.SpeedController()

        self.left_motor = motors.CIMMotor(self.left_speed_controller)
        self.right_motor = motors.CIMMotor(self.right_speed_controller)

        self.left_gearbox = gearboxes.GearBox([self.left_motor], 10, 1)
        self.left_encoder = sensors.Encoder(self.left_gearbox)
        self.right_gearbox = gearboxes.GearBox([self.right_motor], 10, 1)
        self.right_encoder = sensors.Encoder(self.right_gearbox)

        if simple_wheels:
            self.left_wheels = wheels.SimpleWheels(self.left_gearbox, 6)
            self.right_wheels = wheels.SimpleWheels(self.right_gearbox, 6)
        else:
            self.left_wheels = wheels.KOPWheels(self.left_gearbox, 3, 6, bot_weight / 2)
            self.right_wheels = wheels.KOPWheels(self.right_gearbox, 3, 6, bot_weight / 2)

        self.drivetrain_load = loads.TwoDimensionalLoad(bot_weight/32)
        self.drivetrain_load.add_wheel(self.left_wheels, x_origin=-.5)
        self.drivetrain_load.add_wheel(self.right_wheels, x_origin=.5, r_origin=math.pi)

        self.gyro = sensors.Gyro(self.drivetrain_load)

        self.drivetrain_integrator = integrator.Integrator()
        self.drivetrain_integrator.add_ode_update(self.drivetrain_load.get_state_derivatives())
        self.drivetrain_integrator.add_sensor_update([
            self.left_encoder.get_sensor_data(),
            self.right_encoder.get_sensor_data(),
            self.gyro.get_sensor_data()
        ])

        self.update_physics = self.drivetrain_integrator.update_physics
        if calc_ekf:
            self.drivetrain_integrator.build_ekf_updater([
                self.gyro.get_sensor_data(),
                self.left_encoder.get_sensor_data(),
                self.right_encoder.get_sensor_data()
            ])
            self.update_ekf_physics = self.drivetrain_integrator.ekf_physics_update

    def set_values(self, left_value, right_value):
        self.left_speed_controller.set_value(left_value)
        self.right_speed_controller.set_value(right_value)

    def get_state(self):
        return {
            "velocity": self.drivetrain_load.velocity.get_value(),
            "position": self.drivetrain_load.position.get_value()
        }
