from int_dynamics import dynamics
import math
import numpy as np


class MyRobotDynamics:

    def __init__(self):
        self.drivetrain = dynamics.KOPAssembly(120)

        self.lift_speed_controller = dynamics.SpeedController()
        self.lift_motor = dynamics.CIMMotor(self.lift_speed_controller)
        self.lift_gearbox = dynamics.GearBox([self.lift_motor], 20, 0)
        self.lift_encoder = dynamics.Encoder(self.lift_gearbox)
        self.lift_wheel = dynamics.SimpleWheels(self.lift_gearbox, 8)
        self.lift_load = dynamics.OneDimensionalLoad([self.lift_wheel], 60/32)
        self.lift_integrator = dynamics.Integrator()
        self.lift_integrator.build_ode_update(self.lift_load.get_state_derivatives())
        self.lift_integrator.build_sensor_update(self.lift_encoder.get_sensor_data())

        self.get_state()

    def update_sensors(self):
        self.drivetrain.imu.angle.set_percent_vbus(self.sensors["gyro"])
        self.drivetrain.left_encoder.angle.set_percent_vbus(self.sensors["left_encoder"])
        self.drivetrain.right_encoder.angle.set_percent_vbus(self.sensors["right_encoder"])

    def update_controls(self, hal_data=None):
        if hal_data is not None:
            self.controls = {
                "left_drive_cim": hal_data['pwm'][0]['value'],
                "right_drive_cim": hal_data['pwm'][1]['value'],
            }
        self.drivetrain.set_values(self.controls["left_drive_cim"],
                                   self.controls["right_drive_cim"])

    def update_physics(self, dt):
        self.lift_speed_controller.set_percent_vbus(1)
        self.lift_integrator.update_physics(dt)
        self.drivetrain.update_physics(dt)

    def get_sensors(self, hal_data=None):
        self.sensors = {
            "gyro": self.drivetrain.imu.angle.get_value() + np.random.normal(0, .05),
            "left_encoder": self.drivetrain.left_gearbox.position.get_value(),
            "right_encoder": self.drivetrain.right_gearbox.position.get_value()
        }
        if hal_data is not None:
            hal_data['analog_in'][0]['accumulator_value'] = math.degrees(self.sensors["gyro"]) / 2.7901785714285715e-12
            hal_data['encoder'][0]['count'] = math.degrees(self.sensors["left_encoder"])
            hal_data['encoder'][1]['count'] = math.degrees(self.sensors["right_encoder"])

    def get_state(self):
        self.state = {
            "drivetrain": self.drivetrain.get_state(),
        }
        return self.state


def get_dynamics():
    return MyRobotDynamics()
