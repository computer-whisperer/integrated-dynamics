from int_dynamics import dynamics, utilities
import math
import numpy as np
import os


class MyRobotDynamics:

    def __init__(self):
        self.drivetrain = dynamics.KOPAssembly(120)#, calc_ekf=True)
        self.get_state()
        self.get_sensors()
        self.get_controls()

    def get_sensors(self, hal_data=None):
        self.sensors = {
            "gyro": self.drivetrain.imu.angle.get_value() + np.random.normal(0, .05),
            "left_encoder": self.drivetrain.left_gearbox.position.get_value(),
            "right_encoder": self.drivetrain.right_gearbox.position.get_value()
        }
        return self.sensors

    def update_sensors(self):
        self.drivetrain.imu.angle.set_percent_vbus(self.sensors["gyro"])
        self.drivetrain.left_encoder.position.set_percent_vbus(self.sensors["left_encoder"])
        self.drivetrain.right_encoder.position.set_percent_vbus(self.sensors["right_encoder"])

    def get_controls(self):
        self.controls = {
            "left_drive_cim": self.drivetrain.left_speed_controller.percent_vbus.get_value(),
            "right_drive_cim": self.drivetrain.right_speed_controller.percent_vbus.get_value(),
        }
        return self.controls

    def update_controls(self, hal_data=None, add_noise=False):
        if hal_data is not None:
            self.controls = {
                "left_drive_cim": hal_data['pwm'][0]['value'],
                "right_drive_cim": hal_data['pwm'][1]['value'],
            }
        if add_noise:
            self.controls["left_drive_cim"] += np.random.normal(0, .1)
            self.controls["right_drive_cim"] += np.random.normal(0, .1)
        self.drivetrain.set_values(self.controls["left_drive_cim"],
                                   self.controls["right_drive_cim"])

    def update_physics(self, dt):
        self.drivetrain.update_physics(dt)

    def get_state(self):
        self.state = {
            "drivetrain": self.drivetrain.get_state(),
        }
        return self.state

    def get_state_derivatives(self):
        self.drivetrain.get_state_derivatives()


def get_dynamics():
    return utilities.cache_object(MyRobotDynamics, file_path=os.path.abspath(__file__))
