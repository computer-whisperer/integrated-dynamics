#!/usr/bin/env python3
import wpilib
from dynamics import get_dynamics

class MyRobot(wpilib.SampleRobot):
    '''Main robot class'''

    def robotInit(self):
        '''Robot-wide initialization code should go here'''

        self.dynamics = get_dynamics()

        self.stick = wpilib.Joystick(0)

        self.left_motor = wpilib.Talon(0)
        self.right_motor = wpilib.Talon(1)

        self.left_encoder = wpilib.Encoder(0, 1)
        self.right_encoder = wpilib.Encoder(2, 3)

        # Position gets automatically updated as robot moves
        self.gyro = wpilib.AnalogGyro(0)

    def disabled(self):
        '''Called when the robot is disabled'''
        while self.isDisabled():
            wpilib.Timer.delay(0.01)

    def autonomous(self):
        '''Called when autonomous mode is enabled'''

        timer = wpilib.Timer()
        timer.start()

        while self.isAutonomous() and self.isEnabled():

            if timer.get() < 2.0:
                self.robot_drive.arcadeDrive(0, 1)
            else:
                self.robot_drive.arcadeDrive(0, 0)

            wpilib.Timer.delay(0.01)

    def operatorControl(self):
        '''Called when operation control mode is enabled'''

        while self.isOperatorControl() and self.isEnabled():

            x = self.stick.getX()
            y = self.stick.getY()
            left = y-x
            right = -(y+x)
            self.left_motor.set(left)
            self.right_motor.set(right)
            self.dynamics.controls["left_motor"] = left
            self.dynamics.controls["right_motor"] = right
            self.dynamics.sensors["gyro"] = self.gyro.getAngle()
            self.dynamics.sensors["right_encoder"] = self.right_encoder.get()
            self.dynamics.sensors["left_encoder"] = self.left_encoder.get()
            self.dynamics.update_ekf_physics(.1)
            wpilib.Timer.delay(0.1)
        print({
            "left_encoder": self.left_encoder.get(),
            "right_encoder": self.right_encoder.get(),
            "gyro": self.gyro.getAngle(),
            "est_state": self.dynamics.get_state()
        })



if __name__ == '__main__':

    wpilib.run(MyRobot,
               physics_enabled=True)


