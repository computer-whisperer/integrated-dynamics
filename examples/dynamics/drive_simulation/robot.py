#!/usr/bin/env python3
import wpilib
from dynamics import get_dynamics

class MyRobot(wpilib.SampleRobot):

    def robotInit(self):
        self.robot_drive = wpilib.RobotDrive(0, 1)
        self.joystick = wpilib.Joystick(0)

    def operatorControl(self):
        while self.isOperatorControl() and self.isEnabled():
            self.robot_drive.arcadeDrive(self.joystick)

            wpilib.Timer.delay(0.1)

if __name__ == '__main__':
    wpilib.run(MyRobot,
               physics_enabled=True)


