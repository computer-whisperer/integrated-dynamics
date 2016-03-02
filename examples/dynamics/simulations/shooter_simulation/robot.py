#!/usr/bin/env python3
import wpilib


class MyRobot(wpilib.SampleRobot):

    def robotInit(self):
        self.shooter_wheel = wpilib.CANTalon(0)
        self.shooter_wheel.setControlMode(wpilib.CANTalon.ControlMode.Speed)
        self.shooter_wheel.setPID(0.1, 0.0, 0.0, 0)
        self.shooter_wheel.enableControl()

    def operatorControl(self):
        while self.isOperatorControl() and self.isEnabled():
            self.shooter_wheel.set(5000*360)

            wpilib.Timer.delay(0.1)

if __name__ == '__main__':
    wpilib.run(MyRobot,
               physics_enabled=True)


