#!/usr/bin/env python3
import wpilib

class MyRobot(wpilib.SampleRobot):
    '''Main robot class'''

    def robotInit(self):
        '''Robot-wide initialization code should go here'''
        self.talon = wpilib.CANTalon(0)
        self.talon.changeControlMode(wpilib.CANTalon.ControlMode.Position)
        self.talon.setPID(2, 0, 0)
        self.stick = wpilib.Joystick(0)

    def operatorControl(self):
        self.talon.enableControl()
        '''Called when operation control mode is enabled'''
        while self.isOperatorControl() and self.isEnabled():
            if self.stick.getTrigger():
                self.talon.set(100)
            else:
                self.talon.set(0)

            wpilib.Timer.delay(0.05)

if __name__ == '__main__':
    wpilib.run(MyRobot,
               physics_enabled=True)


