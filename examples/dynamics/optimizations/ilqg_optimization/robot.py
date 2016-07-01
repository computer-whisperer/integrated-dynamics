#!/usr/bin/env python3
import wpilib
from dynamics import get_dynamics

class MyRobot(wpilib.SampleRobot):
    '''Main robot class'''

    def robotInit(self):
        '''Robot-wide initialization code should go here'''

        self.dynamics = get_dynamics()
        self.dynamics.init_wpilib_devices()

        self.stick = wpilib.Joystick(0)


    def operatorControl(self):
        '''Called when operation control mode is enabled'''

        while self.isOperatorControl() and self.isEnabled():
            x = self.stick.getX()
            y = self.stick.getY()
            self.dynamics.controllers["left_drive"].set_percent_vbus(y-x)
            self.dynamics.controllers["right_drive"].set_percent_vbus(-(y+x))
            self.dynamics.prediction_update(0.05)
            wpilib.Timer.delay(0.05)
        print(self.dynamics.get_state())



if __name__ == '__main__':

    wpilib.run(MyRobot,
               physics_enabled=True)


