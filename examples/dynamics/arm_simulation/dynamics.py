from int_dynamics import dynamics, utilities
import os


class MyRobotDynamics(dynamics.DynamicsEngine):

    def build_loads(self):
        # Setup a simple arm mover

        # Start with a CIM
        motor = dynamics.RS775Motor()
        # Add a 10:1 gearbox
        gearbox = dynamics.GearBox(motor, 71)
        # Add a 1' arm
        arm = dynamics.SimpleArm(gearbox, 1)
        # Add a 10 lb load and save it as a robot component
        self.loads['arm_load'] = dynamics.OneDimensionalLoad([arm], 10)

        # Init sensor
        self.sensors['arm_encoder'] = dynamics.CANTalonEncoder(gearbox, tics_per_rev=497)

        # Init controller
        self.controllers['arm_talon'] = dynamics.CANTalonSpeedController(motor, 0)
        self.controllers['arm_talon'].add_encoder(self.sensors['arm_encoder'])


def get_dynamics():
    return utilities.cache_object(MyRobotDynamics, file_path=os.path.abspath(__file__))
