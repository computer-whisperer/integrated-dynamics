import int_dynamics.dynamics.components.actuators
from int_dynamics import dynamics


class MyRobotDynamics(dynamics.DynamicsEngine):

    def build_loads(self):
        lift_motor_1 = int_dynamics.dynamics.components.actuators.CIMMotor()
        lift_motor_2 = int_dynamics.dynamics.components.actuators.MiniCIMMotor()
        lift_gearbox = dynamics.GearBox([lift_motor_1], 20, 0)
        lift_wheel = dynamics.SimpleWheels(lift_gearbox, 1)
        self.lift_load = dynamics.OneDimensionalLoad([lift_wheel], 150)
        self.loads["lift"] = self.lift_load
        self.controllers["motor 1"] = dynamics.SpeedController(lift_motor_1)
        self.controllers["motor 2"] = dynamics.SpeedController(lift_motor_2)

