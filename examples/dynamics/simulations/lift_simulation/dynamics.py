from int_dynamics import dynamics


class MyRobotDynamics(dynamics.DynamicsEngine):

    def build_loads(self):
        lift_motor = dynamics.CIMMotor()
        lift_gearbox = dynamics.GearBox([lift_motor], 20, 0)
        lift_wheel = dynamics.SimpleWheels(lift_gearbox, 3)
        self.lift_load = dynamics.OneDimensionalLoad([lift_wheel], 30)
        self.loads["lift"] = self.lift_load
        self.controllers["lift"] = dynamics.SpeedController(lift_motor)
        self.controllers["lift"].set_percent_vbus(1)

