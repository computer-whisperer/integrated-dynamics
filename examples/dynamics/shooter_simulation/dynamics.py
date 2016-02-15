from int_dynamics import dynamics


class MyRobotDynamics(dynamics.DynamicsEngine):

    def build_loads(self):
        shooter_motor = dynamics.CIMMotor()
        shooter_gearbox = dynamics.GearBox([shooter_motor], 2, 0)
        shooter_wheel = dynamics.SimpleWheels(shooter_gearbox, 8)
        self.shooter_load = dynamics.OneDimensionalLoad([shooter_wheel], 5)
        self.loads["shooter_wheel"] = self.shooter_load
        self.controllers["shooter"] = dynamics.SpeedController(shooter_motor)
        self.controllers["shooter"].set_percent_vbus(.1)

    def add_ball(self):
        wheel_mass = self.shooter_load.mass.get_value()
        wheel_vel = self.shooter_load.velocity.get_value()
        wheel_energy = wheel_vel*wheel_mass
        ball_plus_wheel_mass = wheel_mass + .325/32
        ball_plus_wheel_speed = wheel_energy/ball_plus_wheel_mass
        self.shooter_load.mass.set_value(ball_plus_wheel_mass)
        self.shooter_load.velocity.set_value(ball_plus_wheel_speed)
        self.shooter_load.position.set_value(0)


