from int_dynamics import dynamics


class MyRobotDynamics:

    def __init__(self):

        self.shooter_speed_controller = dynamics.SpeedController()
        self.shooter_motors = [dynamics.ThrottleMotor(self.shooter_speed_controller) for _ in range(0, 2)]
        self.shooter_gearbox = dynamics.GearBox(self.shooter_motors, 2, 0)
        self.shooter_wheel = dynamics.SimpleWheels(self.shooter_gearbox, 8)
        self.shooter_load = dynamics.OneDimensionalLoad([self.shooter_wheel], 5/32)
        self.shooter_integrator = dynamics.Integrator()
        self.shooter_integrator.build_ode_update(self.shooter_load.get_state_derivatives())

        self.get_state()

    def update_physics(self, dt):

        self.shooter_speed_controller.set_value(1)
        self.shooter_integrator.update_physics(dt)

    def add_ball(self):
        wheel_mass = self.shooter_load.mass.get_value()
        wheel_vel = self.shooter_load.velocity.get_value()
        wheel_energy = wheel_vel*wheel_mass
        ball_plus_wheel_mass = wheel_mass + .325/32
        ball_plus_wheel_speed = wheel_energy/ball_plus_wheel_mass
        self.shooter_load.mass.set_value(ball_plus_wheel_mass)
        self.shooter_load.velocity.set_value(ball_plus_wheel_speed)
        self.shooter_load.position.set_value(0)

    def get_state(self):
        self.state = {
            "ball": {
                "position": self.shooter_load.position.get_value(),
                "velocity": self.shooter_load.velocity.get_value()
            }
        }
        return self.state


def get_dynamics():
    return MyRobotDynamics()

