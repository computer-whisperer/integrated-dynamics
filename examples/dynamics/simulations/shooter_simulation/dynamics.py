import int_dynamics.dynamics.components.actuators
from int_dynamics import dynamics


class MyRobotDynamics(dynamics.DynamicsEngine):
    SINK_IN_SIMULATION = True
    SINK_TO_NT = True

    def build_loads(self):
        shooter_motor = int_dynamics.dynamics.components.actuators.CIMMotor()
        shooter_gearbox = dynamics.GearBox([shooter_motor], 1, 0)
        shooter_wheel = dynamics.SimpleWheels(shooter_gearbox, 4)
        self.shooter_load = dynamics.OneDimensionalLoad([shooter_wheel], 1)
        self.loads["shooter_wheel"] = self.shooter_load
        self.sensors["shooter_enc"] = dynamics.CANTalonEncoder(shooter_gearbox, 360)
        self.controllers["shooter"] = dynamics.CANTalonSpeedController(shooter_motor, 0)
        self.controllers["shooter"].add_encoder(self.sensors["shooter_enc"])

    def add_ball(self):
        wheel_mass = self.shooter_load.mass.get_value()
        wheel_vel = self.shooter_load.velocity.get_value()
        wheel_energy = wheel_vel*wheel_mass
        ball_plus_wheel_mass = wheel_mass + .325/32
        ball_plus_wheel_speed = wheel_energy/ball_plus_wheel_mass
        self.shooter_load.mass.set_value(ball_plus_wheel_mass)
        self.shooter_load.velocity.set_value(ball_plus_wheel_speed)
        self.shooter_load.position.set_value(0)


