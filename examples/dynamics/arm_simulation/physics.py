from int_dynamics.dynamics.physics_engine import BasePhysicsEngine
from dynamics import MyRobotDynamics


class PhysicsEngine(BasePhysicsEngine):
    def get_dynamics(self):
        return MyRobotDynamics.cached_init("simulation")
