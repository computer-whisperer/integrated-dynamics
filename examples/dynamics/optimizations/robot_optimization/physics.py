from int_dynamics.dynamics.physics_engine import BasePhysicsEngine
from dynamics import get_dynamics

class PhysicsEngine(BasePhysicsEngine):
    def get_dynamics(self):
        return get_dynamics()
