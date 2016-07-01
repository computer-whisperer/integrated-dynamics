import threading
import time
from hal_impl.data import hal_data


class BasePhysicsEngine(object):

    def __init__(self, physics_controller):
        '''
            :param physics_controller: `pyfrc.physics.core.Physics` object
                                       to communicate simulation effects to
        '''

        self.physics_controller = physics_controller
        self.dynamics = self.get_dynamics()
        self.dynamics_thread = threading.Thread(target=self._dynamics_loop)
        self.state = self.dynamics.get_state()
        self.dynamics_thread.start()

    def get_dynamics(self): # Override me!
        return None

    def _dynamics_loop(self):
        while True:
            self.dynamics.simulation_update(0.05, hal_data)
            self.state = self.dynamics.get_state()
            time.sleep(0.05)

    def update_sim(self, hal_data, now, tm_diff):
        '''
            Called when the simulation parameters for the program need to be
            updated.

            :param now: The current time as a float
            :param tm_diff: The amount of time that has passed since the last
                            time that this function was called
        '''

        #print(state)
        # For some reason pyfrc's x and y are flopped
        position = self.state.get("drivetrain", {}).get("position", [0, 0, 0])
        self.physics_controller.x = position[0]
        self.physics_controller.y = position[1]
        self.physics_controller.angle = position[2]


