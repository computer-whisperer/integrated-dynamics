class BasePhysicsEngine(object):

    def __init__(self, physics_controller):
        '''
            :param physics_controller: `pyfrc.physics.core.Physics` object
                                       to communicate simulation effects to
        '''

        self.physics_controller = physics_controller
        self.dynamics = self.get_dynamics()
        self.last_state = self.dynamics.get_state()

    def get_dynamics(self): # Override me!
        return None

    def update_sim(self, hal_data, now, tm_diff):
        '''
            Called when the simulation parameters for the program need to be
            updated.

            :param now: The current time as a float
            :param tm_diff: The amount of time that has passed since the last
                            time that this function was called
        '''

        self.dynamics.simulation_update(tm_diff, hal_data)
        state = self.dynamics.get_state()
        #print(state)
        # For some reason pyfrc's x and y are flopped
        position = state.get("drivetrain", {}).get("position", [0, 0, 0])
        self.physics_controller.y = position[0]
        self.physics_controller.x = position[1]
        self.physics_controller.angle = position[2]

        self.last_state = state

