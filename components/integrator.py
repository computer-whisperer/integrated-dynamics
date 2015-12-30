__author__ = 'christian'
import theano
import theano.tensor as T

class ComponentIntegrator:

    def __init__(self, component, mass):
        self.component = component
        self.mass = mass
        self.dt = T.iscalar()
        self.state =

        # Get the current force out in terms of velocity
        self.state_derivatives = {
            "velocity": velocity,
            "position": self.state["position"] + travel
        }
        self.build_source_state_derivatives(travel, velocity, dt)
        force = self.get_force_tensor()

        # Integrate it to find new_velocity and travel
        acceleration = force/self.mass
        new_velocity = integrate_via_ode(acceleration, velocity, dt, self.state["velocity"])#, [velocity, travel])
        new_travel = (self.state["velocity"] + new_velocity)*dt/2

        # Recalculate state tensors
        self.state_derivatives = {
            "velocity": new_velocity,
            "position": self.state["position"] + new_travel
        }
        #self.build_input_state_tensors(new_travel, new_velocity, dt)

