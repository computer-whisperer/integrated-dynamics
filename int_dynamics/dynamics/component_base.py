__author__ = 'christian'
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import slinalg
from theano.ifelse import ifelse


class DynamicsComponent:

    def __init__(self, input_components=None):
        self.input_components = []
        self.state_derivatives = {}
        self.state_updates = {}
        self.state = {}
        if input_components is not None:
            for component in input_components:
                self.add_input(component)

    def add_input(self, component):
        self.input_components.append({"component": component})

    def build_state_derivatives(self, mass):
        for component in self.input_components:
            component["component"].build_state_derivatives(mass)

    def build_state_updates(self, source_state):
        for component in self.input_components:
            component["component"].build_state_updates()

    def get_state_derivatives(self, mass):
        self.build_state_derivatives(mass)
        for component in self.input_components:
            self.derivatives.update(component["component"].get_state_derivatives(mass))

    def get_state_updates(self):
        updates = []
        derivatives = []
        for component in self.input_components:
            new_updates, new_derivatives = component["component"].get_update_tensors()
            updates.extend(new_updates)
            derivatives.extend(new_derivatives)
        for state in self.state_derivatives:
            derivatives.append([self.state[state], self.state_derivatives[state]])
        for state in self.state_updates:
            updates.append([self.state[state], self.state_updates[state]])
        return updates, derivatives

    def build_integration(self):
        dt = T.scalar(dtype=theano.config.floatX)
        self.get_force_tensor()
        state_updates, state_derivatives = self.get_state_updates()
        states = []
        derivatives = []
        for state, derivative in state_derivatives:
            states.append(state)
            derivatives.append(derivative)
        total_state = T.concatenate(states)
        total_derivative = T.concatenate(derivatives)

        # Integrates state by linearizing state_derivatives around state and solving the resulting
        # ODE of the form
        # x'(wrt) = Ax(wrt) + b

        if total_derivative.ndim == 0:
            A = theano.grad(total_derivative, total_state)
        else:
            A = theano.gradient.jacobian(total_derivative, total_state)
        b = total_derivative - T.dot(total_state, A)

        # Equation given by http://math.stackexchange.com/questions/1567784/matrix-differential-equation-xt-axtb-solution-defined-for-non-invertible/1567806

        # Matrix exponentiation method
        eat = slinalg.expm(A*dt)

        # Two methods to calculate the integral:

        # e^(at) method, given by http://wolfr.am/9mNgcOgM
        eat_integral = T.dot(eat-1, T.inv(A))

        # Taylor series method
        def series_advance(i, last_term, A, wrt):
            next_term = T.dot(last_term, A)*wrt/i
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-5))
        if total_derivative.ndim == 0:
            init_term = dt
        else:
            init_term = dt*T.identity_like(A)
        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 100)],
                               non_sequences=[A, dt],
                               outputs_info=init_term,
                               )
        taylor_integral = T.sum(terms, axis=0) + init_term

        # Decide which integral to use, preferring the eat method when it works
        integral = ifelse(T.any(T.isnan(eat_integral)), taylor_integral, eat_integral)

        new_state = T.dot(eat, total_state) + T.dot(integral, b)

        self.update_state = theano.function([dt], [new_state], updates=(total_state, new_state), profile=False)

