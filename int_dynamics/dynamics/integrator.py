__author__ = 'christian'
import theano
import theano.tensor as T
from theano.printing import Print
from theano.tensor import slinalg
from theano.ifelse import ifelse


def build_integrator(state_derivatives):
        dt = T.scalar(dtype=theano.config.floatX)

        # Trim away any states that are not shared variables

        trimmed_state_derivatives = {}
        for state in state_derivatives:
            if hasattr(state, "get_value"):
                trimmed_state_derivatives[state] = state_derivatives[state]
        state_derivatives = trimmed_state_derivatives

        # Flatten all states, saving order
        states = []
        flat_states = []
        for state in state_derivatives:
            states.append(state)

            if state.ndim < 1:
                state = state.dimshuffle('x')
            flat_states.append(state)

        # Build complete state vector
        state_full = Print("state_full")(T.concatenate(flat_states))

        state_A = []
        state_b = []
        for state in states:
            A_blocks = theano.gradient.jacobian(state_derivatives[state], states, disconnected_inputs='ignore')

            # Correct dimensions
            for i in range(len(A_blocks)):
                if A_blocks[i].ndim < 1:
                    A_blocks[i] = A_blocks[i].dimshuffle('x', 'x')
                elif A_blocks[i].ndim < 2:
                    A_blocks[i] = A_blocks[i].dimshuffle('x', 0)

            # Stack A_blocks
            A_row = T.unbroadcast(T.concatenate(A_blocks, axis=1), 0, 1)
            b = state_derivatives[state] - T.dot(A_row, state_full)

            state_A.append(A_row)
            state_b.append(b)
        A = theano.printing.Print("A")(T.concatenate(state_A))
        b = theano.printing.Print("b")(T.concatenate(state_b))

        # Equation given by http://math.stackexchange.com/questions/1567784/matrix-differential-equation-xt-axtb-solution-defined-for-non-invertible/1567806?noredirect=1#comment3192556_1567806

        # Matrix exponentiation method
        eat = Print("eat")(slinalg.expm(A*dt))

        # Two methods to calculate the integral:

        # e^(at) method, given by http://wolfr.am/9mNgcOgM
        eat_integral = Print("eat_integral")(T.dot(eat-1, T.inv(A)))
        eat_integral = T.unbroadcast(eat_integral, 0, 1)

        # Taylor series method
        def series_advance(i, last_term, A, wrt):
            next_term = T.dot(last_term, A)*wrt/i
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-5))
        init_term = dt*T.identity_like(A)
        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 100)],
                               non_sequences=[A, dt],
                               outputs_info=init_term,
                               )
        taylor_integral = theano.printing.Print("taylor_integral")(T.sum(terms, axis=0) + dt)

        # Decide which integral to use, preferring the eat method when it works
        integral = ifelse(T.any(T.isnan(eat_integral)), taylor_integral, eat_integral)

        new_state = T.dot(eat, flat_states).flatten() + T.dot(integral, b).flatten()
        #index = 0
        #updates = []
        #for state in states:
        #    state_len = T.shape(state)[0]
        #    state_update = new_state[index:state_len]
        #    state_update.reshape(T.shape(state))
        #    updates.append((state, state_update))
        #    index += state_len

        return theano.function([dt], new_state)#, updates=updates, profile=False)