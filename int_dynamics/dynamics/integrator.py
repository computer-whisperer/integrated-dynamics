__author__ = 'christian'
import theano
import theano.tensor as T
from theano.printing import Print
from theano.tensor import slinalg


def build_integrator(state_derivatives):
        dt = T.scalar(dtype=theano.config.floatX)

        # Trim away any states that are not shared variables

        trimmed_state_derivatives = {}
        for state in state_derivatives:
            if hasattr(state, "get_value"):
                trimmed_state_derivatives[state] = state_derivatives[state]
        state_derivatives = trimmed_state_derivatives

        # build two-dimensional states and derivatives
        states = []
        twodim_states = []
        for state in state_derivatives:
            # save state to a list for reference
            states.append(state)

            # Correct state dimensions
            if state.ndim == 0:
                state = state.dimshuffle('x', 'x')
            elif state.ndim == 1:
                state = state.dimshuffle(0, 'x')
            twodim_states.append(state)

        # Build complete state vector
        state_full = T.concatenate(twodim_states)

        state_A = []
        state_b = []
        for state in states:
            A_blocks = theano.gradient.jacobian(state_derivatives[state], states, disconnected_inputs='ignore')

            # Correct dimensions
            for i in range(len(A_blocks)):

                if A_blocks[i].ndim == 0:
                    A_blocks[i] = A_blocks[i].dimshuffle('x', 'x')
                elif A_blocks[i].ndim == 1:
                    if state.ndim == 0:
                        A_blocks[i] = A_blocks[i].dimshuffle('x', 0)
                    else:
                        A_blocks[i] = A_blocks[i].dimshuffle(0, 'x')

            # shuffle derivative to column vector
            if state.ndim == 0:
                deriv = state_derivatives[state].dimshuffle('x', 'x')
            else:
                deriv = state_derivatives[state].dimshuffle(0, 'x')

            # Stack A_blocks
            A_row = T.concatenate(A_blocks, axis=1)
            b = deriv - T.dot(A_row, state_full)

            state_A.append(A_row)
            state_b.append(b)
        A = T.concatenate(state_A)
        b = T.concatenate(state_b)

        # Equation given by http://math.stackexchange.com/questions/1567784/matrix-differential-equation-xt-axtb-solution-defined-for-non-invertible/1567806?noredirect=1#comment3192556_1567806

        # Two methods to calculate the integral:

        # e^(at) method, given by http://wolfr.am/9mNgcOgM
        # eat_integral = Print("eat_integral")(T.dot(eat-1, T.inv(A)))
        # eat_integral = T.unbroadcast(eat_integral, 0, 1)

        # Taylor series method
        init_term = T.identity_like(A)*dt
        def series_advance(i, last_term, A, wrt):
            next_term = T.dot(last_term, A)*wrt/i
            return next_term, theano.scan_module.until(T.all(abs(next_term) < 10e-6))
        terms, _ = theano.scan(series_advance,
                               sequences=[T.arange(2, 200)],
                               non_sequences=[A, dt],
                               outputs_info=init_term,
                               )
        integral = T.sum(terms, axis=0) + init_term

        # Decide which integral to use, preferring the eat method when it works
        # integral = ifelse(T.any(T.isnan(eat_integral)), taylor_integral, eat_integral)

        new_state = (T.dot(slinalg.expm(A*dt), state_full) + T.dot(integral, b)).flatten()
        index = 0
        updates = []
        for state in states:
            if state.ndim == 0:
                state_len = 1
            else:
                state_len = T.shape(state)[0]
            state_update = new_state[index:index+state_len]
            state_update = state_update.reshape(T.shape(state))
            updates.append((state, state_update))
            index += state_len

        return theano.function([dt], [new_state], updates=updates, profile=False)