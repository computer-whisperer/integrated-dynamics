import theano
import theano.tensor as T

def build_ilqg(dynamics_function, cost_function, x0, u0):
    pass

def ilqg_pass(x, u, ):
    pass

def forward_pass(dynamics, cost, x0, u, L, x, du, alpha, lims):
    def control_advance(u, x, alpha, du, prev_x, prev_u):
        next_x = dynamics(prev_u, prev_x)
        next_u = u + T.dot(alpha, du) + T.dot(next_x - x, L.T)
        if lims is not None:
            next_u = T.clip(next_u, lims[:, 0], lims[:, 1])
        return next_x, next_u
    x_new, u_new = theano.scan(
        control_advance,
        sequences=[u, x, L],
        non_sequences=[alpha],
        outputs_info=[])[0]
    c_new = theano.scan(cost, sequences=[u, x])[0]
    return x_new, u_new, c_new

def back_pass():
    pass
