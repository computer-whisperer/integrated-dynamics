from numpy import *
from scipy.linalg import expm
import ilqg
"""
A demo of iLQG/DDP with a control-limited LTI system.
"""

def app_tile(A, reps):
    A_ = A[:]
    if A.ndim < len(reps):
        while len(A_.shape) < len(reps):
            A_ = expand_dims(A_, axis=len(A_.shape))
    return tile(A_, reps)

def lin_dyn_cst(x, u, A, B, Q, R, want_all=False):
    # For a PD quadratic u-cost
    # no cost (nans) is equivalent to u=0

    u[isnan(u)] = 0
    if not want_all:
        f = dot(x, A) + dot(u, B)
        v1 = sum(x * dot(x, Q), axis=1)
        v2 = sum(u * dot(u, R), axis=1)
        c = 0.5*v1 + 0.5*v2
        return f, c
    else:
        N = x.shape[0]
        fx = tile(A, [N, 1, 1])
        fu = tile(B, [N, 1, 1])
        cx = dot(x, Q)
        cu = dot(u, R)
        cxx = tile(Q, [N, 1, 1])
        cxu = tile(zeros(B.shape).T, [N, 1, 1])
        cuu = tile(R, [N, 1, 1])
        return fx, fu, None, None, None, cx, cu, cxx, cxu, cuu


print('A demonstration of the iLQG/DDP algorithm\n'
      'with a random control-limited time-invariant linear system.\n'
      'for details see\nTassa, Mansard & Todorov, ICRA 2014\n'
      '\"Control-Limited Differential Dynamic Programming\"\n')

# make stable linear dynamics
h = .01  # time step
n = 3  # state dimension
m = 2  # control dimension
A = random.randn(n, n)
A = A-A.T  # skew-symmetric = pure imaginary eigenvalues
A = expm(h*A)  # discrete time
B = h*random.randn(m, n)
#A = array([[1, -1.5e-4, -4.6e-5],
#           [1.5e-4,  1,  1.1e-4],
#           [4.5e-5, -1.1e-4,  1]])
#B = array([[-1.7e-5, 1.1e-4],
#           [1.5e-4, 4.4e-6],
#           [-1.4e-5, 3.7e-6]])

# quadratic costs
Q = h*eye(n)
R = .1*h*eye(m)

# control limits
#Op.lims = ones(m,1)*[-1 1]*.6;

# optimization problem
dyncst = lambda x, u, i, want_all=False: lin_dyn_cst(x, u, A, B, Q, R, want_all)
T = 100  # horizon
x0 = random.randn(n)  # initial state
u0 = .1*random.randn(T, m)  # initial controls
#x0 = array([[ 0.07919485],
#      [-0.39150426],
#      [ 0.39676904]])
#u0 = array([[0.08702516, -0.10695812, -0.03761507, 0.00790764, 0.00532442, 0.08218673, -0.04070015, 0.05781017, 0.0340251, 0.15567711],
#            [-0.16786378, 0.08034461, -0.23664327, 0.16031643, 0.15972222, 0.00393588, -0.01797945, -0.14965136, 0.13926328, -0.00071236]])
#u0 = tile(u0, (1, T/10))
# run the optimization
x, u, L, Vx, Vxx, cost = ilqg.ilqg(dyncst, x0, u0, {})
#print(L[:,:,-1])