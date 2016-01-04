from numpy import *
from ilqg import ilqg
# A demo of iLQG/DDP with car-parking dynamics

def finite_difference(fun, x, h=2e-6):
    # simple finite-difference derivatives
    # assumes the function fun() is vectorized

    K, n = x.shape
    H = vstack((zeros(n), h*eye(n)))
    X = x[:, None, :] + H[None, :, :]
    Y = []
    for i in range(K):
        Y.append(fun(X[i]))
    Y = array(Y)
    D = (Y[:, 1:] - Y[:, 0:1])
    J = D/h
    return J


def dyn_cst(x, u, want_all=False):
    # combine car dynamics and cost
    # use helper function finite_difference() to compute derivatives

    if not want_all:
        f = car_dynamics(x, u)
        c = car_cost(x, u)
        return f, c
    else:

        # dynamics derivatives
        xu_dyn = lambda xu: car_dynamics(xu[:, 0:4], xu[:, 4:6])
        J = finite_difference(xu_dyn, hstack((x, u)))
        fx = J[:, 0:4]
        fu = J[:, 4:6]

        xu_Jdyn = lambda xu: finite_difference(xu_dyn, xu)
        JJ = finite_difference(xu_Jdyn, hstack((x, u)))
        JJ = 0.5*(JJ + JJ.transpose([0, 2, 1, 3]))
        fxx = JJ[:, 0:4, 0:4]
        fxu = JJ[:, 0:4, 4:6]
        fuu = JJ[:, 4:6, 4:6]

        # cost first derivatives
        xu_cost = lambda xu: car_cost(xu[:, 0:4], xu[:, 4:6])
        J = finite_difference(xu_cost, hstack((x, u)))
        cx = J[:, 0:4]
        cu = J[:, 4:6]

        # cost second derivatives
        xu_Jcst = lambda xu: finite_difference(xu_cost, xu)
        JJ = finite_difference(xu_Jcst, hstack((x, u)))
        JJ = 0.5*(JJ + JJ.transpose([0, 2, 1]))
        cxx = JJ[:, 0:4, 0:4]
        cxu = JJ[:, 0:4, 4:6]
        cuu = JJ[:, 4:6, 4:6]

        return fx, fu, None, None, None, cx, cu, cxx, cxu, cuu


def dynamics(x, u):

    # === states and controls:
    # x = [x y r]' = [x y r]
    # u = [r l]'     = [right_wheel_out left_wheel_out]

    # constants
    h = 0.05     # h = timestep (seconds)

    # controls
    r_out = u[:, 0]  # w = right wheel out
    l_out = u[:, 1]  # a = left wheel out

    vel, rot = drivetrains.two_motor_drivetrain(l_out, r_out)

    r = x[:, 2] + h*rot  # r = car angle
    z = vstack((cos(r), sin(r))) * h*vel

    dy = vstack([z[0], z[1], rot]).T  # change in state
    y = x + dy  # new state
    return y


def cost(x, u):
    # cost function for car-parking problem
    # sum of 3 terms:
    # lu: quadratic cost on controls
    # lf: final cost on distance from target parking configuration
    # lx: small running cost on distance from origin to encourage tight turns
    final = isnan(u[:, 0])
    u[final, :] = 0

    cu = 1e-1         # control cost coefficients

    cf = array([.1,  .1,   1])  # final cost coefficients
    pf = array([.01, .01, .01]).conj().T  # smoothness scales for final cost

    cx = 1e-3*array([1, 1, 1])  # running cost coefficients
    px = array([.1, .1, .1]).conj().T  # smoothness scales for running cost

    # control cost
    lu = sum(cu*u*u, axis=1)

    # final cost
    if any(final):
        lf = final * dot(cf, sabs(x[final], pf).T)
    else:
        lf = 0

    # running cost
    lx = sum(cx * sabs(x[:, 0:3], px), axis=1)
    # lx = sum(x*x, axis=1)

    # total cost
    c = lu + lx + lf
    return c


def sabs(x, p):
    # smooth absolute-value function (a.k.a pseudo-Huber)
    return sqrt(x*x + p*p) - p


def car_dynamics(x, u):
    
    # === states and controls:
    # x = [x y t v]' = [x y car_angle front_wheel_velocity]
    # u = [w a]'     = [front_wheel_angle accelaration]
    
    # constants
    d  = 2.0      # d = distance between back and front axles
    h  = 0.03     # h = timestep (seconds)
    
    # controls
    w  = u[:, 0] # w = front wheel angle
    a  = u[:, 1] # a = front wheel acceleration
    
    o  = x[:, 2] # o = car angle
                  # z = unit_vector(o)
    z = vstack((cos(o), sin(o)))
    
    v  = x[:, 3] # v = front wheel velocity
    f  = h*v      # f = front wheel rolling distance
                   # b = back wheel rolling distance
    b  = d + f*cos(w) - sqrt(d**2 - (f*sin(w))**2)
                   # do = change in car angle
    dod = arcsin(sin(w)*f/d)
    m = b*z
    dy = vstack([m[0], m[1], dod, h*a]).T   # change in state
    y  = x + dy                # new state
    return y


def car_cost(x, u):
    # cost function for car-parking problem
    # sum of 3 terms:
    # lu: quadratic cost on controls
    # lf: final cost on distance from target parking configuration
    # lx: small running cost on distance from origin to encourage tight turns
    
    final = isnan(u[:, 0])
    u[final, :]  = 0
    
    cu  = 1e-2*array([1, .01])         # control cost coefficients
    
    cf  = array([ .1,  .1,   1,  .3])    # final cost coefficients
    pf  = array([.01, .01, .01,  1]).T    # smoothness scales for final cost
    
    cx  = 1e-3*array([1,  1])          # running cost coefficients
    px  = array([.1, .1]).T             # smoothness scales for running cost
    
    # control cost
    lu    = dot(u**2, cu)
    
    # final cost
    if any(final):
       lf      = final * dot(cf, sabs(x[final], pf).T)
    else:
       lf    = 0
    
    
    # running cost
    lx = dot(sabs(x[:, 0:2],px), cx)
    
    # total const
    c     = lu + lx + lf
    return c


# Set full_DDP=true to compute 2nd order derivatives of the
# dynamics. This will make iterations more expensive, but
# final convergence will be much faster (quadratic)
full_DDP = True

# optimization problem
DYNCST  = lambda x, u, i, want_all=False: dyn_cst(x, u, want_all)
T       = 500              # horizon
x0      = array([1, 1, pi*3/2, 0])   # initial state
u0      = .1*random.randn(T, 2)  # initial controls
#u0 = zeros((T, 2))
options = {}
options["lims"]  = array([[-.5, .5],         # wheel angle limits (radians)
                          [ -2,  2]])       # acceleration limits (m/s^2)

# run the optimization
#options["maxIter"] = 5
x, u, L, Vx, Vxx, cost = ilqg(DYNCST, x0, u0, options)
print("done")
## ======== graphics functions ========
#function h = car_plot(x,u)
#
#body        = [0.9 2.1 0.3]           # body = [width length curvature]
#bodycolor   = 0.5*[1 1 1]
#headlights  = [0.25 0.1 .1 body(1)/2] # headlights [width length curvature x]
#lightcolor  = [1 1 0]
#wheel       = [0.15 0.4 .06 1.1*body(1) -1.1 .9]  # wheels = [width length curvature x yb yf]
#wheelcolor  = 'k'
#
#h = []
#
## make wheels
#for front = 1:2
#   for right = [-1 1]
#      h(+1) = rrect(wheel,wheelcolor)' ##ok<AGROW>
#      if front == 2
#         twist(h(),0,0,u(1))
#      
#      twist(h(),right*wheel(4),wheel(4+front))
#   
#
#
## make body
#h(+1) = rrect(body,bodycolor)
#
## make window (hard coded)
#h(+1) = patch([-.8 .8 .7 -.7],.6+.3*[1 1 -1 -1],'w')
#
## headlights
#h(+1) = rrect(headlights(1:3),lightcolor)
#twist(h(),headlights(4),body(2)-headlights(2))
#h(+1) = rrect(headlights(1:3),lightcolor)
#twist(h(),-headlights(4),body(2)-headlights(2))
#
## put rear wheels at (0,0)
#twist(h,0,-wheel(5))
#
## align to x-axis
#twist(h,0,0,-pi/2)
#
## make origin (hard coded)
#ol = 0.1
#ow = 0.01
#h(+1) = patch(ol*[-1 1 1 -1],ow*[1 1 -1 -1],'k')
#h(+1) = patch(ow*[1 1 -1 -1],ol*[-1 1 1 -1],'k')
#
#twist(h,x(1),x(2),x(3))
#
#function twist(obj,x,y,theta)
## a planar twist: rotate object by theta, then translate by (x,y)
#i = 1i
#if nargin == 3
#   theta = 0
#
#for h = obj
#   Z = get(h,'xdata') + i*get(h,'ydata')
#   Z = Z * exp(i*theta)
#   Z = Z + (x + i*y)
#   set(h,'xdata',real(Z),'ydata',imag(Z))
#
#
#function h = rrect(wlc, C)
## draw a rounded rectangle
#if nargin == 1
#   C = 'w'
#
#
#N        = 25
#
#width    = wlc(1)
#length   = wlc(2)
#curve    = wlc(3)
#
#a        = linspace(0,2*pi,4*N)
#z        = curve*exp(1i*a)
#width    = width-curve
#length   = length-curve
#e        = sum( kron(diag(width*[1 -1 -1 1] + 1i*length *[1 1 -1 -1]), ones(1,N)), 1) 
#z        = z+e
#z        = [z z(1)]
#
#h        = patch(real(z),imag(z),C)

# utility functions, singleton-expanded addition and multiplication
def pp(a, b):
    return a+b

def tt(a, b):
    return a*b