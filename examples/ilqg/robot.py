import wpilib
from robotpy_ext.physics import drivetrains
from numpy import *
import time
from matplotlib import use
use("GTK3Agg")
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from int_dynamics import ilqg
import time


def dynamics_func(x, u, dt=.1):

    # === states and controls:
    # x = [x y r x' y' r']' = [x y r]
    # u = [r l]'     = [right_wheel_out left_wheel_out

    # controls
    l_out = u[0]  # w = right wheel out
    r_out = u[1]  # a = left wheel out

    return drivetrains.two_motor_drivetrain(x, dt, l_out, r_out)

#    theta = x[2] + dt*rot_spd
#    world_vel = array([sin(theta), cos(theta)]) * y_spd
#
#    pos = concatenate((x[:2] + world_vel*dt, theta[None]))

#    return pos


def cost_func(x, u):

    # running cost
    cost = 0
    #cost += max(3 - .1*linalg.norm(x[:2]-array([0, 5]))**2, 0)
    cost += 20**((-linalg.norm(x[:2]-array([-2, 5]))**2)/4**2)
    cost += 15**((-linalg.norm(x[:2]-array([2.5, 10]))**2)/4**2)
    cost += 15**((-linalg.norm(x[:2]-array([-2, 15]))**2)/4**2)
    cost += 15**((-linalg.norm(x[:2]-array([4, 18]))**2)/4**2)

    cost -= 20**((-linalg.norm(x[:2]-array([0, 20]))**2)/8**2)

    if any(isnan(u)):
        u[:] = 0
        cost += 3*sabs(linalg.norm(x[:2]-array([0, 20])), .1)

    #  control cost coefficients
    cost += dot(u*u, 4e-1*array([1, 1]))

    return cost


def sabs(x, p):
    # smooth absolute-value function (a.k.a pseudo-Huber)
    return sqrt(x*x + p*p) - p


def cost_graph(path):
    global ax
    #plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = arange(-10, 30, 0.25)
    Y = arange(-5, 25, 0.25)

    X, Y = meshgrid(X, Y)
    Z = zeros(X.shape)
    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            Z[x, y] = .2*(cost_func(array([X[x, y], Y[x, y], 0]), array([0, 0])))

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    path_z = ones(path.shape[0])
    for i in range(path.shape[0]):
        path_z[i] = .2*(.05+cost_func(array([path[i, 0], path[i, 1], 0]), array([0, 0])))

    path_line = ax.plot(path[:, 0], path[:, 1], path_z)

    plt.show()


path_line = None


class IlqgRobot(wpilib.IterativeRobot):

    def robotInit(self):
        # optimization problem
        T = 100              # horizon
        x0 = array([0,  0,  0])   # initial state
        u0 = .1*random.randn(T, 2)  # initial controls
        #u0 = zeros((T, 2))  # initial controls
        options = {}

        # run the optimization
        options["lims"] = array([[-1, 1],
                                 [-1, 1]])

        start_time = time.time()
        self.x, self.u, L, Vx, Vxx, cost = ilqg.ilqg(lambda x, u: dynamics_func(x, u), cost_func, x0, u0, options)
        self.i = 0
        print(self.x[-1])
        print("ilqg took {} seconds".format(time.time() - start_time))
        cost_graph(self.x)
        self.drive = wpilib.RobotDrive(0, 1)
        self.joystick = wpilib.Joystick(0)

    def autonomousInit(self):
        self.autostart = time.time()

    def autonomousPeriodic(self):
        time_elapsed = time.time() - self.autostart
        if time_elapsed < self.u.shape[0]*.1:
            self.drive.tankDrive(self.u[time_elapsed//.1, 0], -self.u[time_elapsed//.1, 1])
        else:
            self.drive.tankDrive(0, 0)

    def teleopPeriodic(self):
        self.drive.arcadeDrive(self.joystick)


if __name__ == "__main__":
    wpilib.run(IlqgRobot)
