from dynamics import get_dynamics
from theano.compile.debugmode import DebugMode

robot_dynamics_1 = get_dynamics()

#for t in range(20):
print(robot_dynamics_1.update_physics(.05))