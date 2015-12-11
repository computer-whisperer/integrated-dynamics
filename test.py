from dynamics import get_dynamics
from theano.compile.debugmode import DebugMode
import time

robot_dynamics_1 = get_dynamics()

for t in range(2):
 print("\n Iteration {} \n".format(t))
 print(robot_dynamics_1.update_physics(.05))
 #time.sleep(.1)