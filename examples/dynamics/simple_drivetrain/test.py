import time
from shooter_dynamics import get_dynamics

start_time = time.time()
robot_dynamics_1 = get_dynamics()
print("going\n")
execute_start = time.time()
iterations = 20
for t in range(iterations):
    robot_dynamics_1.update_physics(.05)
    print(robot_dynamics_1.get_state())
print("\nCompilation took {} seconds.".format(execute_start-start_time))
execute_time = time.time() - execute_start
print("Execution took {} seconds total, {} per iteration".format(execute_time, execute_time/iterations))
