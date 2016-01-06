import time

from dynamics import get_dynamics

start_time = time.time()
robot_dynamics_1 = get_dynamics()
print("going\n")
execute_start = time.time()
iterations = 40
#print(robot_dynamics_1.update_physics(10))
for t in range(iterations):
    print(robot_dynamics_1.update_physics(1))
print("\nCompilation took {} seconds.".format(execute_start-start_time))
execute_time = time.time() - execute_start
print("Execution took {} seconds total, {} per iteration".format(execute_time, execute_time/iterations))
