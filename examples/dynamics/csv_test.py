import time
from csv import DictWriter
from shooter_dynamics import get_dynamics

start_time = time.time()
robot_dynamics_1 = get_dynamics()
print("going\n")
execute_start = time.time()
iterations = 2000
#print(robot_dynamics_1.update_physics(10))
with open("shooter.csv", 'w') as csvfile:
    writer = DictWriter(csvfile, ["position", "velocity"])
    writer.writeheader()
    for t in range(iterations):
        data = robot_dynamics_1.update_physics(.0001)
        writer.writerow(data)
print("\nCompilation took {} seconds.".format(execute_start-start_time))
execute_time = time.time() - execute_start
print("Execution took {} seconds total, {} per iteration".format(execute_time, execute_time/iterations))
