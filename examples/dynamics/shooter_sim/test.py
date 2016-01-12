import time
from shooter_dynamics import get_dynamics

robot_dynamics_1 = get_dynamics()
print("going\n")

for _ in range(6):
    robot_dynamics_1.update_physics(1)
    print(robot_dynamics_1.get_state())

print("\n Adding Ball! \n")
robot_dynamics_1.add_ball()

for _ in range(6):
    robot_dynamics_1.update_physics(.001)
    print(robot_dynamics_1.get_state())
