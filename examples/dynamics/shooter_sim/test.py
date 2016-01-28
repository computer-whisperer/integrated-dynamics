import time
from shooter_dynamics import get_dynamics

robot_dynamics_1 = get_dynamics()

print("\n2 seconds of spinning up. Each frame is half a second.\n")

for _ in range(4):
    robot_dynamics_1.update_physics(.5)
    print(robot_dynamics_1.get_state())

robot_dynamics_1.add_ball()

print("\nEngaging ball! Assuming instant ball acceleration. Each frame is now one millisecond.\n")

for _ in range(8):
    robot_dynamics_1.update_physics(.01)
    print(robot_dynamics_1.get_state())