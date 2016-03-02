from dynamics import MyRobotDynamics
import logging
logging.getLogger().setLevel(logging.DEBUG)
shooter = MyRobotDynamics("simulation")
shooter.controllers["shooter"].set_percent_vbus(1)

print("\n2 seconds of spinning up. Each frame is half a second.\n")

for _ in range(20):
    shooter.simulation_update(.1)
    print(shooter.get_state())

shooter.add_ball()

print("\nEngaging ball! Assuming instant ball acceleration. Each frame is now one millisecond.\n")

for _ in range(8):
    shooter.simulation_update(.01)
    print(shooter.get_state())