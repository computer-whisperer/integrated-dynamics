from dynamics import get_dynamics
import logging
logging.getLogger().setLevel(logging.DEBUG)
shooter = get_dynamics()

print("\n2 seconds of spinning up. Each frame is half a second.\n")

for _ in range(4):
    shooter.simulation_update(.5)
    print(shooter.get_state())

shooter.add_ball()

print("\nEngaging ball! Assuming instant ball acceleration. Each frame is now one millisecond.\n")

for _ in range(8):
    shooter.simulation_update(.01)
    print(shooter.get_state())