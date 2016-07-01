from dynamics import MyRobotDynamics
import logging
logging.getLogger().setLevel(logging.DEBUG)
lift = MyRobotDynamics.cached_init("simulation")
lift.controllers["motor 1"].set_percent_vbus(1)
lift.controllers["motor 2"].set_percent_vbus(1)

time = 0
for _ in range(500):
    lift.simulation_update(.02)
    time += 0.02
    state = lift.get_state()
    print(state)
    if state['lift']['position'] >= 2.5:
        break
else:
    exit()
print("Lifted in {:.02} seconds.".format(time))