from dynamics import MyRobotDynamics
import logging
logging.getLogger().setLevel(logging.DEBUG)
lift = MyRobotDynamics("simulation")

for _ in range(300):
    lift.simulation_update(.06)
    print(lift.get_state())