from dynamics import MyRobotDynamics
import theano
import logging
logging.getLogger().setLevel(logging.DEBUG)
dynamics = MyRobotDynamics.cached_init("simulation")
#theano.printing.pydotprint(dynamics.state_prediction_mean_update, outfile="simulation_update.png", var_with_name_simple=True)
dynamics.controllers["left_cim"].set_percent_vbus(1)
dynamics.controllers["right_cim"].set_percent_vbus(-1)

for _ in range(200):
    dynamics.simulation_update(.1)
    print(dynamics.get_state())