import pickle
import time
import math
from int_dynamics.physics import OpenGLRenderer

start_time = time.time()
print("loading equation builder from cache")
with open("autocache.pkl", "rb") as f:
    builder = pickle.load(f)
print("load took {} seconds".format(time.time() - start_time))

controllers = builder.conts

values = {
    "left_rear_thruster": -0.1,
    "right_rear_thruster": 0.1,
    "left_front_thruster": 0.1,
    "right_front_thruster": -0.1,
    "left_vert_thruster": 0,
    "right_vert_thruster": 0,
}

for name in values:
    controllers[name].set(values[name])

builder.build_simulation_function("theano")
#builder.current_state = [math.sqrt(2)/2, 0, 0, math.sqrt(2)/2, 0, 0, 0]

renderer = OpenGLRenderer(builder)
renderer.main_loop()
