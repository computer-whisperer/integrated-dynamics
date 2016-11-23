import pickle
import time
import math
from int_dynamics.physics import OpenGLRenderer

start_time = time.time()
print("loading equation builder from cache")
with open("autocache.pkl", "rb") as f:
    builder = pickle.load(f)
print("load took {} seconds".format(time.time() - start_time))

builder.build_simulation_function()
builder.current_state = [math.sqrt(2)/2, 0, 0, math.sqrt(2)/2, 0, 0, 0]

renderer = OpenGLRenderer(builder)
renderer.main_loop()
