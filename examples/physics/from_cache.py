import pickle
import time
import math

start_time = time.time()
print("loading equation builder from cache")
with open("autocache.pkl", "rb") as f:
    equation_builder = pickle.load(f)
print("load took {} seconds".format(time.time() - start_time))

equation_builder.build_simulation_function()
equation_builder.current_state = [math.sqrt(2)/2, 0, 0, math.sqrt(2)/2, 0, 0, 0]

print("Starting simulation")
start_time = time.time()
while equation_builder.get_time() < 10:
    print("  ".join(["{0:0.2f}".format(float(i)) for i in equation_builder.current_state]))
    equation_builder.step_time()
print("10 time steps took {} seconds".format(time.time()-start_time))
print("final state was {}".format(equation_builder.current_state))
