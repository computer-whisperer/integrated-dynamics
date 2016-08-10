import pickle
import time

start_time = time.time()
print("loading integrator from cache")
with open("autocache_dual_pendelum.pkl", "rb") as f:
    integrator = pickle.load(f)
print("load took {} seconds".format(time.time() - start_time))

integrator.build_simulation_function()

print("Starting simulation")
start_time = time.time()
while integrator.get_time() < 10:
    print("  ".join(["{0:0.2f}".format(float(i)) for i in integrator.current_state]))
    integrator.step_time()
print("10 time steps took {} seconds".format(time.time()-start_time))
print("final state was {}".format(integrator.current_state))
