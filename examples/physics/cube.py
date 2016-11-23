from int_dynamics.physics import *
import math

world = WorldBody()
cube = CubeBody(0.2, 0.2, 0.2, 1, name="cube")

cube.forces.append((cube.point, world.frame.y * (-9.81) * cube.body_mass))

joint_1 = Joint.trans_joint()

world.add_child(
    cube,
    joint_1)

integrator = EquationBuilder("cube")
integrator.build_simulation_expressions(world, MotionVector(XYVector(0, 9.81), frame=world.frame))
integrator.build_simulation_function()

print("Starting simulation")
start_time = time.time()
while integrator.get_time() < 10:
    print("  ".join(["{0:0.2f}".format(float(i)) for i in integrator.current_state]))
    #verts = integrator.get_edges()
    integrator.step_time()
print("10 time steps took {} seconds".format(time.time()-start_time))
print("final state was {}".format(integrator.current_state))
