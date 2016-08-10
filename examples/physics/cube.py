from int_dynamics.physics import *
import math

world = WorldBody()
cube = CubeBody(0.2, 0.2, 0.2, 1, name="cube")


world.add_child(
    cube,
    pose=PoseVector(linear_component=XYVector(0, 0)),
    joint_base=PoseVector(linear_component=XYVector(0, 0)),
    joint_pose=PoseVector(linear_component=Quaternion(0, 0, 0, 0, symbol_components=""), angular_component=Quaternion(1, 0, 0, 0, symbol_components="abcd")),
    joint_motion=MotionVector(linear_component=Quaternion(0, 0, 0, 0, symbol_components=""), angular_component=Quaternion(0, 0, 0.5, 1, symbol_components="d"))
)

integrator = EulerIntegrator("cube")
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
