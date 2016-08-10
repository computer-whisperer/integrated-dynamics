from int_dynamics.physics import *
import math

world = WorldBody()
link_1 = CubeBody(0.1, 1, 0.1, 1, name="link_1")
link_2 = CubeBody(0.1, 1, 0.1, 1, name="link_2")


world.add_child(
    link_1,
    pose=PoseVector(linear_component=XYVector(0, -0.5)),
    joint_base=PoseVector(linear_component=XYVector(0, 1)),
    joint_pose=PoseVector(angular_component=Angle(math.pi/6, symbols=True, use_constant=True)),
    joint_motion=MotionVector(angular_component=Angle(0, symbols=True, use_constant=False))
)

link_1.add_child(
    link_2,
    pose=PoseVector(linear_component=XYVector(0, -0.5)),
    joint_base=PoseVector(linear_component=XYVector(0, -0.5)),
    joint_pose=PoseVector(angular_component=Angle(0, symbols=True, use_constant=True)),
    joint_motion=MotionVector(angular_component=Angle(0, symbols=True, use_constant=False))
)

integrator = EulerIntegrator("dual_pendulum")
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
