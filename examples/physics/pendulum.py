from int_dynamics.physics import *
import math

world = WorldBody()
link_1 = CubeBody(0.1, 1, 0.1, 1, name="link_1")
link_2 = CubeBody(0.5, 0.5, 0.1, 2, name="link_2")


world.add_child(
    link_1,
    pose=PoseVector(linear_component=XYVector(0, -0.75)),
    joint_base=PoseVector(linear_component=XYVector(0, 2), angular_component=Versor(XYZVector(0, 1, 0), 0)),
    joint_pose=PoseVector(angular_component=Versor(XYZVector(0, 0, 1), math.pi/3, symbols=True)),
    joint_motion=MotionVector(angular_component=XYZVector(0.75, 0, 0, symbols=True))
)

link_1.add_child(
    link_2,
    pose=PoseVector(linear_component=XYVector(0, -0.5)),
    joint_base=PoseVector(linear_component=XYVector(0, -0.5)),
    joint_pose=PoseVector(angular_component=Angle(0, symbols=False, use_constant=True)),
    joint_motion=MotionVector(angular_component=Angle(0, symbols=False, use_constant=False))
)

integrator = EulerIntegrator("dual_pendulum")
integrator.build_simulation_expressions(world, MotionVector(XYVector(0, 9.81), frame=world.frame))
integrator.build_simulation_function()
#integrator.current_state = [1.00,  -0.00, 0.00,  -0.09,  -0.00,  0.00,  -3.41]

print("Starting simulation")
start_time = time.time()
while integrator.get_time() < 10:
    print("  ".join(["{0:0.2f}".format(float(i)) for i in integrator.current_state]))
    #verts = integrator.get_edges()
    integrator.step_time()
print("10 seconds of sim took {} seconds".format(time.time()-start_time))
print("final state was {}".format(integrator.current_state))
