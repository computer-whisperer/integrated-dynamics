from int_dynamics.physics import *
import math

world = WorldBody()
link_1 = CubeBody(0.1, 1, 0.1, 1, name="link_1")
link_1.forces.append((link_1.point, world.frame.y * (-9.81) * link_1.body_mass))

joint_1 = Joint.ball_joint(
    joint_base_lin=[0, 2, 0],
    joint_pose_ang=[math.sqrt(3)/3, math.sqrt(3)/3, 0, math.sqrt(3)/3],
    joint_motion_ang=[0, 0, 0],
    body_pose_lin=[0, -0.5, 0],
#    axis='x'
)

world.add_child(
    link_1,
    joint_1)

link_2 = CubeBody(0.1, 1, 0.1, 1, name="link_2")
link_2.forces.append((link_2.point, world.frame.y * (-9.81) * link_2.body_mass))

joint_2 = Joint.elbow_joint(
    joint_base_lin=[0, -0.5, 0],
    joint_pose_ang=[0, 0, 0, -1],
    joint_motion_ang=[0, 0, -0.5],
    body_pose_lin=[0, -0.5, 0], axis='z'
)

# link_1.add_child(
#     link_2,
#     joint_2)

link_3 = CubeBody(0.1, 1, 0.1, 1, name="link_3")
link_3.forces.append((link_3.point, world.frame.y * (-9.81) * link_3.body_mass))

joint_3 = Joint.elbow_joint(
    joint_base_lin=[0, -0.5, 0],
    joint_pose_ang=[0, 0, 0, -1],
    joint_motion_ang=[0, 0, -0.5],
    body_pose_lin=[0, -0.5, 0], axis='z'
)

#link_2.add_child(
#    link_3,
#    joint_3)

#
# link_4 = CubeBody(0.1, 1, 0.1, 1, name="link_3")
# link_4.forces.append((link_4.point, world.frame.y * (-9.81) * link_4.body_mass))
#
# joint_4 = Joint.elbow_joint(
#     joint_base_lin=[0, -0.5, 0],
#     joint_pose_ang=[0, 0, -1],
#     joint_motion_ang=[0, 0, -0.5],
#     body_pose_lin=[0, -0.5, 0], axis='z'
# )
#
# link_3.add_child(
#     link_4,
#     joint_4)

builder = EquationBuilder("dual_pendulum")
builder.build_simulation_expressions(world)
builder.build_simulation_function()
#integrator.current_state = [1.00,  -0.00, 0.00,  -0.09,  -0.00,  0.00,  -3.41]

renderer = OpenGLRenderer(builder)
renderer.main_loop()

