from int_dynamics.physics import *
import math

world = WorldBody()
base = CubeBody(0.2, 0.2, 0.2, 1, name="base")
world.add_child(base, Joint.fixed_joint(joint_base_lin=[0, 2, 0]))

link_count = 6
link_len = 4/link_count
prev_body = base
for link_num in range(link_count):
    joint = Joint.elbow_joint(
        joint_base_lin=[0, -link_len/2, 0],
        joint_pose_ang=[math.sqrt(2)/2, 0, 0, math.sqrt(2)/2],
        body_pose_lin=[0, -link_len/2, 0],
        axis='z')
    link = CubeBody(0.1, link_len, 0.1, 1, name="link_{}".format(link_num))
    link.forces.append((link.point, world.frame.y * (-9.81) * link.body_mass))
    prev_body.add_child(link, joint)
    prev_body = link

builder = EquationBuilder("rope")
builder.build_simulation_expressions(world)
builder.build_simulation_function("theano")
#integrator.current_state = [1.00,  -0.00, 0.00,  -0.09,  -0.00,  0.00,  -3.41]

renderer = OpenGLRenderer(builder)
renderer.main_loop()

