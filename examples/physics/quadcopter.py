from int_dynamics.physics import *

blade_ct = 4
radius = 5

world = WorldBody()
quadcopter = CubeBody(radius*2, 1, radius*2, 2, name="quadcopter")
world.add_child(quadcopter, joint_pose=PoseVector(variable=True), joint_motion=MotionVector(variable=True))
blades = [CubeBody(2, 0.1, 2, 0.1, name="rotor_{}".format(i)) for i in range(blade_ct)]
quad_forces = []
blade_speeds = [0, 0, 0, 0]
for i in range(blade_ct):
    angle = i * (sympy.pi*2/blade_ct)
    blade_vel = Quaternion(0, 0, blade_speeds[i], 0, symbol_components="c")
    quadcopter.add_child(
        blades[i],
        joint_base=PoseVector(linear_component=XYZVector(sympy.sin(angle)*radius, 0, sympy.cos(angle)*radius)),
        joint_pose=PoseVector(variable=False),
        joint_motion=MotionVector(angular_component=blade_vel)
    )
    blade_vert_force = blade_vel.c * 1
    local_force = ForceVector(XYZVector(0, blade_vert_force, 0), frame=blades[i].frame)
    root_force = blades[i].frame.root_pose.transform_force(local_force)
    quad_forces.append(root_force)
quadcopter.forces.append(sum(quad_forces))

integrator = EquationBuilder("quadcopter")
integrator.build_simulation_expressions(world, MotionVector(XYVector(0, 9.81), frame=world.frame))
integrator.build_simulation_function()

print("Starting simulation")
start_time = time.time()
while integrator.get_time() < 10:
    print("  ".join(["{0:0.2f}".format(float(i)) for i in integrator.current_state]))
    integrator.step_time()
print("10 time steps took {} seconds".format(time.time()-start_time))
print("final state was {}".format(integrator.current_state))