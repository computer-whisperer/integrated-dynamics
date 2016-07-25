import time

from numpy.testing import assert_almost_equal

from int_dynamics.physics import *


def test_body_1():
    world = WorldBody()
    body = CubeBody(1, 1, 1, 1)
    world.add_child(body)
    integrator = EulerIntegrator()
    integrator.build_simulation_tensors(world)
    while integrator.get_time() < 10:
        integrator.step_time()


def test_composite_body_positions():

    world = WorldBody()

    body_1 = CubeBody(1, 1, 1, 1)
    body_2 = CubeBody(1, 1, 1, 1)
    body_3 = CubeBody(1, 1, 1, 1)
    body_4 = CubeBody(1, 1, 1, 1)
    body_5 = CubeBody(1, 1, 1, 1)

    world.add_child(body_1, joint_pose=PoseVector(XYZVector(0, 4, 0)), joint_motion=MotionVector(angular_component=XYZVector(math.pi, 0, 0)))
    body_1.add_child(body_2, joint_pose=PoseVector(XYZVector(3, 0, 0), Versor(XYZVector(0, 1, 0), math.pi / 2)))
    body_2.add_child(body_3, joint_pose=PoseVector(XYZVector(0, 0, 2)))
    body_2.add_child(body_4, joint_pose=PoseVector(XYZVector(2, 0, 0), Versor(XYZVector(1, 0, 0), math.pi / 4)))
    body_4.add_child(body_5, joint_pose=PoseVector(XYZVector(0, 5, 0)))

    integrator = EulerIntegrator()
    integrator.build_simulation_tensors(world)

    s2o2 = math.sqrt(2) / 2
    assert_almost_equal(body_1.frame.root_pose.get_ndarray(), [0, 0, 4, 0, 1, 0, 0, 0])
    assert_almost_equal(body_2.frame.root_pose.get_ndarray(), [0, 3, 4, 0, s2o2, 0, s2o2, 0])
    assert_almost_equal(body_3.frame.root_pose.get_ndarray(), [0, 5, 4, 0, s2o2, 0, s2o2, 0])
    assert_almost_equal(body_4.frame.root_pose.get_ndarray(), [0, 3, 4, -2, 0.6532815, 0.2705981,  0.6532815, -0.2705981])
    assert_almost_equal(body_5.frame.root_pose.get_ndarray(), [0, 3 + 5 * s2o2, 4 + 5 * s2o2, -2, 0.6532815, 0.2705981,  0.6532815, -0.2705981])
    assert_almost_equal(body_1.frame.root_motion.get_ndarray(), [0, 0, 0, 0, 0, math.pi, 0, 0])
    assert_almost_equal(body_2.frame.root_motion.get_ndarray(), [0, 0, 0, 0, 0, math.pi, 0, 0])
    assert_almost_equal(body_3.frame.root_motion.get_ndarray(), [0, 0, 0, 0, 0, math.pi, 0, 0])
    assert_almost_equal(body_4.frame.root_motion.get_ndarray(), [0, 0, 0, 0, 0, math.pi, 0, 0])
    assert_almost_equal(body_5.frame.root_motion.get_ndarray(), [0, 0, 0, 0, 0, math.pi, 0, 0])


def test_inverse_dynamics_articulated_2d():
    # Three links, essentially a 2D triple inverted pendelum
    world = WorldBody()

    link_1 = CubeBody(1, 10, 1, 1)
    link_2 = CubeBody(1, 10, 1, 1)
    link_3 = CubeBody(1, 10, 1, 1)

    world.add_child(
        link_1,
        pose=PoseVector(XYVector(0, 5)),
        joint_base=PoseVector(XYVector(0, 0)),
        joint_pose=PoseVector(XYVector(0, 0, variable=False), Angle(0, variable=True, use_constant=True)),
        joint_motion=MotionVector(XYVector(0, 0, variable=False), Angle(0, variable=True, use_constant=False))
    )
    link_1.add_child(
        link_2,
        pose=PoseVector(XYVector(0, 5)),
        joint_base=PoseVector(XYVector(0, 5)),
        joint_pose=PoseVector(XYVector(0, 0, variable=False), Angle(0, variable=True, use_constant=True)),
        joint_motion=MotionVector(XYVector(0, 0, variable=False), Angle(0, variable=True, use_constant=False))
    )
    link_2.add_child(
        link_3,
        pose=PoseVector(XYVector(0, 5)),
        joint_base=PoseVector(XYVector(0, 5)),
        joint_pose=PoseVector(XYVector(0, 0, variable=False), Angle(0, variable=True, use_constant=True)),
        joint_motion=MotionVector(XYVector(0, 0, variable=False), Angle(0, variable=True, use_constant=False))
    )

    integrator = EulerIntegrator()
    integrator.build_simulation_tensors(world)
    accel_vector = ExplicitMatrix([[0, 0, 0]])
    force_vector, root_forces = world.get_inverse_dynamics(accel_vector, MotionVector(frame=world.frame))
    force_vector_array = force_vector.get_symbolic_array()
    func = build_symbolic_function(force_vector_array)
    print(func())
    print(count_nodes(force_vector_array))


def test_inverse_dynamics_articulated_3d():
    # five bodies, essentially a tank-drive robot
    world = WorldBody()

    chassis = CubeBody(10, 2, 10, 20)
    wheel_1 = CubeBody(1, 10, 10, 1)
    wheel_2 = CubeBody(1, 10, 10, 1)
    wheel_3 = CubeBody(1, 10, 10, 1)
    wheel_4 = CubeBody(1, 10, 10, 1)

    world.add_child(
        chassis,
        pose=PoseVector(XYZVector(0, 0)),
        joint_base=PoseVector(XYZVector(0, 0)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, variable=True), Versor(XYZVector(), 0, variable=True)),
        joint_motion=MotionVector(XYZVector(0, 0, 0, variable=True), XYZVector(0, 0, 0, variable=True))
    )

    chassis.add_child(
        wheel_1,
        pose=PoseVector(XYVector(0, 0, 0)),
        joint_base=PoseVector(XYZVector(-5, 0, 5)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, variable=False), Quaternion(0, 0, 0, 0, variables="ad")),
        joint_motion=MotionVector(XYZVector(0, 0, 0, variable=False), Quaternion(0, 0, 0, 0, variables="d"))
    )
    chassis.add_child(
        wheel_2,
        pose=PoseVector(XYVector(0, 0, 0)),
        joint_base=PoseVector(XYZVector(5, 0, 5)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, variable=False), Quaternion(0, 0, 0, 0, variables="ad")),
        joint_motion=MotionVector(XYZVector(0, 0, 0, variable=False), Quaternion(0, 0, 0, 0, variables="d"))
    )
    chassis.add_child(
        wheel_3,
        pose=PoseVector(XYVector(0, 0, 0)),
        joint_base=PoseVector(XYZVector(5, 0, -5)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, variable=False), Quaternion(0, 0, 0, 0, variables="ad")),
        joint_motion=MotionVector(XYZVector(0, 0, 0, variable=False), Quaternion(0, 0, 0, 0, variables="d"))
    )
    chassis.add_child(
        wheel_4,
        pose=PoseVector(XYVector(0, 0, 0)),
        joint_base=PoseVector(XYZVector(-5, 0, -5)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, variable=False), Quaternion(0, 0, 0, 0, variables="ad")),
        joint_motion=MotionVector(XYZVector(0, 0, 0, variable=False), Quaternion(0, 0, 0, 0, variables="d"))
    )


    integrator = EulerIntegrator()
    integrator.build_simulation_tensors(world)
    accel_vector = ExplicitMatrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    force_vector, root_forces = world.get_inverse_dynamics(accel_vector, MotionVector(frame=world.frame))
    force_vector_array = force_vector.get_symbolic_array()
    func = build_symbolic_function(force_vector_array)
    start = time.time()
    print(func())
    print("took {}".format(time.time() - start))
    print(count_nodes(force_vector_array))







