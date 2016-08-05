import time

from numpy.testing import assert_almost_equal
import numpy as np

from int_dynamics.physics import *
import math


def test_body_1():
    world = WorldBody()
    body = CubeBody(1, 1, 1, 1)
    world.add_child(body, joint_pose=PoseVector(variable=True), joint_motion=MotionVector(variable=True))
    integrator = EulerIntegrator()
    print("begin expression build")
    integrator.build_simulation_expressions(world)
    print("begin function compile")
    integrator.build_simulation_functions()
    print("Starting simulation")
    start_time = time.time()
    while integrator.get_time() < 10:
        print("state at sim time {} was {}".format(integrator.get_time(), integrator.current_state))
        integrator.step_time()
    print("10 time steps took {} seconds".format(start_time))
    print("final state was {}".format(integrator.current_state))


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
        joint_pose=PoseVector(XYVector(0, 0, symbols=False), Angle(0, symbols=True, use_constant=True)),
        joint_motion=MotionVector(XYVector(0, 0, symbols=False), Angle(0, symbols=True, use_constant=False))
    )
    link_1.add_child(
        link_2,
        pose=PoseVector(XYVector(0, 5)),
        joint_base=PoseVector(XYVector(0, 5)),
        joint_pose=PoseVector(XYVector(0, 0, symbols=False), Angle(0, symbols=True, use_constant=True)),
        joint_motion=MotionVector(XYVector(0, 0, symbols=False), Angle(0, symbols=True, use_constant=False))
    )
    link_2.add_child(
        link_3,
        pose=PoseVector(XYVector(0, 5)),
        joint_base=PoseVector(XYVector(0, 5)),
        joint_pose=PoseVector(XYVector(0, 0, symbols=False), Angle(0, symbols=True, use_constant=True)),
        joint_motion=MotionVector(XYVector(0, 0, symbols=False), Angle(0, symbols=True, use_constant=False))
    )

    integrator = EulerIntegrator()
    integrator.init_symbols(world)
    accel_vector = [-1, 3, -5]
    force_vector, root_forces = world.get_inverse_dynamics(accel_vector)
    assert_almost_equal(np.array(Matrix(force_vector).evalf(subs=integrator.build_state_substitutions())).astype(np.float64)[:, 0], [-166.8333333, -83.4166667, -50.25])


def test_inverse_dynamics_articulated_3d():
    # five bodies, essentially a tank-drive robot
    world = WorldBody()

    chassis = CubeBody(10, 2, 10, 20, name="chassis")
    wheel_1 = CubeBody(1, 10, 10, 1, name="wheel_1")
    wheel_2 = CubeBody(1, 10, 10, 1, name="wheel_2")
    wheel_3 = CubeBody(1, 10, 10, 1, name="wheel_3")
    wheel_4 = CubeBody(1, 10, 10, 1.5, name="wheel_4")

    world.add_child(
        chassis,
        pose=PoseVector(XYZVector(0, 0)),
        joint_base=PoseVector(XYZVector(0, 0)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, symbols=True), Versor(XYZVector(0, 1, 0), math.pi/2, symbols=True)),
        joint_motion=MotionVector(XYZVector(0, 0, 0, symbols=True), XYZVector(0, 0, 0, symbols=True))
    )

    chassis.add_child(
        wheel_1,
        pose=PoseVector(XYVector(0, 0, 0)),
        joint_base=PoseVector(XYZVector(-5, 0, 5)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, symbols=False), Quaternion(1, 0, 0, 0, symbol_components="ab")),
        joint_motion=MotionVector(XYZVector(0, 0, 0, symbols=False), Quaternion(0, 0, 0, 0, symbol_components="b"))
    )
    chassis.add_child(
        wheel_2,
        pose=PoseVector(XYVector(0, 0, 0)),
        joint_base=PoseVector(XYZVector(5, 0, 5)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, symbols=False), Quaternion(1, 0, 0, 0, symbol_components="ab")),
        joint_motion=MotionVector(XYZVector(0, 0, 0, symbols=False), Quaternion(0, 0, 0, 0, symbol_components="b"))
    )
    chassis.add_child(
        wheel_3,
        pose=PoseVector(XYVector(0, 0, 0)),
        joint_base=PoseVector(XYZVector(5, 0, -5)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, symbols=False), Quaternion(1, 0, 0, 0, symbol_components="ab")),
        joint_motion=MotionVector(XYZVector(0, 0, 0, symbols=False), Quaternion(0, 0, 0, 0, symbol_components="b"))
    )
    chassis.add_child(
        wheel_4,
        pose=PoseVector(XYVector(0, 0, 0)),
        joint_base=PoseVector(XYZVector(-5, 0, -5)),
        joint_pose=PoseVector(XYZVector(0, 0, 0, symbols=False), Quaternion(1, 0, 0, 0, symbol_components="ab")),
        joint_motion=MotionVector(XYZVector(0, 0, 0, symbols=False), Quaternion(0, 0, 0, 0, symbol_components="b"))
    )

    integrator = EulerIntegrator()
    integrator.init_symbols(world)

    accel_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    force_vector, root_forces = world.get_inverse_dynamics(accel_vector)
    force_vector_mat = Matrix(force_vector)
    #sympy.preview(force_vector_mat, output="dvi")
    print(integrator.build_state_substitutions())
    assert_almost_equal(np.array(force_vector_mat.evalf(subs=integrator.build_state_substitutions())).astype(np.float64)[:, 0],
                        [0, 0, 0, 0, 0, -25, 0, 0, 0, 25])




