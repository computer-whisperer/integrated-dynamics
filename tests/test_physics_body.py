import math

from numpy.testing import assert_almost_equal

from int_dynamics.physics import *


def test_body_1():
    world = WorldBody()
    body = CubeBody(1, 1, 1, 1)
    world.add_free_child(body)
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

    world.add_free_child(body_1, PoseVector(XYZVector(0, 4, 0)), MotionVector(angular_component=XYZVector(math.pi, 0, 0)))
    body_1.add_fixed_child(body_2, PoseVector(XYZVector(3, 0, 0), Versor(XYZVector(0, 1, 0), math.pi / 2)))
    body_2.add_fixed_child(body_3, PoseVector(XYZVector(0, 0, 2)))
    body_2.add_fixed_child(body_4, PoseVector(XYZVector(2, 0, 0), Versor(XYZVector(1, 0, 0), math.pi / 4)))
    body_4.add_fixed_child(body_5, PoseVector(XYZVector(0, 5, 0)))

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


