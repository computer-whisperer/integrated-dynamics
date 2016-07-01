import math

from numpy.testing import assert_almost_equal

from int_dynamics.physics import physics2d
from int_dynamics.physics.types import *


def test_body_1():
    world = physics2d.WorldBody()
    body = physics2d.RectangularBody(1, 1, 1)
    world.add_free_child(body)
    integrator = physics2d.EulerIntegrator()
    integrator.build_simulation_tensors(world)
    while integrator.get_time() < 10:
        integrator.step_time()


def test_composite_body_positions():
    world = physics2d.WorldBody()

    body_1 = physics2d.RectangularBody(1, 1, 1)
    body_2 = physics2d.RectangularBody(1, 1, 1)
    body_3 = physics2d.RectangularBody(1, 1, 1)
    body_4 = physics2d.RectangularBody(1, 1, 1)
    body_5 = physics2d.RectangularBody(1, 1, 1)

    world.add_free_child(body_1, XYVector(0, 4))
    body_1.add_fixed_child(body_2, PoseVector(XYVector(3, 0), Angle(math.pi/2)))
    body_2.add_fixed_child(body_3, PoseVector(XYVector(2, 0), Angle(0)))
    body_2.add_fixed_child(body_4, PoseVector(XYVector(0, 2), Angle(math.pi/4)))
    body_4.add_fixed_child(body_5, PoseVector(XYVector(0, 5), Angle(0)))

    integrator = physics2d.EulerIntegrator()
    integrator.build_simulation_tensors(world)

    assert_almost_equal(body_1.root_pos.get_ndarray(), [0, 4])
    assert_almost_equal(body_2.root_pos.get_ndarray(), [3, 4])
    assert_almost_equal(body_3.root_pos.get_ndarray(), [3, 6])
    assert_almost_equal(body_4.root_pos.get_ndarray(), [1, 4])
    assert_almost_equal(body_5.root_pos.get_ndarray(), [1-5*math.sqrt(2)/2, 4-5*math.sqrt(2)/2])


def test_composite_body_velocities():
    world = physics2d.WorldBody()

    body_1 = physics2d.RectangularBody(1, 1, 1)
    body_2 = physics2d.RectangularBody(1, 1, 1)
    body_3 = physics2d.RectangularBody(1, 1, 1)
    body_4 = physics2d.RectangularBody(1, 1, 1)
    body_5 = physics2d.RectangularBody(1, 1, 1)

    world.add_free_child(body_1, XYVector(0, 4), Angle(0), XYVector(0, 0), Angle(math.pi))
    body_1.add_fixed_child(body_2, XYVector(3, 0))
    body_1.add_fixed_child(body_3, XYVector(-3, 0))
    body_1.add_fixed_child(body_4, XYVector(0, 4), Angle(math.pi/4))
    body_3.add_free_child(body_5, XYVector(3, 4), Angle(0), XYVector(0, -1), Angle(-math.pi))

    integrator = physics2d.EulerIntegrator()
    integrator.build_simulation_tensors(world)

    assert_almost_equal(body_1.root_vel.get_ndarray(), [0, 0])
    assert_almost_equal(body_1.root_rvel.get_ndarray(), [math.pi])

    assert_almost_equal(body_2.root_vel.get_ndarray(), [0, math.pi*3])
    assert_almost_equal(body_2.root_rvel.get_ndarray(), [math.pi])

    assert_almost_equal(body_3.root_vel.get_ndarray(), [0, -math.pi*3])
    assert_almost_equal(body_3.root_rvel.get_ndarray(), [math.pi])

    assert_almost_equal(body_4.root_vel.get_ndarray(), [-math.pi*4, 0])
    assert_almost_equal(body_4.root_rvel.get_ndarray(), [math.pi])

    assert_almost_equal(body_5.root_vel.get_ndarray(), [-math.pi*4, -1])
    assert_almost_equal(body_5.root_rvel.get_ndarray(), [0])
