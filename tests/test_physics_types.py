import math

import numpy as np
from numpy.testing import assert_almost_equal

from int_dynamics.physics.types import *


def test_versors():
    v = XYZVector(0, 0, 1)
    s2o2 = math.sqrt(2)/2

    q1 = Versor(XYZVector(0, 1, 0), math.pi/4)
    v1 = q1 * v * q1.transpose()
    assert_almost_equal(v1.get_ndarray(), np.array([0, s2o2, 0, s2o2]))

    q2 = Versor(XYZVector(-1, 0, 0), math.pi/4)
    v2 = q2 * v * q2.transpose()
    assert_almost_equal(v2.get_ndarray(), np.array([0, 0, s2o2, s2o2]))

    q3 = q1*q2
    v3 = q3 * v * q3.transpose()
    assert_almost_equal(v3.get_ndarray(), np.array([0, 0.5, s2o2, 0.5]))


def test_pose_vector_init():
    p1 = PoseVector(XYZVector(0, 1, 0), Versor(XYZVector(0, 1, 0), math.pi/4))
    assert_almost_equal(p1.get_ndarray(), np.array([0, 0, 1, 0, 0.92387953251128674, 0.0, 0.38268343236508978, 0.0]))


def test_pose_vector_transform_pose():
    f1 = Frame("f1")
    f2 = Frame("f2")
    p1 = PoseVector(XYZVector(0, 1, 0), Versor(XYZVector(0, 1, 0), math.pi/4), frame=f1, end_frame=f2)
    p2 = PoseVector(frame=f2)
    p3 = PoseVector(XYZVector(100, 0, 0), frame=f2)
    p4 = PoseVector(angular_component=Versor(XYZVector(0, 1, 0), -math.pi/4), frame=f2)
    p12 = p1.transform_pose(p2)
    assert_almost_equal(p12.get_ndarray(), np.array([0, 0, 1, 0, 0.92387953251128674, 0.0, 0.38268343236508978, 0.0]))
    p13 = p1.transform_pose(p3)
    assert_almost_equal(p13.get_ndarray(), np.array([0, 70.7106781, 1, -70.7106781, 0.92387953251128674, 0.0, 0.38268343236508978, 0.0]))
    p14 = p1.transform_pose(p4)
    assert_almost_equal(p14.get_ndarray(), np.array([0, 0, 1, 0, 1, 0, 0, 0]))


def test_pose_vector_transform_motion():
    f1 = Frame("f1")
    f2 = Frame("f2")
    f3 = Frame("f3")
    p1 = PoseVector(XYZVector(0, 1, 0), Versor(XYZVector(0, 1, 0), math.pi/4), frame=f1, end_frame=f3)
    p2 = PoseVector(XYZVector(100, 0, 0), frame=f2, end_frame=f3)

    m1 = MotionVector(XYZVector(10, 0, 0), frame=f3)
    m11 = p1.transform_motion(m1)
    assert_almost_equal(m11.get_ndarray(), np.array([0, 7.0710678, 0, -7.0710678, 0, 0, 0, 0]))
    m12 = p2.transform_motion(m1)
    assert_almost_equal(m12.get_ndarray(), np.array([0, 10, 0, 0, 0, 0, 0, 0]))

    m2 = MotionVector(angular_component=XYZVector(0, 1, 0), frame=f3)
    m21 = p1.transform_motion(m2)
    assert_almost_equal(m21.get_ndarray(), np.array([0, 0, 0, 0, 0, 0, 1, 0]))
    m22 = p2.transform_motion(m2)
    assert_almost_equal(m22.get_ndarray(), np.array([0, 0, 0, 100, 0, 0, 1, 0]))


def test_pose_vector_inverse():
    f1 = Frame("f1")
    f2 = Frame("f2")
    p1 = PoseVector(XYZVector(12, 13, 14), Versor(XYZVector(1, 0, 0), math.pi / 3), frame=f1, end_frame=f2)
    p1i = p1.inverse()
    p2 = PoseVector(XYZVector(0, 1, 0), Versor(XYZVector(0, 1, 0), math.pi/4), frame=f2)
    p3 = p1.transform_pose(p2)
    p4 = p1i.transform_pose(p3)
    assert_almost_equal(p4.get_ndarray(), p2.get_ndarray())

    m1 = MotionVector(XYZVector(10, 0.5, 2354), XYZVector(10, 2, -16.8), frame=f2)
    m2 = p1.transform_motion(m1)
    m3 = p1i.transform_motion(m2)
    assert_almost_equal(m1.get_ndarray(), m3.get_ndarray())


def test_inertia_moment():
    frame = Frame(name="test")
    i1 = InertiaMoment(sympy.diag(1, 1, 1), XYZVector(), 1, frame)
    force = i1.dot(MotionVector(XYZVector(0, 0, 1), frame=frame))
    force_ndarray = force.get_ndarray()
    assert_almost_equal(force_ndarray, np.array([0, 0, 0, 1, 0, 0, 0, 0]))
    frame2 = Frame(name="test2")
    p1 = PoseVector(XYZVector(1, 0, 0), frame=frame2, end_frame=frame)
    i2 = p1.transform_inertia(i1)
    force = i2.dot(MotionVector(XYZVector(0, 0, 1), frame=frame2))
    force_ndarray = force.get_ndarray()
    assert_almost_equal(force_ndarray, np.array([0, 0, 0, 1, 0, 0, -1, 0]))


def test_inertia_moment_transforms():
    np.set_printoptions(precision=3, suppress=True)

    root_frame = Frame(name="root")
    local_frame = Frame(name="local")
    p1 = PoseVector(XYZVector(10, 0, 0), Versor(XYZVector(0, 1, 0), math.pi/4), frame=root_frame, end_frame=local_frame)
    i1 = InertiaMoment(sympy.diag(5, 5, 25), XYZVector(), 1, local_frame)
    local_accel = MotionVector(XYZVector(0, 0, 0), XYZVector(0, 0, 1), frame=local_frame)
    print("local accel: {}".format(local_accel.get_ndarray()))
    local_force_1 = i1.dot(local_accel)
    print("local force 1: {}".format(local_force_1.get_ndarray()))
    world_force_1 = p1.transform_force(local_force_1)
    print("world force 1: {}".format(world_force_1.get_ndarray()))
    i2 = p1.transform_inertia(i1)
    world_accel = p1.transform_motion(local_accel)
    print("world accel: {}".format(world_accel.get_ndarray()))
    world_force_2 = i2.dot(world_accel)
    print("world force 2: {}".format(world_force_2.get_ndarray()))
    assert_almost_equal(world_force_1.get_ndarray(), world_force_2.get_ndarray())
    local_force_2 = p1.inverse().transform_force(world_force_2)
    print("local force 2: {}".format(local_force_2.get_ndarray()))
    assert_almost_equal(local_force_1.get_ndarray(), local_force_2.get_ndarray())

