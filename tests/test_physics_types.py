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


def test_pose_vector():
    p = PoseVector(XYZVector(0, 1, 0), Versor(XYZVector(0, 1, 0), math.pi/4))
