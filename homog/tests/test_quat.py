import homog
from homog.quat import *
import pytest


def test_rand_quat():
    rq = rand_quat((1, 2, 3, 5))
    assert rq.shape == (1, 2, 3, 5, 4)
    assert np.allclose(np.linalg.norm(rq, axis=-1), 1)


def test_rot_quat_conversion():
    x = homog.random_xform((5, 6, 7), cen=[0, 0, 0, 1])
    q = rot_to_quat(x)
    y = quat_to_xform(q)
    assert x.shape == y.shape
    assert np.allclose(x, y)
    q = homog.quat.rand_quat()
    x = quat_to_xform(q)
    p = rot_to_quat(x)
    assert p.shape == q.shape
    assert np.allclose(p, q)
