import homog
from homog.quat import *
import pytest

try:
    import numba
    only_if_numba = lambda f: f
except ImportError:
    import pytest
    only_if_numba = pytest.mark.skip


def test_rand_quat():
    rq = rand_quat((1, 2, 3, 5))
    assert rq.shape == (1, 2, 3, 5, 4)
    assert np.allclose(np.linalg.norm(rq, axis=-1), 1)


def test_quat_mult():
    # from pyquaternion
    assert list(quat_multiply([1, 0, 0, 0], [1, 0, 0, 0])) == [1, 0, 0, 0]
    assert list(quat_multiply([1, 0, 0, 0], [0, 1, 0, 0])) == [0, 1, 0, 0]
    assert list(quat_multiply([1, 0, 0, 0], [0, 0, 1, 0])) == [0, 0, 1, 0]
    assert list(quat_multiply([1, 0, 0, 0], [0, 0, 0, 1])) == [0, 0, 0, 1]
    assert list(quat_multiply([0, 1, 0, 0], [1, 0, 0, 0])) == [0, 1, 0, 0]
    assert list(quat_multiply([0, 1, 0, 0], [0, 1, 0, 0])) == [-1, 0, 0, 0]
    assert list(quat_multiply([0, 1, 0, 0], [0, 0, 1, 0])) == [0, 0, 0, 1]
    assert list(quat_multiply([0, 1, 0, 0], [0, 0, 0, 1])) == [0, 0, -1, 0]
    assert list(quat_multiply([0, 0, 1, 0], [1, 0, 0, 0])) == [0, 0, 1, 0]
    assert list(quat_multiply([0, 0, 1, 0], [0, 1, 0, 0])) == [0, 0, 0, -1]
    assert list(quat_multiply([0, 0, 1, 0], [0, 0, 1, 0])) == [-1, 0, 0, 0]
    assert list(quat_multiply([0, 0, 1, 0], [0, 0, 0, 1])) == [0, 1, 0, 0]
    assert list(quat_multiply([0, 0, 0, 1], [1, 0, 0, 0])) == [0, 0, 0, 1]
    assert list(quat_multiply([0, 0, 0, 1], [0, 1, 0, 0])) == [0, 0, 1, 0]
    assert list(quat_multiply([0, 0, 0, 1], [0, 0, 1, 0])) == [0, -1, 0, 0]
    assert list(quat_multiply([0, 0, 0, 1], [0, 0, 0, 1])) == [-1, 0, 0, 0]


@only_if_numba
def test_numba_quat_mult():
    Q = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
         [0.5, 0.5, 0.5, 0.5]]
    for q in Q:
        for r in Q:
            q = np.array(q, dtype='f8')
            r = np.array(r, dtype='f8')
            assert np.all(quat_multiply(q, r) == gu_quat_multiply(q, r))

    q = homog.quat.rand_quat(1000)
    r = homog.quat.rand_quat(1000)
    assert np.allclose(quat_multiply(q, r), gu_quat_multiply(q, r))


def test_rot_quat_conversion_rand():
    x = homog.rand_xform((5, 6, 7), cart_sd=0)
    assert np.all(homog.is_homog_xform(x))
    q = rot_to_quat(x)
    assert np.all(is_valid_quat_rot(q))
    y = quat_to_xform(q)
    assert np.all(homog.is_homog_xform(y))
    assert x.shape == y.shape
    assert np.allclose(x, y)
    q = homog.quat.rand_quat()
    assert np.all(is_valid_quat_rot(q))
    x = quat_to_xform(q)
    assert np.all(homog.is_homog_xform(x))
    p = rot_to_quat(x)
    assert np.all(is_valid_quat_rot(p))
    assert p.shape == q.shape
    assert np.allclose(p, q)


@only_if_numba
def test_rot_quat_conversion_rand_numba():
    x = homog.rand_xform((5, 6, 7), cart_sd=0)
    assert np.all(homog.is_homog_xform(x))
    q = gu_rot_to_quat(x)
    assert np.all(is_valid_quat_rot(q))
    y = quat_to_xform(q)
    assert np.all(homog.is_homog_xform(y))
    assert x.shape == y.shape
    assert np.allclose(x, y)
    q = homog.quat.rand_quat()
    assert np.all(is_valid_quat_rot(q))
    x = quat_to_xform(q)
    assert np.all(homog.is_homog_xform(x))
    p = gu_rot_to_quat(x)
    assert np.all(is_valid_quat_rot(p))
    assert p.shape == q.shape
    assert np.allclose(p, q)


def test_rot_quat_conversion_cases():
    R22 = np.sqrt(2) / 2
    cases = np.array([[1.00, 0.00, 0.00, 0.00], [0.00, 1.00, 0.00, 0.00], [
        0.00, 0.00, 1.00, 0.00
    ], [0.00, 0.00, 0.00,
        1.00], [+0.5, +0.5, +0.5, +0.5], [+0.5, -0.5, -0.5, -0.5], [
            +0.5, -0.5, +0.5, +0.5
        ], [+0.5, +0.5, -0.5, -0.5], [+0.5, +0.5, -0.5, +0.5], [
            +0.5, -0.5, +0.5, -0.5
        ], [+0.5, -0.5, -0.5, +0.5], [+0.5, +0.5, +0.5, -0.5], [
            +R22, +R22, 0.00, 0.00
        ], [+R22, 0.00, +R22, 0.00], [+R22, 0.00, 0.00, +R22], [
            0.00, +R22, +R22, 0.00
        ], [0.00, +R22, 0.00,
            +R22], [0.00, 0.00, +R22, +R22], [+R22, -R22, 0.00,
                                              0.00], [+R22, 0.00, -R22, 0.00],
                      [+R22, 0.00, 0.00, -R22], [0.00, +R22, -R22, 0.00],
                      [0.00, +R22, 0.00, -R22], [0.00, 0.00, +R22, -R22]])
    assert np.all(is_valid_quat_rot(cases))
    x = quat_to_xform(cases)
    assert homog.is_homog_xform(x)
    q = xform_to_quat(x)
    assert np.all(is_valid_quat_rot(q))
    assert np.allclose(cases, q)
