import homog
from homog.binning import *
import pytest
from traitlets import TraitError


def test_get_half48cell_face_basic():
    assert np.min(np.linalg.norm(half48cell_faces, axis=-1)) == 1
    assert np.max(np.linalg.norm(half48cell_faces, axis=-1)) == 1
    assert half48cell_face([0, 0, 0, 1]) == 3
    assert half48cell_face([1, -1, 1, 1]) == 6
    assert half48cell_face([0.5, 0.5, 0, 0]) == 12
    assert half48cell_face([-0.5, 0, 0, 0.5]) == 20
    assert np.allclose(half48cell_face(half48cell_faces), np.arange(24))
    q = homog.quat.rand_quat((7, 6, 5))
    x = half48cell_face(q)
    assert x.shape == (7, 6, 5)


def test_XformBinner_init():
    with pytest.raises(TraitError):
        XformBinner(cart_resl=-1)
    with pytest.raises(TraitError):
        XformBinner(ori_resl=-1)
    with pytest.raises(TraitError):
        XformBinner(cart_bound=-1)
    assert XformBinner().cart_resl == 1
    assert XformBinner(cart_resl=2).cart_resl == 2

    assert XformBinner().ori_nside == 4
    assert XformBinner(ori_resl=80).ori_nside == 1
    assert XformBinner(ori_resl=40).ori_nside == 2
    assert XformBinner(ori_resl=22).ori_nside == 3
    assert XformBinner(ori_resl=15).ori_nside == 4
    assert XformBinner(ori_resl=12).ori_nside == 5
    assert XformBinner(ori_resl=9).ori_nside == 6
    assert XformBinner(ori_resl=8).ori_nside == 7
    assert XformBinner(ori_resl=5).ori_nside == 11
    assert XformBinner(ori_resl=3).ori_nside == 18
    assert XformBinner(ori_resl=2).ori_nside == 27
    assert XformBinner(ori_resl=1).ori_nside == 53
