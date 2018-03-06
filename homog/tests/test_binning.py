import homog
from homog.binning import *
import pytest


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
