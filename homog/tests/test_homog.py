from homog import *
import numpy as np
from numpy.testing import assert_allclose
import pytest

try:
    import numba
    only_if_numba = lambda f: f
except ImportError:
    import pytest
    only_if_numba = pytest.mark.skip


def test_sym():
    assert sym.tetrahedral_frames.shape == (12, 4, 4)
    assert sym.octahedral_frames.shape == (24, 4, 4)
    assert sym.icosahedral_frames.shape == (60, 4, 4)
    x = np.concatenate([
        sym.tetrahedral_frames, sym.octahedral_frames, sym.icosahedral_frames
    ])
    assert np.all(x[..., 3, 3] == 1)
    assert np.all(x[..., 3, :3] == 0)
    assert np.all(x[..., :3, 3] == 0)


def test_homo_rotation_single():
    axis0 = hnormalized(np.random.randn(3))
    ang0 = np.pi / 4.0
    r = hrot(list(axis0), float(ang0))
    a = fast_axis_of(r)
    n = hnorm(a)
    assert np.all(abs(a / n - axis0) < 0.001)
    assert np.all(abs(np.arcsin(n / 2) - ang0) < 0.001)


def test_homo_rotation_ein_single():
    axis0 = hnormalized(np.random.randn(3))
    ang0 = np.pi / 4.0
    r = hrot(list(axis0), float(ang0), func='ein')
    a = fast_axis_of(r)
    n = hnorm(a)
    assert np.all(abs(a / n - axis0) < 0.001)
    assert np.all(abs(np.arcsin(n / 2) - ang0) < 0.001)


def test_homo_rotation_center():
    AAC = assert_allclose
    AAC([0, 2, 0, 1],
        hrot([1, 0, 0], 180, [0, 1, 0]) @ (0, 0, 0, 1),
        atol=1e-5)
    AAC([0, 1, -1, 1],
        hrot([1, 0, 0], 90, [0, 1, 0]) @ (0, 0, 0, 1),
        atol=1e-5)
    AAC([-1, 1, 2, 1],
        hrot([1, 1, 0], 180, [0, 1, 1]) @ (0, 0, 0, 1),
        atol=1e-5)


def test_homo_rotation_array():
    shape = (1, 2, 1, 3, 4, 1, 1)
    axis0 = hnormalized(np.random.randn(*(shape + (3, ))))
    ang0 = np.random.rand(*shape) * (0.99 * np.pi / 2 + 0.005 * np.pi / 2)
    r = hrot(axis0, ang0)
    a = fast_axis_of(r)
    n = hnorm(a)[..., np.newaxis]
    assert np.all(abs(a / n - axis0) < 0.001)
    assert np.all(abs(np.arcsin(n[..., 0] / 2) - ang0) < 0.001)


def test_homo_rotation_ein_array():
    shape = (1, 2, 1, 3, 4, 1, 1)
    axis0 = hnormalized(np.random.randn(*(shape + (3,))))
    ang0 = np.random.rand(*shape) * (0.99 * np.pi / 2 + 0.005 * np.pi / 2)
    r = hrot(axis0, ang0, func='ein')
    a = fast_axis_of(r)
    n = hnorm(a)[..., np.newaxis]
    assert np.all(abs(a / n - axis0) < 0.001)
    assert np.all(abs(np.arcsin(n[..., 0] / 2) - ang0) < 0.001)


def test_homo_rotation_angle():
    ang = np.random.rand(1000) * np.pi
    a = rand_unit()
    u = proj_perp(a, rand_vec())
    x = hrot(a, ang)
    ang2 = angle(u, x @ u)
    assert np.allclose(ang, ang2, atol=1e-5)


def test_homo_rotation_ein_angle():
    ang = np.random.rand(1000) * np.pi
    a = random_unit()
    u = proj_perp(a, random_vec())
    x = hrot(a, ang, func='ein')
    ang2 = angle(u, x @ u)
    assert np.allclose(ang, ang2, atol=1e-5)


def test_htrans():
    assert htrans([1, 3, 7]).shape == (4, 4)
    assert_allclose(htrans([1, 3, 7])[:3, 3], (1, 3, 7))

    with pytest.raises(ValueError):
        htrans([4, 3, 2, 1])

    s = (2, )
    t = np.random.randn(*s, 3)
    ht = htrans(t)
    assert ht.shape == s + (4, 4)
    assert_allclose(ht[..., :3, 3], t)


def test_hcross():
    assert np.allclose(hcross([1, 0, 0], [0, 1, 0]), [0, 0, 1])
    assert np.allclose(hcross([1, 0, 0, 0], [0, 1, 0, 0]), [0, 0, 1, 0])
    a, b = np.random.randn(3, 4, 5, 3), np.random.randn(3, 4, 5, 3)
    c = hcross(a, b)
    assert np.allclose(hdot(a, c), 0)
    assert np.allclose(hdot(b, c), 0)


def test_axis_angle_of():
    ax, an = axis_angle_of(hrot([10, 10, 0], np.pi))
    assert 1e-5 > abs(ax[0] - ax[1])
    assert 1e-5 > abs(ax[2])
    ax, an = axis_angle_of(hrot([0, 1, 0], np.pi))
    assert 1e-5 > abs(ax[0])
    assert 1e-5 > abs(ax[1]) - 1
    assert 1e-5 > abs(ax[2])

    ax, an = axis_angle_of(hrot([0, 1, 0], np.pi * 0.25))
    print(ax, an)
    assert_allclose(ax, [0, 1, 0, 0], atol=1e-5)
    assert 1e-5 > abs(an - np.pi * 0.25)
    ax, an = axis_angle_of(hrot([0, 1, 0], np.pi * 0.75))
    print(ax, an)
    assert_allclose(ax, [0, 1, 0, 0], atol=1e-5)
    assert 1e-5 > abs(an - np.pi * 0.75)

    ax, an = axis_angle_of(hrot([1, 0, 0], np.pi / 2))
    print(np.pi / an)
    assert 1e-5 > abs(an - np.pi / 2)


def test_axis_angle_of_rand():
    shape = (
        4,
        5,
        6,
        7,
        8,
    )
    axis = hnormalized(np.random.randn(*shape, 3))
    angl = np.random.random(shape) * np.pi / 2
    rot = hrot(axis, angl, dtype='f8')
    ax, an = axis_angle_of(rot)
    assert_allclose(axis, ax, rtol=1e-5)
    assert_allclose(angl, an, rtol=1e-5)


def test_is_valid_rays():
    assert not is_valid_rays([[0, 1], [0, 0], [0, 0], [0, 0]])
    assert not is_valid_rays([[0, 0], [0, 0], [0, 0], [1, 0]])
    assert not is_valid_rays([[0, 0], [0, 3], [0, 0], [1, 0]])
    assert is_valid_rays([[0, 0], [0, 1], [0, 0], [1, 0]])


def test_rand_ray():
    r = rand_ray()
    assert np.all(r[..., 3, :] == (1, 0))
    assert r.shape == (4, 2)
    assert_allclose(hnorm(r[..., :3, 1]), 1)

    r = rand_ray(shape=(5, 6, 7))
    assert np.all(r[..., 3, :] == (1, 0))
    assert r.shape == (5, 6, 7, 4, 2)
    assert_allclose(hnorm(r[..., :3, 1]), 1)


def test_proj_prep():
    assert_allclose([2, 3, 0], proj_perp([0, 0, 1], [2, 3, 99]))
    assert_allclose([2, 3, 0], proj_perp([0, 0, 2], [2, 3, 99]))
    a, b = np.random.randn(2, 5, 6, 7, 3)
    pp = proj_perp(a, b)
    assert_allclose(hdot(a, pp), 0, atol=1e-5)


def test_point_in_plane():
    plane = rand_ray((5, 6, 7))
    assert np.all(point_in_plane(plane, plane[..., :3, 0]))
    pt = proj_perp(plane[..., :3, 1], np.random.randn(3))
    assert np.all(point_in_plane(plane, plane[..., :3, 0] + pt))


def test_ray_in_plane():
    plane = rand_ray((5, 6, 7))
    dirn = proj_perp(plane[..., :3, 1], np.random.randn(5, 6, 7, 3))
    ray = hray(plane[..., :3, 0] + np.cross(plane[..., :3, 1], dirn) * 7, dirn)
    assert np.all(ray_in_plane(plane, ray))


def test_intersect_planes():
    with pytest.raises(ValueError):
        intersect_planes(
            np.array([[0, 0, 0, 2], [0, 0, 0, 0]]).T,
            np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T)
    with pytest.raises(ValueError):
        intersect_planes(
            np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T,
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]).T)
    with pytest.raises(ValueError):
        intersect_planes(
            np.array([[0, 0, 1], [0, 0, 0, 0]]).T,
            np.array([[0, 0, 1], [0, 0, 0, 1]]).T)
    with pytest.raises(ValueError):
        intersect_planes(
            np.array(9 * [[[0, 0], [0, 0], [0, 0], [1, 0]]]),
            np.array(2 * [[[0, 0], [0, 0], [0, 0], [1, 0]]]))

    # isct, sts = intersect_planes(np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]),
    # np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]))
    # assert isct.shape[:-2] == sts.shape == (9,)
    # assert np.all(sts == 2)

    # isct, sts = intersect_planes(np.array([[1, 0, 0, 1], [1, 0, 0, 0]]),
    # np.array([[0, 0, 0, 1], [1, 0, 0, 0]]))
    # assert sts == 1

    isct, sts = intersect_planes(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T)
    assert sts == 0
    assert isct[2, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (0, 0, 1))

    isct, sts = intersect_planes(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T)
    assert sts == 0
    assert isct[1, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (0, 1, 0))

    isct, sts = intersect_planes(
        np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T)
    assert sts == 0
    assert isct[0, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (1, 0, 0))

    isct, sts = intersect_planes(
        np.array([[7, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 9, 0, 1], [0, 1, 0, 0]]).T)
    assert sts == 0
    assert_allclose(isct[:3, 0], [7, 9, 0])
    assert_allclose(abs(isct[:3, 1]), [0, 0, 1])

    isct, sts = intersect_planes(
        np.array([[0, 0, 0, 1], hnormalized([1, 1, 0, 0])]).T,
        np.array([[0, 0, 0, 1], hnormalized([0, 1, 1, 0])]).T)
    assert sts == 0
    assert_allclose(abs(isct[:, 1]), hnormalized([1, 1, 1]))

    p1 = hray([2, 0, 0, 1], [1, 0, 0, 0])
    p2 = hray([0, 0, 0, 1], [0, 0, 1, 0])
    isct, sts = intersect_planes(p1, p2)
    assert sts == 0
    assert np.all(ray_in_plane(p1, isct))
    assert np.all(ray_in_plane(p2, isct))

    p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.],
                   [-0.80966465, -0.18557869, 0.55677976, 0.]]).T
    p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.],
                   [-0.92436319, -0.0221499, 0.38087016, 0.]]).T
    isct, sts = intersect_planes(p1, p2)
    assert sts == 0
    assert np.all(ray_in_plane(p1, isct))
    assert np.all(ray_in_plane(p2, isct))


def test_intersect_planes_rand():
    # origin case
    plane1, plane2 = rand_ray(shape=(2, 1))
    plane1[..., :3, 0] = 0
    plane2[..., :3, 0] = 0
    isect, status = intersect_planes(plane1, plane2)
    assert np.all(status == 0)
    assert np.all(ray_in_plane(plane1, isect))
    assert np.all(ray_in_plane(plane2, isect))

    # orthogonal case
    plane1, plane2 = rand_ray(shape=(2, 1))
    plane1[..., :, 1] = hnormalized([0, 0, 1])
    plane2[..., :, 1] = hnormalized([0, 1, 0])
    isect, status = intersect_planes(plane1, plane2)
    assert np.all(status == 0)
    assert np.all(ray_in_plane(plane1, isect))
    assert np.all(ray_in_plane(plane2, isect))

    # general case
    plane1, plane2 = rand_ray(shape=(2, 5, 6, 7, 8, 9))
    isect, status = intersect_planes(plane1, plane2)
    assert np.all(status == 0)
    assert np.all(ray_in_plane(plane1, isect))
    assert np.all(ray_in_plane(plane2, isect))


def test_axis_ang_cen_of_rand():
    shape = (5, 6, 7, 8, 9)
    axis0 = hnormalized(np.random.randn(*shape, 3))
    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
    cen0 = np.random.randn(*shape, 3) * 100.0

    helical_trans = np.random.randn(*shape)[..., None] * axis0
    rot = hrot(axis0, ang0, cen0, dtype='f8')
    rot[..., :, 3] += helical_trans
    axis, ang, cen = axis_ang_cen_of(rot)

    assert_allclose(axis0, axis, rtol=1e-5)
    assert_allclose(ang0, ang, rtol=1e-5)
    #  check rotation doesn't move cen
    cenhat = (rot @ cen[..., None]).squeeze()
    assert_allclose(cen + helical_trans, cenhat, rtol=1e-5, atol=1e-5)


def test_hinv_rand():
    shape = (
        5,
        6,
        7,
        8,
        9,
    )
    axis0 = hnormalized(np.random.randn(*shape, 3))
    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
    cen0 = np.random.randn(*shape, 3) * 100.0
    helical_trans = np.random.randn(*shape)[..., None] * axis0
    rot = hrot(axis0, ang0, cen0, dtype='f8')
    rot[..., :, 3] += helical_trans
    assert np.allclose(np.eye(4), hinv(rot) @ rot)


def test_hstub():
    sh = (5, 6, 7, 8, 9)
    u = h_rand_points(sh)
    v = h_rand_points(sh)
    w = h_rand_points(sh)
    s = hstub(u, v, w)
    assert is_homog_xform(s)

    assert is_homog_xform(hstub([1, 2, 3], [5, 6, 4], [9, 7, 8]))


def test_line_line_dist():
    lld = line_line_distance
    assert lld(hray([0, 0, 0], [1, 0, 0]), hray([0, 0, 0], [1, 0, 0])) == 0
    assert lld(hray([0, 0, 0], [1, 0, 0]), hray([1, 0, 0], [1, 0, 0])) == 0
    assert lld(hray([0, 0, 0], [1, 0, 0]), hray([0, 1, 0], [1, 0, 0])) == 1
    assert lld(hray([0, 0, 0], [1, 0, 0]), hray([0, 1, 0], [0, 0, 1])) == 1


def test_line_line_closest_points():
    lld = line_line_distance
    llcp = line_line_closest_points
    p, q = llcp(hray([0, 0, 0], [1, 0, 0]), hray([0, 0, 0], [0, 1, 0]))
    assert np.all(p == [0, 0, 0, 1]) and np.all(q == [0, 0, 0, 1])
    p, q = llcp(hray([0, 1, 0], [1, 0, 0]), hray([1, 0, 0], [0, 1, 0]))
    assert np.all(p == [1, 1, 0, 1]) and np.all(q == [1, 1, 0, 1])
    p, q = llcp(hray([1, 1, 0], [1, 0, 0]), hray([1, 1, 0], [0, 1, 0]))
    assert np.all(p == [1, 1, 0, 1]) and np.all(q == [1, 1, 0, 1])
    p, q = llcp(hray([1, 2, 3], [1, 0, 0]), hray([4, 5, 6], [0, 1, 0]))
    assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])
    p, q = llcp(hray([1, 2, 3], [-13, 0, 0]), hray([4, 5, 6], [0, -7, 0]))
    assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])
    p, q = llcp(hray([1, 2, 3], [1, 0, 0]), hray([4, 5, 6], [0, 1, 0]))
    assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])

    r1, r2 = hray([1, 2, 3], [1, 0, 0]), hray([4, 5, 6], [0, 1, 0])
    x = rand_xform((5, 6, 7))
    xinv = np.linalg.inv(x)
    p, q = llcp(x @ r1, x @ r2)
    assert np.allclose((xinv @ p[..., None]).squeeze(-1), [4, 2, 3, 1])
    assert np.allclose((xinv @ q[..., None]).squeeze(-1), [4, 2, 6, 1])

    shape = (
        5,
        6,
        7,
    )
    r1 = rand_ray(cen=np.random.randn(*shape, 3))
    r2 = rand_ray(cen=np.random.randn(*shape, 3))
    p, q = llcp(r1, r2)
    assert p.shape[:-1] == shape and q.shape[:-1] == shape
    lldist0 = hnorm(p - q)
    lldist1 = lld(r1, r2)
    assert np.allclose(lldist0, lldist1, atol=1e-3, rtol=1e-3)


@only_if_numba
def test_numba_line_line_distance():
    for i in range(100):
        ray1 = rand_ray(cen=np.random.randn(3))
        ray2 = rand_ray(cen=np.random.randn(3))
        pt1, pt2 = ray1[:, 0], ray2[:, 0]
        ax1, ax2 = ray1[:, 1], ray2[:, 1]
        d = line_line_distance_pa(pt1, ax1, pt2, ax2)
        d2 = numba_line_line_distance_pa(pt1, ax1, pt2, ax2)
        assert np.allclose(d, d2)


def test_dihedral():
    assert 0.00001 > abs(np.pi / 2 -
                         dihedral([1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]))
    assert 0.00001 > abs(-np.pi / 2 -
                         dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]))
    a, b, c = hpoint([1, 0, 0]), hpoint([0, 1, 0]), hpoint([0, 0, 1]),
    n = hpoint([0, 0, 0])
    x = rand_xform(10)
    assert np.allclose(
        dihedral(a, b, c, n), dihedral(x @ a, x @ b, x @ c, x @ n))
    for ang in np.arange(-np.pi + 0.001, np.pi, 0.1):
        x = hrot([0, 1, 0], ang)
        d = dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], x @ [1, 0, 0, 0])
        assert abs(ang - d) < 0.000001


def test_angle():
    assert 0.0001 > abs(angle([1, 0, 0], [0, 1, 0]) - np.pi / 2)
    assert 0.0001 > abs(angle([1, 1, 0], [0, 1, 0]) - np.pi / 4)


def test_align_around_axis():
    axis = rand_unit(1000)
    u = rand_vec()
    ang = np.random.rand(1000) * np.pi
    x = hrot(axis, ang)
    v = x @ u
    uprime = align_around_axis(axis, u, v) @ u
    assert np.allclose(angle(v, uprime), 0, atol=1e-5)


def test_align_vectors_minangle():

    tgt1 = [-0.816497, -0.000000, -0.577350, 0]
    tgt2 = [0.000000, 0.000000, 1.000000, 0]
    orig1 = [0.000000, 0.000000, 1.000000, 0]
    orig2 = [-0.723746, 0.377967, -0.577350, 0]
    x = align_vectors(orig1, orig2, tgt1, tgt2)
    assert np.allclose(tgt1, x @ orig1, atol=1e-5)
    assert np.allclose(tgt2, x @ orig2, atol=1e-5)

    ax1 = np.array([0.12896027, -0.57202471, -0.81003518, 0.])
    ax2 = np.array([0., 0., -1., 0.])
    tax1 = np.array([0.57735027, 0.57735027, 0.57735027, 0.])
    tax2 = np.array([0.70710678, 0.70710678, 0., 0.])
    x = align_vectors(ax1, ax2, tax1, tax2)
    assert np.allclose(x @ ax1, tax1, atol=1e-2)
    assert np.allclose(x @ ax2, tax2, atol=1e-2)


def test_align_vectors_una_case():
    ax1 = np.array([0., 0., -1., 0.])
    ax2 = np.array([0.83822463, -0.43167392, 0.33322229, 0.])
    tax1 = np.array([-0.57735027, 0.57735027, 0.57735027, 0.])
    tax2 = np.array([0.57735027, -0.57735027, 0.57735027, 0.])
    # print(angle_degrees(ax1, ax2))
    # print(angle_degrees(tax1, tax2))
    x = align_vectors(ax1, ax2, tax1, tax2)
    # print(tax1)
    # print(x@ax1)
    # print(tax2)
    # print(x@ax2)
    assert np.allclose(x @ ax1, tax1, atol=1e-2)
    assert np.allclose(x @ ax2, tax2, atol=1e-2)


@only_if_numba
def test_numba_axis_angle_of():
    x = rand_xform((100, ))

    # tn = time.time()
    axs, ang = numba_axis_angle_of(x)
    # tn = time.time() - tn
    # tp = time.time()
    refaxs, refang = axis_angle_of(x)
    # tp = time.time() - tp
    # print(tn, tp)

    assert np.allclose(refaxs, axs)
    assert np.allclose(refang, ang)


@only_if_numba
def test_numba_point_in_plane():
    plane = rand_ray((100, ))
    for i in range(100):
        assert numba_point_in_plane(plane[i], plane[i, :3, 0])
    pt = proj_perp(plane[..., :3, 1], np.random.randn(3))
    for i in range(100):
        assert numba_point_in_plane(plane[i], plane[i, :3, 0] + pt[i])


@pytest.mark.skip
def test_intersect_planes():
    with pytest.raises(ValueError):
        numba_intersect_planes(
            np.array([[0, 0, 0, 2], [0, 0, 0, 0]]).T,
            np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T)
    with pytest.raises(ValueError):
        numba_intersect_planes(
            np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T,
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]).T)
    # with pytest.raises(ValueError):
    # numba_intersect_planes(
    # np.array([[0, 0, 1], [0, 0, 0, 0]]).T,
    # np.array([[0, 0, 1], [0, 0, 0, 1]]).T
    # )

    # isct, sts = numba_intersect_planes(np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]),
    # np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]))
    # assert isct.shape[:-2] == sts.shape == (9,)
    # assert np.all(sts == 2)

    # isct, sts = numba_intersect_planes(np.array([[1, 0, 0, 1], [1, 0, 0, 0]]),
    # np.array([[0, 0, 0, 1], [1, 0, 0, 0]]))
    # assert sts == 1

    isct = numba_intersect_planes(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T)
    assert isct[2, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (0, 0, 1))

    isct = numba_intersect_planes(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T)
    # assert sts == 0
    assert isct[1, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (0, 1, 0))

    isct = numba_intersect_planes(
        np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T)
    assert isct[0, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (1, 0, 0))

    isct = numba_intersect_planes(
        np.array([[7, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 9, 0, 1], [0, 1, 0, 0]]).T)
    assert np.allclose(isct[:3, 0], [7, 9, 0])
    assert np.allclose(abs(isct[:3, 1]), [0, 0, 1])

    isct = numba_intersect_planes(
        np.array([[0, 0, 0, 1], hnormalized([1, 1, 0, 0])]).T,
        np.array([[0, 0, 0, 1], hnormalized([0, 1, 1, 0])]).T)
    assert np.allclose(abs(isct[:, 1]), hnormalized([1, 1, 1]))

    p1 = hray([2, 0, 0, 1], [1, 0, 0, 0])
    p2 = hray([0, 0, 0, 1], [0, 0, 1, 0])
    isct = numba_intersect_planes(p1, p2)
    assert np.all(ray_in_plane(p1, isct))
    assert np.all(ray_in_plane(p2, isct))

    p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.],
                   [-0.80966465, -0.18557869, 0.55677976, 0.]]).T
    p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.],
                   [-0.92436319, -0.0221499, 0.38087016, 0.]]).T
    isct = numba_intersect_planes(p1, p2)
    assert np.all(ray_in_plane(p1, isct))
    assert np.all(ray_in_plane(p2, isct))


@only_if_numba
def test_numba_intersect_planes_rand():
    # origin case
    plane1, plane2 = rand_ray(shape=(2, 1))
    plane1[..., :3, 0] = 0
    plane2[..., :3, 0] = 0
    isect = numba_intersect_planes(plane1.squeeze(), plane2.squeeze())
    assert np.all(ray_in_plane(plane1, isect))
    assert np.all(ray_in_plane(plane2, isect))

    # orthogonal case
    plane1, plane2 = rand_ray(shape=(2, 1))
    plane1[..., :, 1] = hnormalized([0, 0, 1])
    plane2[..., :, 1] = hnormalized([0, 1, 0])
    isect = numba_intersect_planes(plane1.squeeze(), plane2.squeeze())
    assert np.all(ray_in_plane(plane1, isect))
    assert np.all(ray_in_plane(plane2, isect))

    # general case
    for i in range(len(plane1)):
        plane1 = rand_ray(shape=(1, )).squeeze()
        plane2 = rand_ray(shape=(1, )).squeeze()
        isect = numba_intersect_planes(plane1, plane2)
        assert np.all(ray_in_plane(plane1, isect))
        assert np.all(ray_in_plane(plane2, isect))
