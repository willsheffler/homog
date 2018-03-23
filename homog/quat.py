import numpy as np


def is_valid_quat_rot(quat):
    assert quat.shape[-1] == 4
    return np.isclose(1, np.linalg.norm(quat, axis=-1))


def quat_to_upper_half(quat):
    ineg0 = (quat[..., 0] < 0)
    ineg1 = (quat[..., 0] == 0) * (quat[..., 1] < 0)
    ineg2 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] < 0)
    ineg3 = ((quat[..., 0] == 0) * (quat[..., 1] == 0) *
             (quat[..., 2] == 0) * (quat[..., 3] < 0))
    # print(ineg0.shape)
    # print(ineg1.shape)
    # print(ineg2.shape)
    # print(ineg3.shape)
    ineg = ineg0 + ineg1 + ineg2 + ineg3
    quat[ineg] = -quat[ineg]
    return quat


def rand_quat(shape=()):
    q = np.random.randn(*shape, 4)
    q /= np.linalg.norm(q, axis=-1)[..., np.newaxis]
    return quat_to_upper_half(q)


def rot_to_quat(xform):
    x = np.asarray(xform)
    t0, t1, t2 = x[..., 0, 0], x[..., 1, 1], x[..., 2, 2]
    tr = t0 + t1 + t2
    quat = np.empty(x.shape[:-2] + (4,))

    case0 = tr > 0
    S0 = np.sqrt(tr[case0] + 1) * 2
    quat[case0, 0] = 0.25 * S0
    quat[case0, 1] = (x[case0, 2, 1] - x[case0, 1, 2]) / S0
    quat[case0, 2] = (x[case0, 0, 2] - x[case0, 2, 0]) / S0
    quat[case0, 3] = (x[case0, 1, 0] - x[case0, 0, 1]) / S0

    case1 = ~case0 * (t0 >= t1) * (t0 >= t2)
    S1 = np.sqrt(1.0 + x[case1, 0, 0] - x[case1, 1, 1] - x[case1, 2, 2]) * 2
    quat[case1, 0] = (x[case1, 2, 1] - x[case1, 1, 2]) / S1
    quat[case1, 1] = 0.25 * S1
    quat[case1, 2] = (x[case1, 0, 1] + x[case1, 1, 0]) / S1
    quat[case1, 3] = (x[case1, 0, 2] + x[case1, 2, 0]) / S1

    case2 = ~case0 * (t1 > t0) * (t1 >= t2)
    S2 = np.sqrt(1.0 + x[case2, 1, 1] - x[case2, 0, 0] - x[case2, 2, 2]) * 2
    quat[case2, 0] = (x[case2, 0, 2] - x[case2, 2, 0]) / S2
    quat[case2, 1] = (x[case2, 0, 1] + x[case2, 1, 0]) / S2
    quat[case2, 2] = 0.25 * S2
    quat[case2, 3] = (x[case2, 1, 2] + x[case2, 2, 1]) / S2

    case3 = ~case0 * (t2 > t0) * (t2 > t1)
    S3 = np.sqrt(1.0 + x[case3, 2, 2] - x[case3, 0, 0] - x[case3, 1, 1]) * 2
    quat[case3, 0] = (x[case3, 1, 0] - x[case3, 0, 1]) / S3
    quat[case3, 1] = (x[case3, 0, 2] + x[case3, 2, 0]) / S3
    quat[case3, 2] = (x[case3, 1, 2] + x[case3, 2, 1]) / S3
    quat[case3, 3] = 0.25 * S3

    assert (np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3)
            == np.prod(xform.shape[:-2]))

    return quat_to_upper_half(quat)

xform_to_quat = rot_to_quat


def quat_to_rot(quat, dtype='f8', shape=(3, 3)):
    quat = np.asarray(quat)
    assert quat.shape[-1] == 4
    qr = quat[..., 0]
    qi = quat[..., 1]
    qj = quat[..., 2]
    qk = quat[..., 3]
    outshape = quat.shape[:-1]
    rot = np.zeros(outshape + shape, dtype=dtype)
    rot[..., 0, 0] = 1 - 2 * (qj**2 + qk**2)
    rot[..., 0, 1] = 2 * (qi * qj - qk * qr)
    rot[..., 0, 2] = 2 * (qi * qk + qj * qr)
    rot[..., 1, 0] = 2 * (qi * qj + qk * qr)
    rot[..., 1, 1] = 1 - 2 * (qi**2 + qk**2)
    rot[..., 1, 2] = 2 * (qj * qk - qi * qr)
    rot[..., 2, 0] = 2 * (qi * qk - qj * qr)
    rot[..., 2, 1] = 2 * (qj * qk + qi * qr)
    rot[..., 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return rot


def quat_to_xform(quat, dtype='f8'):
    r = quat_to_rot(quat, dtype, shape=(4, 4))
    r[..., 3, 3] = 1
    return r
