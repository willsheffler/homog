import numpy as np


def rand_quat(shape=()):
    q = np.random.randn(*shape, 4)
    return q / np.linalg.norm(q, axis=-1)[..., np.newaxis]


def rot_to_quat(xform):
    t = xform[..., 0, 0] + xform[..., 1, 1] + xform[..., 2, 2]
    r = np.sqrt(1 + t)
    s = 0.5 / r
    quat = np.empty(xform.shape[:-2] + (4,))
    quat[..., 0] = r / 2
    quat[..., 1] = (xform[..., 2, 1] - xform[..., 1, 2]) * s
    quat[..., 2] = (xform[..., 0, 2] - xform[..., 2, 0]) * s
    quat[..., 3] = (xform[..., 1, 0] - xform[..., 0, 1]) * s
    return quat


def quat_to_rot(quat, dtype='f8', shape=(3, 3)):
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
