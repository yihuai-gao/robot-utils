from typing import Any
import numpy as np
import numpy.typing as npt
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation


def qmult(q1: npt.NDArray[Any], q2: npt.NDArray[Any]) -> npt.NDArray[Any]:
    q = np.array(
        [
            q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
            q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
            q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
            q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
        ]
    )

    return q


def qconjugate(q: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def to_xyzw(wxyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if wxyz.ndim == 1:
        return np.concatenate([wxyz[1:], wxyz[0:1]])
    elif wxyz.ndim == 2:
        return np.concatenate([wxyz[:, 1:], wxyz[:, 0:1]], axis=1)
    else:
        raise ValueError("wxyz must be a 1D or 2D array")


def to_wxyz(xyzw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if xyzw.ndim == 1:
        return np.concatenate([xyzw[3:], xyzw[:3]])
    elif xyzw.ndim == 2:
        return np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)
    else:
        raise ValueError("xyzw must be a 1D or 2D array")


def interpolate_xyz_wxyz(
    pose_left: npt.NDArray[np.float64],
    pose_right: npt.NDArray[np.float64],
    timestamp_left: float,
    timestamp_right: float,
    timestamp: float,
) -> npt.NDArray[np.float64]:
    assert (
        pose_left.shape == pose_right.shape == (7,)
    ), f"pose_left.shape: {pose_left.shape}, pose_right.shape: {pose_right.shape}"
    assert timestamp_left <= timestamp < timestamp_right
    ratio = (timestamp - timestamp_left) / (timestamp_right - timestamp_left)
    pos = pose_left[:3] + ratio * (pose_right[:3] - pose_left[:3])
    rot_left = st.Rotation.from_quat(to_xyzw(pose_left[3:]))
    rot_right = st.Rotation.from_quat(to_xyzw(pose_right[3:]))
    rots = st.Rotation.concatenate([rot_left, rot_right])
    rot = st.Slerp([0, 1], rots)([ratio])
    return np.concatenate([pos, to_wxyz(rot.as_quat().squeeze())])


def get_absolute_pose(
    init_pose_xyz_wxyz: npt.NDArray[Any],
    relative_pose_xyz_wxyz: npt.NDArray[Any],
):
    """The new pose is in the same frame of reference as the initial pose"""
    new_pose_xyz_wxyz = np.zeros(7, init_pose_xyz_wxyz.dtype)
    relative_pos_in_init_frame_as_quat_wxyz = np.zeros(4, init_pose_xyz_wxyz.dtype)
    relative_pos_in_init_frame_as_quat_wxyz[1:] = relative_pose_xyz_wxyz[:3]
    init_rot_qinv = qconjugate(init_pose_xyz_wxyz[3:])
    relative_pos_in_world_frame_as_quat_wxyz = qmult(
        qmult(init_pose_xyz_wxyz[3:], relative_pos_in_init_frame_as_quat_wxyz),
        init_rot_qinv,
    )
    new_pose_xyz_wxyz[:3] = (
        init_pose_xyz_wxyz[:3] + relative_pos_in_world_frame_as_quat_wxyz[1:]
    )
    quat = qmult(init_pose_xyz_wxyz[3:], relative_pose_xyz_wxyz[3:])
    if quat[0] < 0:
        quat = -quat
    new_pose_xyz_wxyz[3:] = quat
    return new_pose_xyz_wxyz


def get_relative_pose(
    new_pose_xyz_wxyz: npt.NDArray[Any],
    init_pose_xyz_wxyz: npt.NDArray[Any],
):
    """The two poses are in the same frame of reference"""
    relative_pose_xyz_wxyz = np.zeros(7, new_pose_xyz_wxyz.dtype)
    relative_pos_in_world_frame_as_quat_wxyz = np.zeros(4, new_pose_xyz_wxyz.dtype)
    relative_pos_in_world_frame_as_quat_wxyz[1:] = (
        new_pose_xyz_wxyz[:3] - init_pose_xyz_wxyz[:3]
    )
    init_rot_qinv = qconjugate(init_pose_xyz_wxyz[3:])
    relative_pose_xyz_wxyz[:3] = qmult(
        qmult(init_rot_qinv, relative_pos_in_world_frame_as_quat_wxyz),
        init_pose_xyz_wxyz[3:],
    )[1:]
    quat = qmult(init_rot_qinv, new_pose_xyz_wxyz[3:])
    if quat[0] < 0:
        quat = -quat
    relative_pose_xyz_wxyz[3:] = quat
    return relative_pose_xyz_wxyz


def get_relative_poses(
    poses_xyz_wxyz: npt.NDArray[Any],
    init_pose_xyz_wxyz: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    return np.array(
        [
            get_relative_pose(pose_xyz_wxyz, init_pose_xyz_wxyz)
            for pose_xyz_wxyz in poses_xyz_wxyz
        ]
    )


def invert_pose(pose_xyz_wxyz: npt.NDArray[Any]) -> npt.NDArray[Any]:
    qinv = qconjugate(pose_xyz_wxyz[3:])
    pos_quat_wxyz = np.zeros(4, pose_xyz_wxyz.dtype)
    pos_quat_wxyz[1:] = pose_xyz_wxyz[:3]
    rotated_pos = qmult(
        qmult(qinv, pos_quat_wxyz),
        pose_xyz_wxyz[3:],
    )
    inverted_pose = np.zeros(7, pose_xyz_wxyz.dtype)
    inverted_pose[:3] = -rotated_pos[1:]
    if qinv[0] < 0:
        qinv = -qinv
    inverted_pose[3:] = qinv
    return inverted_pose




def normalize(vec: npt.NDArray[Any], eps: float = 1e-12) -> npt.NDArray[Any]:
    norm: npt.NDArray[Any] = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out: npt.NDArray[Any] = (vec.T / norm).T
    return out

def quat_wxyz_to_rot_6d(quat_wxyz: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Convert a quaternion to a 6D representation: the first two rows of the corresponding rotation matrix.
    https://arxiv.org/pdf/1812.07035
    quat_wxyz: (4, )
    return: (6, )
    """
    assert quat_wxyz.shape == (4,)
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]

    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )

    rot_6d = np.zeros(6)
    rot_6d[:3] = R[0, :]
    rot_6d[3:] = R[1, :]

    return rot_6d


def rot_6d_to_quat_wxyz(rot_6d: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Convert a 6D representation to a quaternion.
    https://arxiv.org/pdf/1812.07035
    rot_6d: (6, )
    return: (4, )
    """

    assert rot_6d.shape == (6,)
    a1, a2 = rot_6d[:3], rot_6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    m = np.zeros((3, 3))
    m[0, :] = b1
    m[1, :] = b2
    m[2, :] = b3

    trace = np.trace(m)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    if w < 0:
        w, x, y, z = -w, -x, -y, -z

    return np.array([w, x, y, z])


def quat_wxyz_to_rot_6d_batch(quat_wxyz: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    input (..., 4)
    output (..., 6)
    """
    assert quat_wxyz.shape[-1] == 4
    input_shape = quat_wxyz.shape[:-1]
    quat_wxyz = quat_wxyz.copy().reshape(-1, 4)
    rot_6d = np.zeros((quat_wxyz.shape[0], 6))
    for i in range(quat_wxyz.shape[0]):
        rot_6d[i] = quat_wxyz_to_rot_6d(quat_wxyz[i])
    return rot_6d.reshape(*input_shape, 6)


def rot_6d_to_quat_wxyz_batch(rot_6d: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    input (..., 6)
    output (..., 4)
    """
    assert rot_6d.shape[-1] == 6
    input_shape = rot_6d.shape[:-1]
    rot_6d = rot_6d.copy().reshape(-1, 6)
    quat_wxyz = np.zeros((rot_6d.shape[0], 4))
    for i in range(rot_6d.shape[0]):
        quat_wxyz[i] = rot_6d_to_quat_wxyz(rot_6d[i])
    return quat_wxyz.reshape(*input_shape, 4)

def rot_6d_to_mat(d6: npt.NDArray[Any]) -> npt.NDArray[Any]:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1: npt.NDArray[Any] = normalize(a1)
    b2: npt.NDArray[Any] = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2: npt.NDArray[Any] = normalize(b2)
    b3: npt.NDArray[Any] = np.cross(b1, b2, axis=-1)
    out: npt.NDArray[Any] = np.stack((b1, b2, b3), axis=-2)
    return out


def mat_to_rot_6d(mat: npt.NDArray[Any]) -> npt.NDArray[Any]:
    batch_dim = mat.shape[:-2]
    out: npt.NDArray[Any] = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def pose_9d_to_xyz_wxyz(pose_9d: npt.NDArray[Any]) -> npt.NDArray[Any]:
    assert pose_9d.shape[-1] == 9, f"pose_9d.shape: {pose_9d.shape}"
    pos_xyz = pose_9d[..., :3]
    rot_mqt = rot_6d_to_mat(pose_9d[..., 3:])
    rot_wxyz = to_wxyz(Rotation.from_matrix(rot_mqt).as_quat())
    return np.concatenate([pos_xyz, rot_wxyz], axis=-1)


def xyz_wxyz_to_pose_9d(pose_xyz_wxyz: npt.NDArray[Any]) -> npt.NDArray[Any]:
    assert pose_xyz_wxyz.shape[-1] == 7, f"pose_xyz_wxyz.shape: {pose_xyz_wxyz.shape}"
    pos_xyz = pose_xyz_wxyz[..., :3]
    rot_mqt = Rotation.from_quat(to_xyzw(pose_xyz_wxyz[..., 3:])).as_matrix()
    rot_9d = mat_to_rot_6d(rot_mqt)
    return np.concatenate([pos_xyz, rot_9d], axis=-1)


def positive_w(
    pose_xyz_wxyz: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if pose_xyz_wxyz[3] < 0.0:
        pose_xyz_wxyz[3:] = -pose_xyz_wxyz[3:]
    return pose_xyz_wxyz
