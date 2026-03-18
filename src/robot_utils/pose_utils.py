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

def qinterp(
    q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64], t: float
) -> npt.NDArray[np.float64]:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: First quaternion (wxyz format)
        q2: Second quaternion (wxyz format)
        t: Interpolation parameter (0 to 1); 0: q1, 1: q2

    Returns:
        Interpolated quaternion (wxyz format)
    """
    # Compute the cosine of the angle between the quaternions
    dot = np.dot(q1, q2)

    # If the dot product is negative, we need to negate one of the quaternions
    # to ensure we take the shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If the quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Calculate the angle between quaternions
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Compute interpolation coefficients
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = (s0 * q1) + (s1 * q2)

    if q[0] < 0:
        q = -q

    return q

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
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
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




def rotvec_to_xyz_wxyz(xyz_rotvec: list[float] | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    xyz_rotvec = np.asarray(xyz_rotvec)

    xyz = xyz_rotvec[:3]
    rotvec = xyz_rotvec[3:]

    # Convert rotation vector to quaternion
    rotation = Rotation.from_rotvec(rotvec)
    quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]

    # Reorder to [w, x, y, z]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # Concatenate position and quaternion
    xyz_wxyz = np.concatenate([xyz, quat_wxyz])

    return xyz_wxyz

def rotm2rotvec(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrix to rotation vector
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(theta, 0):
        return np.zeros(3)
    else:
        k = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]
        )
        k = k / (2 * np.sin(theta))
        return theta * k


def rotvec2rotm(rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation vector to rotation matrix
    """
    theta = np.linalg.norm(rotvec)
    if np.isclose(theta, 0):
        return np.eye(3)
    else:
        k = rotvec / theta
        K = np.array(
            [
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        return R


def rpy2rotm(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert roll-pitch-yaw angles to rotation matrix
    """
    roll, pitch, yaw = rpy
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def rotm2rpy(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrix to roll-pitch-yaw angles
    """
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])


def convert_batch_to_10d(eef_xyz_wxyz: npt.NDArray[np.float32], gripper_width: npt.NDArray[np.float32]):
    """
    eef_xyz_wxyz: (batch_size, obs_history_len, 7)
    gripper_width: (batch_size, obs_history_len, 1)

    return:
        robot0_10d: (batch_size, obs_history_len, 10)
    """

    assert eef_xyz_wxyz.shape[0:2] == gripper_width.shape[0:2]

    batch_size, obs_history_len = eef_xyz_wxyz.shape[0:2]

    pose10d = np.zeros((batch_size, obs_history_len, 10))
    pose10d[:, :, :3] = eef_xyz_wxyz[:, :, :3]
    pose10d[:, :, 3:9] = quat_wxyz_to_rot_6d_batch(eef_xyz_wxyz[:, :, 3:])
    pose10d[:, :, 9] = gripper_width[:, :, 0]

    return pose10d

def convert_10d_to_batch(pose10d: npt.NDArray[np.float32]):
    """
    pose10d: (batch_size, obs_history_len, 10)

    return:
        eef_xyz_wxyz: (batch_size, obs_history_len, 7)
        gripper_width: (batch_size, obs_history_len, 1)
    """
    batch_size, obs_history_len = pose10d.shape[0:2]
    eef_xyz_wxyz = np.zeros((batch_size, obs_history_len, 7), dtype=np.float32)
    eef_xyz_wxyz[:, :, :3] = pose10d[:, :, :3]
    eef_xyz_wxyz[:, :, 3:] = rot_6d_to_quat_wxyz_batch(pose10d[:, :, 3:9])
    gripper_width = pose10d[:, :, 9:]

    return eef_xyz_wxyz, gripper_width