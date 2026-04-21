"""Lie algebra utilities for SE(3)/SO(3) operations.

Provides quaternion-to-rotation, screw exponentials, FK, Jacobian, and SE(3) inverse.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion
# ---------------------------------------------------------------------------

def quat2SO3(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to rotation matrix.

    Args:
        quaternion: shape (K, 4), order (x, y, z, w).

    Returns:
        Rotation matrices, shape (K, 3, 3).
    """
    assert quaternion.shape[1] == 4

    K = quaternion.shape[0]
    R = np.zeros((K, 3, 3), dtype=quaternion.dtype)

    x, y, z, w = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    xx, yy, zz, ww = x**2, y**2, z**2, w**2
    n = ww + xx + yy + zz
    s = np.zeros(K, dtype=quaternion.dtype)
    nonzero = n != 0
    s[nonzero] = 2 / n[nonzero]

    xy = s * x * y
    xz = s * x * z
    yz = s * y * z
    xw = s * x * w
    yw = s * y * w
    zw = s * z * w

    xx = s * xx
    yy = s * yy
    zz = s * zz

    idxs = np.arange(K)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw
    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw
    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R


# ---------------------------------------------------------------------------
# Skew-symmetric
# ---------------------------------------------------------------------------

def skew(w: np.ndarray) -> np.ndarray:
    """Compute skew-symmetric matrix from 3-vector or 6-vector (se(3))."""
    if len(w) == 3:
        return np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0],
        ])
    if len(w) == 6:
        W = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0],
        ])
        return np.vstack([
            np.concatenate([W, w[3:].reshape(3, 1)], axis=1),
            np.array([0, 0, 0, 0]),
        ])
    raise ValueError(f"Expected length 3 or 6, got {len(w)}")


def invskew(W: np.ndarray) -> np.ndarray:
    """Extract vector from skew-symmetric matrix."""
    if len(W) == 3:
        return np.array([-W[1, 2], W[0, 2], -W[0, 1]])
    if len(W) == 4:
        w = np.array([-W[1, 2], W[0, 2], -W[0, 1]])
        return np.hstack([w, W[:3, 3]])
    raise ValueError(f"Expected 3x3 or 4x4, got {W.shape}")


# ---------------------------------------------------------------------------
# Exponential maps
# ---------------------------------------------------------------------------

def exp_so3(w: np.ndarray) -> np.ndarray:
    """SO(3) exponential map: rotation vector -> rotation matrix."""
    if w.shape == (3, 3):
        w = invskew(w)
    if len(w) != 3:
        raise ValueError(f"Expected 3-vector, got length {len(w)}")

    wnorm = np.linalg.norm(w)
    if wnorm < 1e-14:
        return np.eye(3)

    W = skew(w)
    cw, sw = np.cos(wnorm), np.sin(wnorm)
    return np.eye(3) + (sw / wnorm) * W + ((1 - cw) / wnorm**2) * W @ W


def exp_se3(S: np.ndarray) -> np.ndarray:
    """SE(3) exponential map: twist vector -> transformation matrix."""
    if S.shape == (4, 4):
        S = invskew(S)
    if len(S) != 6:
        raise ValueError(f"Expected 6-vector, got length {len(S)}")

    w, v = S[:3], S[3:]
    wnorm = np.linalg.norm(w)

    T = np.eye(4)
    if wnorm < 1e-14:
        T[:3, 3] = v
    else:
        W = skew(w)
        cw, sw = np.cos(wnorm), np.sin(wnorm)
        P = np.eye(3) + ((1 - cw) / wnorm**2) * W + ((wnorm - sw) / wnorm**3) * W @ W
        T[:3, :3] = exp_so3(w)
        T[:3, 3] = (P @ v).ravel()
    return T


def Screw_to_SE3(S: np.ndarray, theta: float) -> np.ndarray:
    """Convert screw axis + angle to SE(3) transformation."""
    if len(S) != 6:
        raise ValueError(f"Expected 6-vector, got length {len(S)}")
    return exp_se3(S * theta)


# ---------------------------------------------------------------------------
# Adjoint
# ---------------------------------------------------------------------------

def Adjoint_SE3(T: np.ndarray) -> np.ndarray:
    """6x6 adjoint representation of SE(3) matrix."""
    R = T[:3, :3]
    p = T[:3, 3]
    AdT = np.zeros((6, 6))
    AdT[:3, :3] = R
    AdT[3:, :3] = skew(p) @ R
    AdT[3:, 3:] = R
    return AdT


# ---------------------------------------------------------------------------
# Forward kinematics / Jacobian
# ---------------------------------------------------------------------------

def forward_kinematics(
    joint_pos: np.ndarray,
    S_screw: list,
    initial_ee_frame: np.ndarray,
) -> tuple:
    """Product-of-exponentials forward kinematics.

    Returns:
        (link_frames, ee_frame) where link_frames has shape (n_joints, 4, 4).
    """
    link_frames = []
    T = np.eye(4)
    for q, S in zip(joint_pos, S_screw):
        T = T @ exp_se3(S * q)
        link_frames.append(T.copy())
    link_frames = np.array(link_frames)
    ee_frame = link_frames[-1] @ initial_ee_frame
    return link_frames, ee_frame


def get_SpaceJacobian(S_screw: list, link_frames: np.ndarray) -> np.ndarray:
    """Compute space Jacobian from screw axes and link frames."""
    cols = [S_screw[0].reshape(6, 1)]
    for T, S in zip(link_frames[:-1], S_screw[1:]):
        cols.append((Adjoint_SE3(T) @ S).reshape(6, 1))
    return np.hstack(cols)


# ---------------------------------------------------------------------------
# SE(3) inverse
# ---------------------------------------------------------------------------

def invSE3(SE3: np.ndarray) -> np.ndarray:
    """Efficient SE(3) inverse (supports single or batch).

    Args:
        SE3: shape (4, 4) or (N, 4, 4).

    Returns:
        Inverse SE(3) matrix, same shape as input.
    """
    is_single = SE3.ndim == 2
    if is_single:
        SE3 = SE3[np.newaxis]

    R = SE3[:, :3, :3]
    p = SE3[:, :3, 3:]
    inv_R = R.transpose(0, 2, 1)

    result = np.zeros_like(SE3)
    result[:, :3, :3] = inv_R
    result[:, :3, 3] = -np.squeeze(inv_R @ p, -1)
    result[:, 3, 3] = 1

    if is_single:
        result = result[0]
    return result
