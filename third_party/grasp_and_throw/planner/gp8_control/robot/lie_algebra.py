"""Lie algebra utilities for rigid-body transformations.

Provides SO(3) and SE(3) exponential/logarithm maps, skew-symmetric matrix
operations, and adjoint representations used by Product-of-Exponentials (PoE)
robot kinematics.

Ported and cleaned up from legacy/src/utils/Lie_numpy.py.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Skew-symmetric helpers
# ---------------------------------------------------------------------------

def skew(w: np.ndarray) -> np.ndarray:
    """Convert a 3-vector to a 3x3 skew-symmetric matrix.

    Args:
        w: Array of shape (3,).

    Returns:
        3x3 skew-symmetric matrix [w]_x such that [w]_x @ v == cross(w, v).
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.shape[0] != 3:
        raise ValueError(f"Expected 3-vector, got shape {w.shape}")
    return np.array([
        [0.0,  -w[2],  w[1]],
        [w[2],  0.0,  -w[0]],
        [-w[1], w[0],  0.0],
    ])


def skew_se3(S: np.ndarray) -> np.ndarray:
    """Convert a 6-vector (w, v) to a 4x4 se(3) matrix.

    Args:
        S: Array of shape (6,) with [omega; v].

    Returns:
        4x4 matrix [[w]_x, v; 0, 0].
    """
    S = np.asarray(S, dtype=np.float64).ravel()
    if S.shape[0] != 6:
        raise ValueError(f"Expected 6-vector, got shape {S.shape}")
    M = np.zeros((4, 4))
    M[:3, :3] = skew(S[:3])
    M[:3, 3] = S[3:6]
    return M


def unskew(W: np.ndarray) -> np.ndarray:
    """Extract 3-vector from a 3x3 skew-symmetric matrix.

    Args:
        W: 3x3 skew-symmetric matrix.

    Returns:
        3-vector w such that skew(w) == W.
    """
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def unskew_se3(M: np.ndarray) -> np.ndarray:
    """Extract 6-vector from a 4x4 se(3) matrix.

    Args:
        M: 4x4 se(3) matrix.

    Returns:
        6-vector [omega; v].
    """
    w = unskew(M[:3, :3])
    v = M[:3, 3]
    return np.concatenate([w, v])


# ---------------------------------------------------------------------------
# SO(3) exponential and logarithm
# ---------------------------------------------------------------------------

def exp_so3(w: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) (3-vector) to SO(3) (rotation matrix).

    Uses the Rodrigues formula: R = I + sin(t)/t [w]_x + (1-cos(t))/t^2 [w]_x^2
    where t = ||w||.

    Args:
        w: Rotation vector of shape (3,). The direction is the axis and the
           magnitude is the angle in radians.

    Returns:
        3x3 rotation matrix R in SO(3).
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.shape[0] != 3:
        raise ValueError(f"Expected 3-vector, got shape {w.shape}")

    theta = np.linalg.norm(w)
    if theta < 1e-14:
        return np.eye(3)

    W = skew(w)
    s = np.sin(theta) / theta
    c = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + s * W + c * (W @ W)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithm map from SO(3) to so(3) (returns 3-vector, not matrix).

    Args:
        R: 3x3 rotation matrix.

    Returns:
        3-vector w such that exp_so3(w) == R.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = np.clip(np.trace(R), -1.0, 3.0)

    if np.abs(trace - 3.0) < 1e-10:
        # theta ~ 0, identity rotation
        return np.zeros(3)

    if np.abs(trace + 1.0) < 1e-6:
        # theta ~ pi
        # Find the column of R+I with largest norm
        diag = np.diag(R) + 1.0
        k = np.argmax(diag)
        v = R[:, k].copy()
        v[k] += 1.0
        w = np.pi / np.sqrt(2.0 * (1.0 + R[k, k])) * v
        return w

    theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    W = (R - R.T) / (2.0 * np.sin(theta)) * theta
    return unskew(W)


# ---------------------------------------------------------------------------
# SE(3) exponential and logarithm
# ---------------------------------------------------------------------------

def exp_se3(S: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) (6-vector) to SE(3) (4x4 homogeneous).

    For a screw axis S = [w; v] the exponential is:
        T = [exp_so3(w),  G(w)*v;  0, 1]
    where G(w) = I*t + (1-cos t)/t^2 [w]_x + (t-sin t)/t^3 [w]_x^2  (t=||w||).

    If ||w|| ~ 0 (pure translation): T = [I, v; 0, 1].

    Args:
        S: 6-vector [omega; v]. If omega has unit norm, the magnitude encodes
           the rotation angle. For general inputs the angle is ||omega||.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    S = np.asarray(S, dtype=np.float64).ravel()
    if S.shape[0] != 6:
        raise ValueError(f"Expected 6-vector, got shape {S.shape}")

    w = S[:3]
    v = S[3:6]
    theta = np.linalg.norm(w)

    T = np.eye(4)
    if theta < 1e-14:
        # Pure translation
        T[:3, 3] = v
    else:
        W = skew(w)
        t_inv = 1.0 / theta
        G = (np.eye(3)
             + (1.0 - np.cos(theta)) * t_inv**2 * W
             + (theta - np.sin(theta)) * t_inv**3 * (W @ W))
        T[:3, :3] = exp_so3(w)
        T[:3, 3] = G @ v
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    """Logarithm map from SE(3) (4x4) to se(3) (returns 6-vector).

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        6-vector [omega; v].
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]

    w = log_so3(R)
    theta = np.linalg.norm(w)

    if theta < 1e-10:
        return np.concatenate([np.zeros(3), p])

    W = skew(w)
    t_inv = 1.0 / theta
    # Inverse of G: G_inv = 1/t I - 1/2 [w/t] + (1/t - 1/(2*tan(t/2))) [w/t]^2
    Wn = W * t_inv  # normalized
    half_cot = 1.0 / np.tan(theta / 2.0) if np.abs(theta) > 1e-14 else 0.0
    G_inv = (t_inv * np.eye(3)
             - 0.5 * Wn
             + (t_inv - 0.5 * half_cot) * (Wn @ Wn))
    v = theta * (G_inv @ p)
    return np.concatenate([w, v])


# ---------------------------------------------------------------------------
# Screw-axis helpers (for PoE kinematics)
# ---------------------------------------------------------------------------

def screw_to_se3(S: np.ndarray, theta: float) -> np.ndarray:
    """Compute the SE(3) matrix for a unit screw axis S rotated by theta.

    This is equivalent to exp_se3(S * theta) but avoids re-normalisation.

    Args:
        S: Unit screw axis [omega; v] with ||omega|| = 1 (revolute) or
           ||omega|| = 0 and ||v|| = 1 (prismatic).
        theta: Joint displacement (angle in radians for revolute, distance for
               prismatic).

    Returns:
        4x4 homogeneous transformation matrix.
    """
    S = np.asarray(S, dtype=np.float64).ravel()
    w = S[:3]
    v = S[3:6]
    wnorm = np.linalg.norm(w)

    T = np.eye(4)
    if wnorm < 1e-14:
        # Prismatic joint
        T[:3, 3] = v * theta
    else:
        W = skew(w)
        ct = np.cos(theta)
        st = np.sin(theta)
        # Rodrigues rotation
        T[:3, :3] = np.eye(3) + st * W + (1.0 - ct) * (W @ W)
        # Translation component
        G = theta * np.eye(3) + (1.0 - ct) * W + (theta - st) * (W @ W)
        T[:3, 3] = G @ v
    return T


# ---------------------------------------------------------------------------
# Adjoint representations
# ---------------------------------------------------------------------------

def adjoint_se3(T: np.ndarray) -> np.ndarray:
    """6x6 Adjoint representation of an SE(3) matrix.

    Ad_T maps spatial twists: V_s = Ad_T @ V_b.

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        6x6 adjoint matrix.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    pR = skew(p) @ R

    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:6, :3] = pR
    Ad[3:6, 3:6] = R
    return Ad


def small_adjoint_se3(S: np.ndarray) -> np.ndarray:
    """6x6 small adjoint (ad) of a twist S = [omega; v].

    Args:
        S: 6-vector twist.

    Returns:
        6x6 matrix [ad_S].
    """
    w = S[:3]
    v = S[3:6]
    Ww = skew(w)
    Wv = skew(v)

    ad = np.zeros((6, 6))
    ad[:3, :3] = Ww
    ad[3:6, :3] = Wv
    ad[3:6, 3:6] = Ww
    return ad


# ---------------------------------------------------------------------------
# SE(3) inverse
# ---------------------------------------------------------------------------

def inv_se3(T: np.ndarray) -> np.ndarray:
    """Efficient inverse of a homogeneous transformation matrix.

    Uses the structure of SE(3) to avoid a general 4x4 matrix inversion:
        T^{-1} = [R^T, -R^T p; 0, 1].

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        4x4 inverse transformation matrix.
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv
