"""Trajectory generation primitives.

Provides trapezoidal velocity profiles and neural-network basis-function
trajectories used for pick and throw motions.
"""

from __future__ import annotations

import numpy as np


# =========================================================================
# Trapezoidal velocity profile (time-optimal, per-joint)
# =========================================================================

def opt_time_1d(xi, vi, xf, vf, M1, M2):
    """Compute time-optimal duration for 1D trapezoidal motion."""
    if vf >= vi:
        if xf - xi < 0.5 / M2 * (vf**2 - vi**2):
            xi, xf, vi, vf = -xi, -xf, -vi, -vf
    else:
        if xf - xi < 0.5 / M2 * (vi**2 - vf**2):
            xi, xf, vi, vf = -xi, -xf, -vi, -vf

    if xf - xi <= -0.5 / M2 * (vi**2 + vf**2) + M1**2 / M2:
        t1 = (-vi + np.sqrt(0.5 * (vi**2 + vf**2) + M2 * (xf - xi))) / M2
        t2 = (-vf + np.sqrt(0.5 * (vi**2 + vf**2) + M2 * (xf - xi))) / M2
    else:
        t1 = (M1 - vi) / M2
        t12 = ((xf - xi) - (-0.5 / M2 * (vi**2 + vf**2) + M1**2 / M2)) / M1
        t2 = (M1 - vf) / M2
        return t1 + t12 + t2
    return t1 + t2


def opt_time(qi, dqi, qf, dqf, M1, M2):
    """Compute time-optimal duration for multi-joint trapezoidal motion."""
    n = len(qi)
    times = np.array([
        opt_time_1d(qi[i], dqi[i], qf[i], dqf[i], M1[i], M2[i])
        for i in range(n)
    ])
    return float(np.max(times))


def _trajectory_1d(xi, vi, xf, vf, M1, M2, T, hertz=20.0, offset=0.0):
    """Generate 1D trapezoidal trajectory."""
    L = int((T - offset) * hertz)
    t = np.arange(L + 1) / hertz + offset
    traj = np.zeros(L + 1)
    vel = np.zeros(L + 1)

    if vf >= vi:
        N = (M2 * (xf - xi) + (vi**2 - vf**2) / 2) / (vi - vf + T * M2)
    else:
        N = (M2 * (xf - xi) + (vf**2 - vi**2) / 2) / (vf - vi + T * M2)

    if (N - vi) * (N - vf) < 0:
        t1 = abs(N - vi) / M2
        t2 = abs(vf - N) / M2
        idx1 = int(t1 * hertz) + 1
        idx2 = int((t1 + T - t1 - t2) * hertz) + 1

        if N > vi:
            traj[:idx1] = xi + vi * t[:idx1] + 0.5 * M2 * t[:idx1]**2
            traj[idx1:idx2] = xi + vi * t1 + 0.5 * M2 * t1**2 + N * (t[idx1:idx2] - t1)
            traj[idx2:] = xi + vi * t1 + 0.5 * M2 * t1**2 + N * (T - t1 - t2 + t1 - t1) + N * (t[idx2:] - (t1 + T - t1 - t2)) + 0.5 * M2 * (t[idx2:] - (t1 + T - t1 - t2))**2
            vel[:idx1] = vi + M2 * t[:idx1]
            vel[idx1:idx2] = N
            vel[idx2:] = N + M2 * (t[idx2:] - (t1 + T - t1 - t2))
        else:
            traj[:idx1] = xi + vi * t[:idx1] - 0.5 * M2 * t[:idx1]**2
            traj[idx1:idx2] = xi + vi * t1 - 0.5 * M2 * t1**2 + N * (t[idx1:idx2] - t1)
            traj[idx2:] = xi + vi * t1 - 0.5 * M2 * t1**2 + N * (T - t1 - t2) + N * (t[idx2:] - (t1 + T - t1 - t2)) - 0.5 * M2 * (t[idx2:] - (t1 + T - t1 - t2))**2
            vel[:idx1] = vi - M2 * t[:idx1]
            vel[idx1:idx2] = N
            vel[idx2:] = N - M2 * (t[idx2:] - (t1 + T - t1 - t2))
    else:
        if N > vi:
            b = vi + vf + T * M2
            c = (vi**2 + vf**2) / 2 + M2 * (xf - xi)
            N = (b - np.sqrt(b * b - 4 * c + 1e-8)) / 2
            if abs(N) > M1:
                return None
        else:
            b = vi + vf - T * M2
            c = (vi**2 + vf**2) / 2 - M2 * (xf - xi)
            N = (b + np.sqrt(b * b - 4 * c + 1e-8)) / 2
            if abs(N) > M1:
                return None

        t1 = abs(N - vi) / M2
        t2 = abs(vf - N) / M2
        t12 = T - (t1 + t2)
        idx1 = int(t1 * hertz) + 1
        idx2 = int((t1 + t12) * hertz) + 1

        if N > vi:
            traj[:idx1] = xi + vi * t[:idx1] + 0.5 * M2 * t[:idx1]**2
            traj[idx1:idx2] = xi + vi * t1 + 0.5 * M2 * t1**2 + N * (t[idx1:idx2] - t1)
            traj[idx2:] = xi + vi * t1 + 0.5 * M2 * t1**2 + N * t12 + N * (t[idx2:] - (t1 + t12)) - 0.5 * M2 * (t[idx2:] - (t1 + t12))**2
            vel[:idx1] = vi + M2 * t[:idx1]
            vel[idx1:idx2] = N
            vel[idx2:] = N - M2 * (t[idx2:] - (t1 + t12))
        else:
            traj[:idx1] = xi + vi * t[:idx1] - 0.5 * M2 * t[:idx1]**2
            traj[idx1:idx2] = xi + vi * t1 - 0.5 * M2 * t1**2 + N * (t[idx1:idx2] - t1)
            traj[idx2:] = xi + vi * t1 - 0.5 * M2 * t1**2 + N * t12 + N * (t[idx2:] - (t1 + t12)) + 0.5 * M2 * (t[idx2:] - (t1 + t12))**2
            vel[:idx1] = vi - M2 * t[:idx1]
            vel[idx1:idx2] = N
            vel[idx2:] = N + M2 * (t[idx2:] - (t1 + t12))

    return traj, vel


def trajectory(qi, dqi, qf, dqf, M1, M2, hertz=20.0, offset=0.0):
    """Generate multi-joint trapezoidal trajectory."""
    n = len(qi)
    T = opt_time(qi, dqi, qf, dqf, M1, M2)
    L = int((T - offset) * hertz)
    timestep = np.arange(L + 1) / hertz + offset
    traj = np.zeros((n, L + 1))
    vel = np.zeros((n, L + 1))
    for i in range(n):
        result = _trajectory_1d(qi[i], dqi[i], qf[i], dqf[i], M1[i], M2[i], T, hertz, offset)
        if result is not None:
            traj[i], vel[i] = result
    return traj, vel, timestep


def trajectory_3points(q1, dq1, q2, dq2, q3, dq3, M1, M2, hertz=20.0, offset=0.0):
    """Generate trajectory through 3 via-points (used for pick motion)."""
    traj1, vel1, ts1 = trajectory(q1, dq1, q2, dq2, M1, M2, hertz=hertz, offset=offset)
    t1 = opt_time(q1, dq1, q2, dq2, M1, M2)
    offset1 = (traj1.shape[1] / hertz + offset) - t1

    traj2, vel2, ts2 = trajectory(q2, dq2, q3, dq3, M1, M2, hertz=hertz, offset=offset1)
    return (
        np.concatenate((traj1, traj2), axis=1),
        np.concatenate((vel1, vel2), axis=1),
        np.concatenate((ts1, t1 + ts2)),
    )


# =========================================================================
# NN basis-function trajectory (used for throw motion)
# =========================================================================

def new_trajectory(s, q0, qT, w, T):
    """Generate trajectory using NN-learned basis function weights.

    Ensures boundary conditions: q(0)=q0, q(T)=qT, dq(0)=dq(T)=0.

    Args:
        s: normalized timestep array, shape (S,), values in [0, 1].
        q0: initial joint angles, shape (k,).
        qT: final joint angles, shape (k,).
        w: basis function weights, shape (B, k).
        T: total trajectory time (seconds).

    Returns:
        (q, dq, ddq, dddq, t) - position, velocity, acceleration, jerk, time.
    """
    s = s[:, None]

    # 5th-order polynomial (ensures zero velocity/acceleration at boundaries)
    f = 10 * s**3 - 15 * s**4 + 6 * s**5
    df = 30 * (s**2 - 2 * s**3 + s**4)
    ddf = 60 * (s - 3 * s**2 + 2 * s**3)
    dddf = 60 * (1 - 6 * s + 6 * s**2)

    # Boundary-vanishing function
    g = s**3 * (s - 1)**3
    dg = 3 * (-s**2 + 4 * s**3 - 5 * s**4 + 2 * s**5)
    ddg = 6 * (-s + 6 * s**2 - 10 * s**3 + 5 * s**4)
    dddg = 6 * (-1 + 12 * s - 30 * s**2 + 20 * s**3)

    # Gaussian basis functions
    B = w.shape[0]
    centers = np.arange(B) / float(B - 1)
    phi = np.exp(-(B**2) * (s - centers)**2)
    dphi = -2 * B**2 * (s - centers) * phi
    ddphi = (-2 * B**2 + 4 * B**4 * (s - centers)**2) * phi
    dddphi = (12 * B**4 * (s - centers) - 8 * B**6 * (s - centers)**3) * phi

    q = q0 + f * (qT - q0) + g * np.dot(phi, w)
    dq = (df * (qT - q0) + np.dot(dg * phi + g * dphi, w)) / T
    ddq = (ddf * (qT - q0) + np.dot(ddg * phi + 2 * dg * dphi + g * ddphi, w)) / T**2
    dddq = (dddf * (qT - q0) + np.dot(dddg * phi + 3 * ddg * dphi + 3 * dg * ddphi + g * dddphi, w)) / T**3

    t = (s * T).squeeze()
    return q, dq, ddq, dddq, t


def pad(x: np.ndarray) -> np.ndarray:
    """Pad array with zero column for 6th joint (from 5-DOF to 6-DOF)."""
    pad_shape = list(x.shape[:-1]) + [1]
    return np.concatenate((x, np.zeros(pad_shape)), axis=-1)
