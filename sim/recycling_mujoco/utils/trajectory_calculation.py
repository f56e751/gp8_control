#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def opt_time(qi, dqi, qf, dqf, M1, M2):
    num_joint = qi.shape[0]
    possible_T = [[-np.inf, np.inf]]
    for i in range(num_joint):
        T1, T2, T3 = opt_time_1D(qi[i], dqi[i], qf[i], dqf[i], M1[i], M2[i])
        possible_Ti = [[T1, T2], [T3, np.inf]]
        new_possible_T = []
        for range1 in possible_T:
            for range2 in possible_Ti:
                lb = max(range1[0], range2[0])
                ub = min(range1[1], range2[1])
                if lb <= ub:
                    new_possible_T.append([lb, ub])
        possible_T = new_possible_T
    min_T = np.inf
    for range_final in possible_T:
        if range_final[0] < min_T:
            min_T = range_final[0]
    return min_T

def opt_time_1D(xi, vi, xf, vf, M1, M2):
    delta_x = xf - xi
    if vi * vf > 0:
        if vi > 0:
            vi, vf, delta_x = -vi, -vf, -delta_x    # dual problem    
        if delta_x > (2 * M1 * M1 - vi * vi - vf * vf) / (2 * M2):
            T1 = (delta_x - (2 * M1 * M1 - vi * vi - vf * vf) / (2 * M2)) / M1 + (2 * M1 - vi - vf) / M2
            T2 = np.inf
            T3 = np.inf
        elif delta_x > (vi + vf) * np.abs(vi - vf) / (2 * M2):
            T1 = (-(vi + vf) + np.sqrt(2 * (vi * vi + vf * vf) + 4 * M2 * delta_x)) / M2
            T2 = np.inf
            T3 = np.inf
        else:
            if delta_x > -(2 * M1 * M1 - vi * vi - vf * vf) / (2 * M2):
                T1 = ((vi + vf) + np.sqrt(2 * (vi * vi + vf * vf) - 4 * M2 * delta_x)) / M2
            else:
                T1 = -(delta_x + (2 * M1 * M1 - vi * vi - vf * vf) / (2 * M2)) / M1 + (2 * M1 + vi + vf) / M2
            if delta_x > -(vi * vi + vf * vf) / (2 * M2):
                T2 = (-(vi + vf) - np.sqrt(2 * (vi * vi + vf * vf) + 4 * M2 * delta_x)) / M2
                T3 = (-(vi + vf) + np.sqrt(2 * (vi * vi + vf * vf) + 4 * M2 * delta_x)) / M2
            else:
                T2 = np.inf
                T3 = np.inf
    else:
        if delta_x > (2 * M1 * M1 - vi * vi - vf * vf) / (2 * M2):
            T1 = (delta_x - (2 * M1 * M1 - vi * vi - vf * vf) / (2 * M2)) / M1 + (2 * M1 - vi - vf) / M2
        elif delta_x > (vi + vf) * np.abs(vi - vf) / (2 * M2):
            T1 = (-(vi + vf) + np.sqrt(2 * (vi * vi + vf * vf) + 4 * M2 * delta_x)) / M2
        elif delta_x > -(2 * M1 * M1 - vi * vi - vf * vf) / (2 * M2):
            T1 = ((vi + vf) + np.sqrt(2 * (vi * vi + vf * vf) - 4 * M2 * delta_x)) / M2
        else:
            T1 = -(delta_x + (2 * M1 * M1 - vi * vi - vf * vf) / (2 * M2)) / M1 + (2 * M1 + vi + vf) / M2
        T2 = np.inf
        T3 = np.inf
    return (T1, T2, T3)

def trajectory_1D(xi, vi, xf, vf, M1, M2, T, hertz = 1000.0, offset = 0.0):
    L = int((T - offset) * hertz)
    time = np.arange(L + 1) / hertz + offset
    trajectory = np.zeros(L + 1)
    velocity = np.zeros(L + 1)
    if vf >= vi:
        N = (M2 * (xf - xi) + (vi * vi - vf * vf) / 2) / (vi - vf + T * M2)
    else:
        N = (M2 * (xf - xi) + (vf * vf - vi * vi) / 2) / (vf - vi + T * M2)
    if (N - vi) * (N - vf) < 0:
        t1 = np.abs(N - vi) / M2
        t2 = np.abs(vf - N) / M2
        t12 = T - (t1 + t2)
        if N > vi:
            idx1 = int(t1 * hertz) + 1
            idx2 = int((t1 + t12) * hertz) + 1
            trajectory[:idx1] = xi + vi * time[:idx1] + 0.5 * M2 * time[:idx1] * time[:idx1]
            trajectory[idx1 : idx2] = xi + vi * t1 + 0.5 * M2 * t1 * t1 + N * (time[idx1 : idx2] - t1)
            trajectory[idx2:] = xi + vi * t1 + 0.5 * M2 * t1 * t1 + N * t12 + N * (time[idx2:] - (t1 + t12)) + 0.5 * M2 * (time[idx2:] - (t1 + t12)) * (time[idx2:] - (t1 + t12))
            velocity[:idx1] = vi + M2 * time[:idx1]
            velocity[idx1 : idx2] = N
            velocity[idx2:] = N + M2 * (time[idx2:] - (t1 + t12))
        else:
            idx1 = int(t1 * hertz) + 1
            idx2 = int((t1 + t12) * hertz) + 1
            trajectory[:idx1] = xi + vi * time[:idx1] - 0.5 * M2 * time[:idx1] * time[:idx1]
            trajectory[idx1 : idx2] = xi + vi * t1 - 0.5 * M2 * t1 * t1 + N * (time[idx1 : idx2] - t1)
            trajectory[idx2:] = xi + vi * t1 - 0.5 * M2 * t1 * t1 + N * t12 + N * (time[idx2:] - (t1 + t12)) - 0.5 * M2 * (time[idx2:] - (t1 + t12)) * (time[idx2:] - (t1 + t12))
            velocity[:idx1] = vi - M2 * time[:idx1]
            velocity[idx1 : idx2] = N
            velocity[idx2:] = N - M2 * (time[idx2:] - (t1 + t12))
            
    else:
        if N > vi:
            b = vi + vf + T * M2
            c = (vi * vi + vf * vf) / 2 + M2 * (xf - xi)
            N = (b - np.sqrt(b * b - 4 * c + 1e-8)) / 2
            if np.abs(N) > M1:
                return None
            
            t1 = np.abs(N - vi) / M2
            t2 = np.abs(vf - N) / M2
            t12 = T - (t1 + t2)
            
            idx1 = int(t1 * hertz) + 1
            idx2 = int((t1 + t12) * hertz) + 1
            trajectory[:idx1] = xi + vi * time[:idx1] + 0.5 * M2 * time[:idx1] * time[:idx1]
            trajectory[idx1 : idx2] = xi + vi * t1 + 0.5 * M2 * t1 * t1 + N * (time[idx1 : idx2] - t1)
            trajectory[idx2:] = xi + vi * t1 + 0.5 * M2 * t1 * t1 + N * t12 + N * (time[idx2:] - (t1 + t12)) - 0.5 * M2 * (time[idx2:] - (t1 + t12)) * (time[idx2:] - (t1 + t12))
            velocity[:idx1] = vi + M2 * time[:idx1]
            velocity[idx1 : idx2] = N
            velocity[idx2:] = N - M2 * (time[idx2:] - (t1 + t12))
            
        else:
            b = vi + vf - T * M2
            c = (vi * vi + vf * vf) / 2 - M2 * (xf - xi)
            N = (b + np.sqrt(b * b - 4 * c + 1e-8)) / 2
            if np.abs(N) > M1:
                return None
            
            t1 = np.abs(N - vi) / M2
            t2 = np.abs(vf - N) / M2
            t12 = T - (t1 + t2)
            
            idx1 = int(t1 * hertz) + 1
            idx2 = int((t1 + t12) * hertz) + 1
            trajectory[:idx1] = xi + vi * time[:idx1] - 0.5 * M2 * time[:idx1] * time[:idx1]
            trajectory[idx1 : idx2] = xi + vi * t1 - 0.5 * M2 * t1 * t1 + N * (time[idx1 : idx2] - t1)
            trajectory[idx2:] = xi + vi * t1 - 0.5 * M2 * t1 * t1 + N * t12 + N * (time[idx2:] - (t1 + t12)) + 0.5 * M2 * (time[idx2:] - (t1 + t12)) * (time[idx2:] - (t1 + t12))
            velocity[:idx1] = vi - M2 * time[:idx1]
            velocity[idx1 : idx2] = N
            velocity[idx2:] = N + M2 * (time[idx2:] - (t1 + t12))
                     
    return trajectory, velocity

def trajectory_1D_swing(xi, vi, xf, vf, M1, M2, T, hertz = 1000.0, offset = 0.0):
    L = int((T - offset) * hertz)
    time = np.arange(L + 1) / hertz + offset
    trajectory = np.zeros(L + 1)
    velocity = np.zeros(L + 1)
    if vf >= vi:
        N = (M2 * (xf - xi) + (vi * vi - vf * vf) / 2) / (vi - vf + T * M2)
    else:
        N = (M2 * (xf - xi) + (vf * vf - vi * vi) / 2) / (vf - vi + T * M2)
    if (N - vi) * (N - vf) < 0:
        t1 = np.abs(N - vi) / M2
        t2 = np.abs(vf - N) / M2
        t12 = T - (t1 + t2)
        if N > vi:
            idx1 = int(t1 * hertz) + 1
            idx2 = int((t1 + t12) * hertz) + 1
            trajectory[:idx1] = xi + vi * time[:idx1] + 0.5 * M2 * time[:idx1] * time[:idx1]
            trajectory[idx1 : idx2] = xi + vi * t1 + 0.5 * M2 * t1 * t1 + N * (time[idx1 : idx2] - t1)
            trajectory[idx2:] = xi + vi * t1 + 0.5 * M2 * t1 * t1 + N * t12 + N * (time[idx2:] - (t1 + t12)) + 0.5 * M2 * (time[idx2:] - (t1 + t12)) * (time[idx2:] - (t1 + t12))
            velocity[:idx1] = vi + M2 * time[:idx1]
            velocity[idx1 : idx2] = N
            velocity[idx2:] = N + M2 * (time[idx2:] - (t1 + t12))
        else:
            idx1 = int(t1 * hertz) + 1
            idx2 = int((t1 + t12) * hertz) + 1
            trajectory[:idx1] = xi + vi * time[:idx1] - 0.5 * M2 * time[:idx1] * time[:idx1]
            trajectory[idx1 : idx2] = xi + vi * t1 - 0.5 * M2 * t1 * t1 + N * (time[idx1 : idx2] - t1)
            trajectory[idx2:] = xi + vi * t1 - 0.5 * M2 * t1 * t1 + N * t12 + N * (time[idx2:] - (t1 + t12)) - 0.5 * M2 * (time[idx2:] - (t1 + t12)) * (time[idx2:] - (t1 + t12))
            velocity[:idx1] = vi - M2 * time[:idx1]
            velocity[idx1 : idx2] = N
            velocity[idx2:] = N - M2 * (time[idx2:] - (t1 + t12))
            
    else:
        if N > vi:
            b = vi + vf + T * M2
            c = (vi * vi + vf * vf) / 2 + M2 * (xf - xi)
            N = (b - np.sqrt(b * b - 4 * c + 1e-8)) / 2
            if np.abs(N) > M1:
                return None
            
            t1 = np.abs(N - vi) / M2
            t2 = np.abs(vf - N) / M2
            t12 = T - (t1 + t2)
            
            idx1 = int(t1 * hertz) + 1
            idx2 = int((t1 + t12) * hertz) + 1
            trajectory[:idx1] = xi + vi * time[:idx1] + 0.5 * M2 * time[:idx1] * time[:idx1]
            trajectory[idx1 : idx2] = xi + vi * t1 + 0.5 * M2 * t1 * t1 + N * (time[idx1 : idx2] - t1)
            trajectory[idx2:] = xi + vi * t1 + 0.5 * M2 * t1 * t1 + N * t12 + N * (time[idx2:] - (t1 + t12)) - 0.5 * M2 * (time[idx2:] - (t1 + t12)) * (time[idx2:] - (t1 + t12))
            velocity[:idx1] = vi + M2 * time[:idx1]
            velocity[idx1 : idx2] = N
            velocity[idx2:] = N - M2 * (time[idx2:] - (t1 + t12))
            
        else:
            b = vi + vf - T * M2
            c = (vi * vi + vf * vf) / 2 - M2 * (xf - xi)
            N = (b + np.sqrt(b * b - 4 * c + 1e-8)) / 2
            if np.abs(N) > M1:
                return None
            
            t1 = np.abs(N - vi) / M2
            t2 = np.abs(vf - N) / M2
            t12 = T - (t1 + t2)
            
            idx1 = int(t1 * hertz) + 1
            idx2 = int((t1 + t12) * hertz) + 1
            trajectory[:idx1] = xi + vi * time[:idx1] - 0.5 * M2 * time[:idx1] * time[:idx1]
            trajectory[idx1 : idx2] = xi + vi * t1 - 0.5 * M2 * t1 * t1 + N * (time[idx1 : idx2] - t1)
            trajectory[idx2:] = xi + vi * t1 - 0.5 * M2 * t1 * t1 + N * t12 + N * (time[idx2:] - (t1 + t12)) + 0.5 * M2 * (time[idx2:] - (t1 + t12)) * (time[idx2:] - (t1 + t12))
            velocity[:idx1] = vi - M2 * time[:idx1]
            velocity[idx1 : idx2] = N
            velocity[idx2:] = N + M2 * (time[idx2:] - (t1 + t12))
                     
    return time, trajectory, velocity

def estimate_dq_end(t, q, k=5):
    num_points = min(k, len(t)-1)

    if num_points <= 0:
        return np.zeros_like(q[0])
    
    dq_end_list = []
    for i in range(1, num_points + 1):
        dt = t[-i] - t[-i-1]
        if dt > 1e-6: 
            dq_end_list.append((q[-i] - q[-i-1]) / dt)

    if not dq_end_list:
        return np.zeros_like(q[0])
    
    return np.mean(dq_end_list, axis=0)

def get_shortest_path_joint(q_current, q_target):
    diff = q_target - q_current
    shortest_diff = (diff + np.pi) % (2 * np.pi) - np.pi
    q_target_corrected = q_current + shortest_diff

    return q_target_corrected

def trajectory_bangbang(qi, dqi, qf, dqf, M1, M2, hertz = 1000.0, offset = 0.0):
    qi = np.array(qi)
    dqi = np.array(dqi)
    qf = np.array(qf)
    dqf = np.array(dqf)
    num_joint = qi.shape[0]
    # time = np.zeros(num_joint)
    # for i in range(num_joint):
    #     time[i] = opt_time_1D(qi[i], dqi[i], qf[i], dqf[i], M1[i], M2[i])
    
    # T = np.max(time)
    T = opt_time(qi, dqi, qf, dqf, M1, M2)
    L = int((T - offset) * hertz)
    trajectory = np.zeros((num_joint, L + 1))
    velocity = np.zeros((num_joint, L + 1))
    for i in range(num_joint):
        trajectory[i, :], velocity[i, :] = trajectory_1D(qi[i], dqi[i], qf[i], dqf[i], M1[i], M2[i], T, hertz = hertz, offset = offset)
        
    return trajectory, velocity

def concat_with_bangbang_before(t_push, q_push, dq_push, q_push_before, M1, M2, hertz, dqi=None):
    q_start = q_push[0]
    dq_start = dq_push[0]
    if dqi is None:
        dqi = np.zeros_like(dq_start)
    else:
        dqi = np.clip(dqi, -M1, M1)
    dqf = np.clip(dq_start, -M1, M1)

    q_before, dq_before = trajectory_bangbang(q_push_before, dqi, q_start, dqf, M1, M2, hertz=hertz)
    q_before = q_before.T
    dq_before = dq_before.T
    
    t_before = np.arange(q_before.shape[0]) * (1.0 / hertz)
    t_push_shifted = t_before[-1] + t_push

    t_all = np.concatenate([t_before, t_push_shifted[1:]])
    q_all = np.vstack([q_before, q_push[1:]])
    dq_all = np.vstack([dq_before, dq_push[1:]])

    return t_all, q_all, dq_all

def get_se3_from_xy(pos, x_vec, y_vec):
        p = np.array(pos)

        y_axis = np.array(y_vec)
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0])
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = np.array(x_vec)
        if np.linalg.norm(x_axis) < 1e-6:
            y_axis = np.array([1.0, 0.0, 0.0])
        x_axis = x_axis / np.linalg.norm(x_axis)

        z_axis = np.cross(x_axis, y_axis)

        if np.linalg.norm(z_axis) < 1e-6:
            z_axis = np.array([0.0, 0.0, 1.0]) 
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        T = np.eye(4)
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis  # pushing vector
        T[:3, 2] = z_axis  # side vector
        T[:3, 3] = p

        return T