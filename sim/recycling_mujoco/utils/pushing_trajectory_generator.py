#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from utils.trajectory_calculation import *

class PushingTrajectoryGenerator:
    def __init__(self, robot, M1, M2, M2h, hertz=240.0, hover_height=0.02):
        self.robot = robot
        self.M1 = M1
        self.M2 = M2
        self.M2h = M2h
        self.hertz = hertz
        self.pushing_after_height = hover_height

    def trajectory_linear_optimal(self, xi, xf, v, a):
        distance = np.linalg.norm(xf - xi)
        times = opt_time_1D(0.0, 0.0, distance, 0.0, v, a)
        T_ideal = times[0]
        t_array, traj_cart, vel_cart = trajectory_1D_swing(0.0, 0.0, distance, 0.0, v, a, T_ideal, self.hertz)
        
        return t_array, traj_cart, vel_cart, T_ideal

    def calculate_pushing_geometry(self, obj_pos, target_xy, pushing_params):
            p_start_margin = pushing_params.get('start_margin', 0.10)
            p_dist = pushing_params.get('distance', 0.15)

            target_pos = np.array([target_xy[0], target_xy[1], obj_pos[2]])
            target_vec = target_pos - obj_pos
            target_dist = np.linalg.norm(target_vec)
            unit_vec = target_vec / (target_dist + 1e-6)
            
            start_pos = obj_pos - unit_vec * p_start_margin
            end_pos = obj_pos + unit_vec * p_dist
            hover_pos = np.concatenate((start_pos[:2], [self.pushing_after_height]))

            return unit_vec, start_pos, end_pos, hover_pos

    def get_pushing_SE3_poses(self, start_pos, end_pos, push_dir=None, swing_angle=0.0):
            start_pos = np.asarray(start_pos, dtype=float).reshape(3)
            end_pos = np.asarray(end_pos, dtype=float).reshape(3)
            
            # Basis construction
            if push_dir is not None:
                y_hat = np.asarray(push_dir, dtype=float).reshape(3)
                y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-12)
            else:
                line_vec = end_pos - start_pos
                line_len = np.linalg.norm(line_vec)
                if line_len < 1e-9:
                    return None, None, None, None
                else:
                    y_hat = line_vec / line_len
            
            world_z = np.array([0.0, 0.0, 1.0])
            z_temp = np.cross(y_hat, world_z)
            
            if np.linalg.norm(z_temp) < 1e-6:
                # Vertical pushing fallback
                z_hat = np.array([1.0, 0.0, 0.0]) 
            else:
                z_hat = z_temp / np.linalg.norm(z_temp)

            # First entry check (vs Reference Down)
            x_check = np.cross(y_hat, z_hat)
            if np.dot(x_check, np.array([0.0, 0.0, -1.0])) < 0:
                z_hat = -z_hat
                    
            # X_hat & Re-orthogonalization
            x_hat = np.cross(y_hat, z_hat)
            x_hat = x_hat / (np.linalg.norm(x_hat) + 1e-12)
            z_hat = np.cross(x_hat, y_hat)
            z_hat = z_hat / (np.linalg.norm(z_hat) + 1e-12)
            
            # Base rotation matrix
            R_base_mat = np.column_stack([x_hat, y_hat, z_hat])
            # base_r = R.from_dcm(R_base_mat)
            base_r = R.from_matrix(R_base_mat)
            
            # Apply swing angle
            if abs(swing_angle) < 1e-6:
                r_start = base_r
                r_end = base_r
            else:
                # 3-Point Swing
                angle_rad = np.radians(swing_angle)
                
                rot_back   = R.from_rotvec(z_hat * -angle_rad)
                rot_follow = R.from_rotvec(z_hat * angle_rad)
                
                r_start  = rot_back * base_r
                r_end    = rot_follow * base_r
                
            # Construct SE(3) matrices
            T_start = np.eye(4)
            # T_start[:3, :3] = r_start.as_dcm()
            T_start[:3, :3] = r_start.as_matrix()
            T_start[:3, 3] = start_pos
            
            T_end = np.eye(4)
            # T_end[:3, :3] = r_end.as_dcm()
            T_end[:3, :3] = r_end.as_matrix()
            T_end[:3, 3] = end_pos
            
            return T_start, T_end, z_hat, base_r
    
    def apply_variable_time_scaling_to_spline(
        self, times, q_path, M1, M2, hertz, 
        dq_start=None, dq_end=None, 
        margin=1.0, resolution=1000, max_iter=5, 
        debug=False, 
        beta_smooth_method="softmin",   # "hardmin"(기존), "softmin", "softmin+movavg"
        softmin_alpha=50.0,             # 30~100 추천 (클수록 hardmin에 가까움)
        movavg_window=21,               # softmin+movavg 일 때 사용 (홀수 권장)
        movavg_passes=1, 
        beta_post_movavg_window=51,   # 31~101 추천(홀수)
        beta_post_movavg_passes=1
    ):
        # Helper
        def _max_ratio_info(arr, lim):
            ratio = np.abs(arr) / lim[None, :]
            flat = np.argmax(ratio)
            ti, ji = np.unravel_index(flat, ratio.shape)
            return float(ratio[ti, ji]), (int(ti), int(ji)), float(arr[ti, ji]), float(lim[ji])
            
        def _smooth_beta_profile(beta_v_limit, beta_a_limit,
                             method="softmin", alpha=50.0,
                             movavg_window=21, movavg_passes=1):
            v = np.asarray(beta_v_limit, dtype=float)
            a = np.asarray(beta_a_limit, dtype=float)

            if method == "hardmin":
                beta = np.minimum(v, a)

            elif method in ("softmin", "softmin+movavg"):
                # 안정적인 softmin (log-sum-exp 형태)
                # softmin(x,y) = m - (1/α) log(exp(-α(x-m)) + exp(-α(y-m)))
                alpha = float(alpha)
                m = np.minimum(v, a)
                z = np.exp(-alpha * (v - m)) + np.exp(-alpha * (a - m))
                beta = m - (1.0 / alpha) * np.log(z)
            else:
                raise ValueError("Unknown beta_smooth_method")

            # 선택: 이동평균으로 한 번 더 완화 (단, 제약 위반 방지를 위해 beta를 절대 키우지 않음)
            if method == "softmin+movavg":
                w = int(movavg_window)
                if w >= 3:
                    if w % 2 == 0:
                        w += 1
                    pad = w // 2
                    kernel = np.ones(w, dtype=float) / w
                    b = beta
                    for _ in range(int(movavg_passes)):
                        bpad = np.pad(b, (pad, pad), mode="reflect")
                        b_sm = np.convolve(bpad, kernel, mode="valid")
                        b = np.minimum(b, b_sm)  # 절대 증가시키지 않음(안전)
                    beta = b

            return beta
        
        def _lower_envelope_movavg(b, window=51, passes=1):
            w = int(window)
            if w < 3:
                return b
            if w % 2 == 0:
                w += 1
            pad = w // 2
            kernel = np.ones(w, dtype=float) / w

            out = b
            for _ in range(int(passes)):
                bpad = np.pad(out, (pad, pad), mode="reflect")
                b_sm = np.convolve(bpad, kernel, mode="valid")
                # 안전: beta는 절대 커지면 안됨(제약 위반 방지)
                out = np.minimum(out, b_sm)
            return out
        
        # Variable set
        times = np.asarray(times, dtype=float)
        q_path = np.asarray(q_path, dtype=float)
        M1 = np.maximum(np.asarray(M1, dtype=float), 1e-6)
        M2 = np.maximum(np.asarray(M2, dtype=float), 1e-6)
        current_margin = max(1.0, float(margin))
        
        # Spline fitting
        t0 = times[0]
        s = times - t0  # t_original
        S_total = s[-1]  # T_old
        
        if S_total <= 0:
            raise ValueError("Total time duration must be positive.")
        if not np.all(np.diff(s) > 0):
            raise ValueError("Times must be strictly increasing.")
        
        if dq_start is None:
            bc_start = (1, np.zeros_like(q_path[0]))
        else:
            bc_start = (1, np.asarray(dq_start, dtype=float))

        if dq_end is None:
            bc_end = (1, np.zeros_like(q_path[-1]))
        else:
            bc_end = (1, np.asarray(dq_end, dtype=float))        

        cs = CubicSpline(s, q_path, axis=0, bc_type=(bc_start, bc_end))

        # Sampling
        s_grid = np.linspace(0, S_total, resolution)
        ds = s_grid[1] - s_grid[0]

        q_s = cs(s_grid, 1)
        q_ss = cs(s_grid, 2)

        # Limit caculation (variable scaling)
        abs_q_s = np.abs(q_s) + 1e-12
        abs_q_ss = np.abs(q_ss) + 1e-12
        limit_v = M1 / abs_q_s
        limit_a = np.sqrt(M2 / abs_q_ss)
        beta_v_limit = np.min(limit_v, axis=1)  # beta <= M1 / |q'|
        beta_a_limit = np.min(limit_a, axis=1)  # beta <= sqrt(M2 / |q''|)
        
        beta_hard_base = np.minimum(beta_v_limit, beta_a_limit)
        beta_base = _smooth_beta_profile(
            beta_v_limit, beta_a_limit,
            method=beta_smooth_method,
            alpha=softmin_alpha,
            movavg_window=movavg_window,
            movavg_passes=movavg_passes
        )
        
        if debug:
            print("[TS] beta smoothing:", beta_smooth_method, "alpha=", softmin_alpha)
            print("[TS] max|beta_base - hardmin| =", np.max(np.abs(beta_base - beta_hard_base)))
            # base에서의 db/ds 비교
            db_hard = np.gradient(beta_hard_base, ds, edge_order=2)
            db_base = np.gradient(beta_base, ds, edge_order=2)
            print("[TS] max|db/ds| hard=", np.max(np.abs(db_hard)), " base=", np.max(np.abs(db_base)))
        
        final_result = None
        
        for it in range(max_iter):
            beta = beta_base / current_margin
            beta = np.minimum(beta, 1.0)
            beta = np.maximum(beta, 1e-6)
            
            beta = _lower_envelope_movavg(beta, beta_post_movavg_window, beta_post_movavg_passes)

            dt_steps = ds / beta[:-1]
            new_times = np.concatenate([[0.0], np.cumsum(dt_steps)])
            T_new = new_times[-1]

            # Resampling
            dt = 1.0 / hertz
            final_t = np.arange(0.0, T_new, dt)
            if final_t.size == 0 or (T_new - final_t[-1]) > 1e-12:
                final_t = np.append(final_t, T_new)
            
            s_of_t = PchipInterpolator(new_times, s_grid, extrapolate=False)
            s_query = s_of_t(final_t)
            s_query[np.isnan(s_query)] = 0.0
            s_query[np.isposinf(s_query)] = S_total
            s_query[np.isneginf(s_query)] = 0.0
            s_query = np.clip(s_query, 0.0, S_total)
            
            s_dot_val  = s_of_t.derivative(1)(final_t)
            s_ddot_val = s_of_t.derivative(2)(final_t)
            
            s_dot_val = np.clip(s_dot_val, 0.0, 1e9)
            s_ddot_val = np.clip(s_ddot_val, -1e9, 1e9)
            beta_ds = np.gradient(beta, ds, edge_order=2)
            
            q_val = cs(s_query)
            q_s_val = cs(s_query, 1)
            q_ss_val = cs(s_query, 2)
            
            dq_out = q_s_val * s_dot_val[:, None]  # dq/dt = q'(s) * s_dot
            ddq_geom  = q_ss_val * (s_dot_val[:, None]**2)
            ddq_sddot = q_s_val  * (s_ddot_val[:, None])
            ddq_out   = ddq_geom + ddq_sddot  # d2q/dt2 = q''(s) * s_dot^2 + q'(s) * s_ddot
            
            # Validation
            vel_info = _max_ratio_info(dq_out, M1)
            acc_info = _max_ratio_info(ddq_out, M2)
            acc_geom_info  = _max_ratio_info(ddq_geom, M2)
            acc_sddot_info = _max_ratio_info(ddq_sddot, M2)
            
            vel_violation = vel_info[0]
            acc_violation = acc_info[0]
            
            need = max(vel_violation, np.sqrt(acc_violation))
                
            if need <= 1.001:
                final_result = (final_t, q_val, dq_out, ddq_out, T_new / S_total)
                break
            else:
                scale_factor = max(1.05, need)
                current_margin *= scale_factor
                
        if final_result is None:
            final_result = (final_t, q_val, dq_out, ddq_out, T_new / S_total)
        
        final_t, q_val, dq_out, ddq_out, k = final_result
        
        return final_t, q_val, dq_out, ddq_out, k

    def solve_linear_swing_segment_M2_heu(self, start_pos, end_pos, pushing_velocity, acc_limit, obj_pos, push_dir=None, swing_angle=0.0, swing_radius=0.325, seed_q=None):
            # Setting
            start_pos = np.asarray(start_pos, dtype=float).reshape(3)
            end_pos = np.asarray(end_pos, dtype=float).reshape(3)
            obj_pos = np.asarray(obj_pos, dtype=float).reshape(3)
            
            # seed initialization
            if seed_q is None:
                try:
                    if hasattr(self.robot, "get_joint_positions"):
                        seed_q = np.array(self.robot.get_joint_positions())
                    else:
                        seed_q = np.zeros(6)
                except:
                    seed_q = np.zeros(6)
            else:
                seed_q = np.asarray(seed_q, dtype=float).reshape(-1)
                if seed_q.shape[0] != 6: seed_q = np.zeros(6)
            
            T_start, T_end, z_hat, base_r = self.get_pushing_SE3_poses(start_pos, end_pos, push_dir=push_dir, swing_angle=swing_angle)
            
            if T_start is None: # Line length too short
                dt = 1.0 / self.hertz
                times = np.array([0.0, dt], dtype=float)
                valid_q = np.vstack([seed_q, seed_q]).astype(float)
                valid_v = np.zeros((2, 6), dtype=float)
                # Fallback z_ret
                z_ret = np.array([1.0, 0.0, 0.0])
                return times, valid_q, valid_v, z_ret
            
            # Push direction & Cartesian waypoints
            if push_dir is None:
                line_vec = end_pos - start_pos
                line_len = np.linalg.norm(line_vec)
                push_dir = line_vec / line_len
            else:
                line_vec = end_pos - start_pos
                line_len = np.linalg.norm(line_vec)

            t_base, s_base, v_base, T_ideal = self.trajectory_linear_optimal(start_pos, end_pos, pushing_velocity, acc_limit)
            s_base = np.clip(s_base, 0.0, line_len)
            traj_points = s_base.copy()

            # Target reaching check
            if len(traj_points) > 0 and abs(traj_points[-1] - line_len) > 1e-6:
                traj_points = np.append(traj_points, line_len)
            elif len(traj_points) == 0:
                traj_points = np.array([0.0, line_len])

            # Increase function
            traj_points = np.maximum.accumulate(traj_points)

            # Impact length
            dist_to_impact = np.dot(obj_pos - start_pos, push_dir)
            dist_to_impact = np.clip(dist_to_impact, 0.0, line_len)

            # Impact point insertion
            insert_idx = np.searchsorted(traj_points, dist_to_impact)
            tol = 1e-6
            is_duplicate = False
            if insert_idx > 0 and abs(traj_points[insert_idx-1] - dist_to_impact) < tol:
                is_duplicate = True
            elif insert_idx < len(traj_points) and abs(traj_points[insert_idx] - dist_to_impact) < tol:
                is_duplicate = True

            if not is_duplicate:
                traj_points = np.insert(traj_points, insert_idx, dist_to_impact)
            
            eps_s = 1e-9
            keep = np.r_[True, np.diff(traj_points) > eps_s]
            traj_points = traj_points[keep]
            
            t_points = np.interp(traj_points, s_base, t_base)
            
            eps_t = 1e-9
            keep_t = np.r_[True, np.diff(t_points) > eps_t]
            traj_points = traj_points[keep_t]
            t_points = t_points[keep_t]
            
            # Waypoints generation
            waypoints = start_pos + push_dir * traj_points[:, np.newaxis]
            N_points = len(waypoints)

            # Impact point indexing & Lift height calculation
            i_hit = np.argmin(np.abs(traj_points - dist_to_impact))
            if i_hit == 0 or i_hit == N_points - 1:
                swing_angle = 0.0
                lift_height = 0.0
            else:
                if swing_radius > 0 and abs(swing_angle) > 1e-6:
                    rad_angle = np.radians(abs(swing_angle))
                    lift_height = swing_radius * np.sin(rad_angle)
                else:
                    lift_height = 0.0
                    
            # Arc generation
            if lift_height > 1e-6:
                indices = np.arange(N_points)
                mask = indices > i_hit  # Boolean masking
                
                dist_at_impact = traj_points[i_hit]
                dist_at_end = traj_points[-1]
                remain_len_real = dist_at_end - dist_at_impact

                if remain_len_real > 1e-6:
                    s_after = traj_points[mask] - dist_at_impact
                    s_ratio = s_after / remain_len_real
                    s_ratio = np.clip(s_ratio, 0.0, 1.0)
                    
                    z_offset = lift_height * 0.5 * (1 - np.cos(s_ratio * np.pi))
                    world_z = np.array([0.0, 0.0, 1.0])
                    waypoints[mask] += z_offset[:, np.newaxis] * world_z
            
            # t progress
            T_total = t_points[-1]
            if T_total < 1e-12: 
                T_total = 1.0
            t_arr = np.clip(t_points / T_total, 0.0, 1.0)
            
            # Hit time
            t_hit = t_arr[i_hit]
            eps = 1e-4
            t_hit = np.clip(t_hit, eps, 1.0 - eps)
            
            # SLERP Optimization
            if abs(swing_angle) < 1e-6:
                key_times = [0.0, 1.0]
                # No rotation
                # key_rots = R.from_dcm([base_r.as_dcm(), base_r.as_dcm()])
                key_rots = R.from_matrix([base_r.as_matrix(), base_r.as_matrix()])
            else:
                # 3-Point Swing
                angle_rad = np.radians(swing_angle)
                
                rot_back   = R.from_rotvec(z_hat * -angle_rad) 
                rot_follow = R.from_rotvec(z_hat * angle_rad)
                
                r_start  = rot_back * base_r 
                r_impact = base_r 
                r_end    = rot_follow * base_r 
                
                key_times = [0.0, t_hit, 1.0]
                # key_rots  = R.from_dcm([r_start.as_dcm(), r_impact.as_dcm(), r_end.as_dcm()])
                key_rots  = R.from_matrix([r_start.as_matrix(), r_impact.as_matrix(), r_end.as_matrix()])
            
            slerp = Slerp(key_times, key_rots)
            
            # Solve Inverse kinematics
            temp_q = []
            prev_q = seed_q

            for i in range(N_points):
                t = t_arr[i]
                pos = waypoints[i]
                current_r = slerp([t])
                # current_rot_mat = current_r.as_dcm()[0]
                current_rot_mat = current_r.as_matrix()[0]
                
                T = np.eye(4)
                T[:3, :3] = current_rot_mat
                T[:3, 3] = pos
                
                q = self.robot.inverse_kinematics_np_push(T, seed=prev_q)

                if q is None:
                    if prev_q is not None:
                        q = prev_q
                    else:
                        q = np.zeros(6)
                        print("Warning: IK failed at the start of segment.")

                temp_q.append(np.array(q))
                prev_q = q
            q_array = np.array(temp_q)

            # Spline & Time scaling & Resampling
            target_linear_vel = push_dir * pushing_velocity

            dq_start_target = None
            dq_end_target = estimate_dq_end(t_points, q_array, k=5)

            safe_ratio = 0.5
            dq_end_target = np.clip(dq_end_target, -self.M1 * safe_ratio, self.M1 * safe_ratio)  # dqf clip
            
            final_t, valid_q, valid_v, valid_a, k = self.apply_variable_time_scaling_to_spline(
                t_points, 
                q_array, 
                self.M1, self.M2h, self.hertz, 
                dq_start=dq_start_target, 
                dq_end=dq_end_target, 
                margin=1.0, 
                resolution=1000, 
                debug=False,  
                beta_smooth_method="softmin+movavg",
                softmin_alpha=10.0,
                beta_post_movavg_window=51,
                beta_post_movavg_passes=2
            )
            
            return final_t, valid_q, valid_v

    def generate_robot_pushing_trajectory(self, obj_pos, target_xy, pushing_params, last_q_end): 
            p_vel = pushing_params.get('velocity', 1.2)
            p_acc = pushing_params.get('acc_limit', 15.0)
            swing_angle = pushing_params.get('swing_angle', 0.0)
            swing_radius = pushing_params.get('swing_radius', 0.10)
            
            # Pushing vector & position setting
            unit_vec, start_pos, end_pos, _ = self.calculate_pushing_geometry(obj_pos, target_xy, pushing_params)
            
            t_linear, q_linear, dq_linear = self.solve_linear_swing_segment_M2_heu(
                start_pos, 
                end_pos, 
                p_vel, 
                p_acc, 
                obj_pos, 
                push_dir=unit_vec, 
                swing_angle=swing_angle, 
                swing_radius=swing_radius, 
                seed_q=last_q_end, 
                )
            
            q_linear = q_linear.copy()
            dq_linear = dq_linear.copy()
            
            # Start pos offset
            if last_q_end is not None:
                ref_pos = last_q_end
            else:
                ref_pos = q_linear[0]
            
            optimized_start = get_shortest_path_joint(ref_pos, q_linear[0])
            start_diff = optimized_start - q_linear[0]
            q_linear = q_linear + start_diff
            
            # Step 1 & 2: [Hover -> Start] (BangBang) + [Start -> End] (Linear)
            t_full, q_full, dq_full = concat_with_bangbang_before(t_linear, q_linear, dq_linear, last_q_end, self.M1, self.M2, self.hertz)
            
            return t_full, q_full, dq_full
    
    def generate_robot_pushingafter_trajectory(self, next_obj_pos, next_target_xy, pushing_params, last_q_end, last_dq_end):
        # Next pushing vector & pos upper next
        next_unit_vec, _, _, pos_upper_next = self.calculate_pushing_geometry(next_obj_pos, next_target_xy, pushing_params)

        next_vec_for_ik = next_unit_vec
        
        x_axis_vec = np.array([0.0, 0.0, -1.0]) 
        T_next_hover = get_se3_from_xy(pos_upper_next, x_axis_vec, next_vec_for_ik)
        q_upper_next_raw = np.array(self.robot.inverse_kinematics_np_push(T_next_hover))
        
        if q_upper_next_raw is None:
            print("  [Warning] IK failed for Next Hover. Holding position.")
            q_upper_next_raw = last_q_end
        else:
            q_upper_next_raw = np.array(q_upper_next_raw)
            
        # Unwinding check
        joint_pos_limits, _ = self.robot.get_bounds()
        q_shortest_path = get_shortest_path_joint(last_q_end, q_upper_next_raw)
        q_final_next = np.zeros(6)
        unwound_joints = []
        
        for j in range(6):
            margin = 0.01
            min_limit = joint_pos_limits[j][0]
            max_limit = joint_pos_limits[j][1]
            val_shortest = q_shortest_path[j]
            
            if min_limit - margin <= val_shortest <= max_limit + margin:
                q_final_next[j] = val_shortest
            else:
                q_final_next[j] = q_upper_next_raw[j]
                unwound_joints.append(j+1)
                
        # if len(unwound_joints) > 0:
        #     print(f"  [Action] Object {i}: Unwinding applied to joints {unwound_joints}.")
        
        q_upper_next = q_final_next
        dqi = np.clip(last_dq_end, -self.M1, self.M1)
        dqf = np.zeros_like(dqi)
        
        q_after, dq_after = trajectory_bangbang(last_q_end, dqi, q_upper_next, dqf, self.M1, self.M2, hertz=self.hertz)
        q_after = q_after.T
        dq_after = dq_after.T
        t_after = np.arange(q_after.shape[0]) * (1.0 / self.hertz)
        
        return t_after, q_after, dq_after