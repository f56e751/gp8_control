#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from utils.Lie_numpy import *

from scipy.spatial.transform import Rotation as R

class Yaskawa():
    def __init__(self):
        # super(Yaskawa, self).__init__()
        self.screws, self.joint_positions = self.get_screws()
        self.M = self.get_M()
        self.joint_bounds, self.jointvel_bounds = self.get_bounds()
        
    def get_screws(self):
        w_1 = np.array([0.0, 0.0, 1.0])
        w_2 = np.array([0.0, 1.0, 0.0])
        w_3 = np.array([0.0, -1.0, 0.0])
        w_4 = np.array([-1.0, 0.0, 0.0])	
        w_5 = np.array([0.0, -1.0, 0.0])
        w_6 = np.array([-1.0, 0.0, 0.0])

        q_1 = np.array([0.0, 0.0, 0.330])
        q_2 = np.array([0.04, 0.0, 0.330])
        q_3 = np.array([0.04, 0.0, 0.675])
        q_4 = np.array([0.38, 0.0, 0.715])
        q_5 = np.array([0.38, 0.0, 0.715])
        q_6 = np.array([0.38, 0.0, 0.715])

        S0 = np.concatenate((w_1, -np.cross(w_1, q_1)))
        S1 = np.concatenate((w_2, -np.cross(w_2, q_2)))
        S2 = np.concatenate((w_3, -np.cross(w_3, q_3)))
        S3 = np.concatenate((w_4, -np.cross(w_4, q_4)))
        S4 = np.concatenate((w_5, -np.cross(w_5, q_5)))
        S5 = np.concatenate((w_6, -np.cross(w_6, q_6)))
        
        return [S0, S1, S2, S3, S4, S5], [q_1, q_2, q_3, q_4, q_5, q_6]

    def get_M(self):
        return np.array([[1, 0, 0, 0.705],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0.715],
                         [0, 0, 0, 1]])
        
    def get_bounds(self):
        M1 = np.array((455, 385, 520, 550, 550, 1000)) * (np.pi / 180)

        bound1 = (-np.radians(170), np.radians(170))
        bound2 = (-np.radians(65), np.radians(145))
        bound3 = (-np.radians(70), np.radians(190))
        bound4 = (-np.radians(190), np.radians(190))
        # Real cell J5 +방향 추가 장착판의 측정 상한과 일치시킨다.
        bound5 = (-np.radians(135), 1.060747742652893)
        bound6 = (-np.radians(360), np.radians(360))

        bound7 = (-M1[0], M1[0])
        bound8 = (-M1[1], M1[1])
        bound9 = (-M1[2], M1[2])
        bound10 = (-M1[3], M1[3])
        bound11 = (-M1[4], M1[4])
        bound12 = (-M1[5], M1[5])

        return [bound1, bound2, bound3, bound4, bound5, bound6], [bound7, bound8, bound9, bound10, bound11, bound12]
        
        
    ##### Kinematics #####    
    def forward_kinematics(self, q):
        return forward_kinematics(q, self.screws, self.M)[1]
    
    # def inverse_kinematics(self, T):
    #     return inverse_kinematics(self.M, T, 0.1 * (np.random.random(6) - 0.5), self.screws, tolerance = 1e-6)
    
    def inverse_kinematics(self, T):
        # Obtaining theta1~theta3 with the consine law
        l_rod = self.M[0, 3] - self.joint_positions[4][0]
        joint5_pos = T[:3, 3] - l_rod * T[:3, 0]    # joint4 pos = joint5 pos = joint6 pos
        
        theta_1 = np.arctan2(joint5_pos[1], joint5_pos[0])
        dist = np.sqrt(joint5_pos[0] * joint5_pos[0] + joint5_pos[1] * joint5_pos[1])
        joint5_pos = np.array([dist, 0.0, joint5_pos[2]])
        
        joint2_zeropos = self.joint_positions[1]
        joint3_zeropos = self.joint_positions[2]
        joint5_zeropos = self.joint_positions[4]
        
        L1 = np.linalg.norm(joint3_zeropos - joint2_zeropos)
        L2 = np.linalg.norm(joint5_zeropos - joint3_zeropos)
        alpha_0 = np.arctan2(joint5_zeropos[2] - joint3_zeropos[2], joint5_zeropos[0] - joint3_zeropos[0])
        
        L = np.linalg.norm(joint5_pos - joint2_zeropos)
        if L > L1 + L2 or L < np.abs(L1 - L2):
            print("encoutered invalid SE(3)")
            return None
        alpha_1 = np.arccos((L1 * L1 + L * L - L2 * L2) / (2 * L1 * L))
        alpha_2 = np.arccos((L1 * L1 + L2 * L2 - L * L) / (2 * L1 * L2))
        alpha_3 = np.arctan2(joint5_pos[2] - joint2_zeropos[2], joint5_pos[0] - joint2_zeropos[0])
        
        theta_2 = np.pi / 2 - (alpha_1 + alpha_3)
        theta_3 = (np.pi / 2 - alpha_0) - (np.pi - alpha_2)
        
        # Obtaining theta4~theta6 with the Euler angle equation
        R_03 = R.from_euler('yz', [theta_2 - theta_3, theta_1])
        R_03_matrix = self._rotation_as_matrix(R_03)
        R_euler = np.matmul(R_03_matrix.T, T[:3, :3])
        
        theta_6, theta_5, theta_4 = self._rotation_from_matrix(R_euler).as_euler('xyx')
        theta_4, theta_5, theta_6 = -theta_4, -theta_5, -theta_6
        
        first_soln = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
        
        # Selecting one of the two solutions of the Euler angle equation
        second_soln = first_soln.copy()
        if first_soln[3] > 0:
            second_soln[3] -= np.pi
        else:
            second_soln[3] += np.pi
        second_soln[4] = -first_soln[4]
        if first_soln[5] > 0:
            second_soln[5] -= np.pi
        else:
            second_soln[5] += np.pi
            
        is_valid1 = True
        is_valid2 = True    
        for i in range(6):
            if first_soln[i] < self.joint_bounds[i][0] or first_soln[i] > self.joint_bounds[i][1]:
                is_valid1 = False
            if second_soln[i] < self.joint_bounds[i][0] or second_soln[i] > self.joint_bounds[i][1]:
                is_valid2 = False
                
        if is_valid1 and is_valid2:
            if np.abs(first_soln[3]) <= np.abs(second_soln[3]):    # This choice of solutions can be modified
                return first_soln
            else:
                return second_soln
        elif is_valid1:
            return first_soln
        elif is_valid2:
            return second_soln
        else:
            print("encoutered invalid SE(3)")
            return None
        
    
    def space_jacobian(self, q):
        link_frames = forward_kinematics(q, self.screws, self.M)[0]
        return get_SpaceJacobian(self.screws, link_frames)

    @staticmethod
    def _rotation_as_matrix(rotation: R) -> np.ndarray:
        if hasattr(rotation, "as_matrix"):
            return rotation.as_matrix()
        return rotation.as_dcm()

    @staticmethod
    def _rotation_from_matrix(matrix: np.ndarray) -> R:
        if hasattr(R, "from_matrix"):
            return R.from_matrix(matrix)
        return R.from_dcm(matrix)

    def wrap_near_bounds(self, x, ref):  # x를 ref(이전 자세) 근처로 2pi shift하되, joint bounds 안에 들어오는 후보들 중 ref에 가장 가까운 것을 선택
        x = np.asarray(x, dtype=float).copy()
        ref = np.asarray(ref, dtype=float)
        out = x.copy()
        two_pi = 2.0 * np.pi

        for i, (lo, hi) in enumerate(self.joint_bounds):
            k0 = int(np.round((ref[i] - x[i]) / two_pi))

            candidates = []
            for k in range(k0 - 2, k0 + 3):
                xi = x[i] + two_pi * k
                if lo <= xi <= hi:
                    candidates.append(xi)

            out[i] = min(candidates, key=lambda v: abs(v - ref[i])) if candidates else x[i]

        return out
    
    def wrap_into_bounds_any(self, x):  # seed가 없을 때: bounds 안에 들어오는 2pi shift가 있으면 그 중에서 0에 가장 가까운(Neutral) 값을 선택
        x = np.asarray(x, dtype=float).copy()
        out = x.copy()
        two_pi = 2.0 * np.pi

        for i, (lo, hi) in enumerate(self.joint_bounds):
            k0 = int(np.round((-x[i]) / two_pi))
            
            candidates = []
            for k in range(k0 - 2, k0 + 3):
                xi = x[i] + two_pi * k
                if lo <= xi <= hi:
                    candidates.append(xi)
            
            out[i] = min(candidates, key=lambda v: abs(v)) if candidates else x[i]
                
        return out
    
    def in_bounds(self, q):  # check solution in bound
        for i, (lo, hi) in enumerate(self.joint_bounds):
            if q[i] < lo or q[i] > hi:
                return False
        return True
    
    def inverse_kinematics_np_push(self, T, seed=None):
        T = np.array(T)
        if seed is not None:
            seed = np.asarray(seed, dtype=float).reshape(-1)

        # Obtaining theta1~theta3 with the consine law
        l_rod = self.M[0, 3] - self.joint_positions[4][0] # tool length
        joint5_pos = T[:3, 3] - l_rod * T[:3, 0]    # joint4 pos = joint5 pos = joint6 pos
        
        theta_1 = np.arctan2(joint5_pos[1], joint5_pos[0])
        dist = np.sqrt(joint5_pos[0] * joint5_pos[0] + joint5_pos[1] * joint5_pos[1])
        joint5_pos = np.array([dist, 0.0, joint5_pos[2]]) # after theta1 rotation
        
        joint2_zeropos = self.joint_positions[1]
        joint3_zeropos = self.joint_positions[2]
        joint5_zeropos = self.joint_positions[4]
        
        L1 = np.linalg.norm(joint3_zeropos - joint2_zeropos)
        L2 = np.linalg.norm(joint5_zeropos - joint3_zeropos)
        alpha_0 = np.arctan2(joint5_zeropos[2] - joint3_zeropos[2], joint5_zeropos[0] - joint3_zeropos[0])
        
        L = np.linalg.norm(joint5_pos - joint2_zeropos)
        if L > L1 + L2 or L < np.abs(L1 - L2):
            print("encoutered invalid SE(3)")
            return None
        alpha_1 = np.arccos((L1 * L1 + L * L - L2 * L2) / (2 * L1 * L))
        alpha_2 = np.arccos((L1 * L1 + L2 * L2 - L * L) / (2 * L1 * L2))
        alpha_3 = np.arctan2(joint5_pos[2] - joint2_zeropos[2], joint5_pos[0] - joint2_zeropos[0])
        
        theta_2 = np.pi / 2 - (alpha_1 + alpha_3)
        theta_3 = (np.pi / 2 - alpha_0) - (np.pi - alpha_2)
        
        # Obtaining theta4~theta6 with the Euler angle equation
        R_03 = R.from_euler('yz', [theta_2 - theta_3, theta_1])
        R_euler = np.matmul(R_03.as_matrix().T, T[:3, :3]) # R30 R06 = R36
        # R_euler = np.matmul(R_03.as_dcm().T, T[:3, :3])    # For older versions of scipy
        
        theta_6, theta_5, theta_4 = R.from_matrix(R_euler).as_euler('xyx')
        # theta_6, theta_5, theta_4 = R.from_dcm(R_euler).as_euler('xyx')    # For older versions of scipy
        theta_4, theta_5, theta_6 = -theta_4, -theta_5, -theta_6
        
        first_soln = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
        
        if np.isnan(first_soln).any():
            return None
        
        # Selecting one of the two solutions of the Euler angle equation
        second_soln = first_soln.copy()
        if first_soln[3] > 0:
            second_soln[3] -= np.pi
        else:
            second_soln[3] += np.pi
        second_soln[4] = -first_soln[4]
        if first_soln[5] > 0:
            second_soln[5] -= np.pi
        else:
            second_soln[5] += np.pi
        
        # Wrap & Check valid
        if seed is not None:
            first_sel  = self.wrap_near_bounds(first_soln,  seed)
            second_sel = self.wrap_near_bounds(second_soln, seed)
        else:
            first_sel  = self.wrap_into_bounds_any(first_soln)
            second_sel = self.wrap_into_bounds_any(second_soln)

        is_valid1 = self.in_bounds(first_sel)
        is_valid2 = self.in_bounds(second_sel)

        if is_valid1 and is_valid2:
            if seed is not None:
                diff1 = np.linalg.norm(first_sel - seed)
                diff2 = np.linalg.norm(second_sel - seed)
                return first_sel if diff1 <= diff2 else second_sel
            else:
                return first_sel if abs(first_sel[3]) <= abs(second_sel[3]) else second_sel
        
        elif is_valid1:
            return first_sel
            
        elif is_valid2:
            return second_sel
            
        else:
            return None
