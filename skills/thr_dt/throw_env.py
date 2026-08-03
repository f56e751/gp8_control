"""Planar throwing environment.

Re-implements the task of the ROS-Gazebo environment used by the paper
(official code: ThrowBot/src/scripts/decision_transformer/env/robot_env_dt.py)
as a self-contained planar simulator.  The paper states that the simulated
dynamic parameters "were chosen arbitrary while in the same scale of the real
one" (S5.1) and that the DT is trained on a far-from-reality simulation, so a
lightweight physics model is consistent with the published setup.  The public
API (reset / update_target / get_state / step and their return values) mirrors
the official RoboticArm class so that the data-collection, training and
evaluation code can follow the original scripts line by line.

Kinematics (x-z vertical plane, x = throw direction):
    shoulder S = (0, h)
    elbow    E = S + l2 * (sin t2, cos t2)                  # t2 from vertical
    wrist    W = E + l3 * (cos a3, sin a3),  a3 = t3 - t2   # GP8-style U axis
    tip      P = W + l5 * (cos a5, sin a5),  a5 = a3 + t5
"""

import numpy as np

from .dt_config import SimConfig   # vendored: was `from config import SimConfig`


class RoboticArm:
    """Planar 3-joint (+gripper) throwing arm; API mirrors the official env."""

    def __init__(self, cfg: SimConfig = None, rng: np.random.Generator = None):
        self.cfg = cfg if cfg is not None else SimConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        # Global params  [repo env/robot_env_dt.py:16-20]
        self.UPDATE_RATE = self.cfg.update_rate
        self.total_time = self.cfg.total_time
        self.number_steps = self.cfg.number_steps
        self.no_rotation = True
        self.smooth_factor = self.cfg.smooth_factor
        self.number_states = 1               # states kept in the state memory

        # HER attributes  [repo env/robot_env_dt.py:29-31]
        self.her = True
        self.target_radius = self.cfg.target_radius

        self.joints = np.array(self.cfg.joint_names)
        self.max_speed = np.asarray(self.cfg.max_speed_deg + (1.0,))  # deg/s (+gripper)
        self.max_speed_factor = self.cfg.max_speed_factor
        self.gripper_thresh = self.cfg.gripper_thresh

        self.target = np.array([1.0, 0.0, 0.0])
        self.reset()
        self.initial_pos = self.object_position.copy()

    # ------------------------------------------------------------------ FK
    def _fk(self, q):
        """Positions of elbow, wrist and tip for joint angles q = (t2, t3, t5)."""
        c = self.cfg
        t2, t3, t5 = q
        a3 = t3 - t2
        a5 = a3 + t5
        S = np.array([0.0, c.shoulder_height])
        E = S + c.l_upper * np.array([np.sin(t2), np.cos(t2)])
        W = E + c.l_fore * np.array([np.cos(a3), np.sin(a3)])
        P = W + c.l_tool * np.array([np.cos(a5), np.sin(a5)])
        return E, W, P

    def _tip_velocity(self, q, qdot):
        """Analytic Jacobian * joint velocities -> tip velocity (vx, vz)."""
        c = self.cfg
        t2, t3, t5 = q
        a3 = t3 - t2
        a5 = a3 + t5
        d_t2 = (c.l_upper * np.array([np.cos(t2), -np.sin(t2)])
                + c.l_fore * np.array([np.sin(a3), -np.cos(a3)])
                + c.l_tool * np.array([np.sin(a5), -np.cos(a5)]))
        d_t3 = (c.l_fore * np.array([-np.sin(a3), np.cos(a3)])
                + c.l_tool * np.array([-np.sin(a5), np.cos(a5)]))
        d_t5 = c.l_tool * np.array([-np.sin(a5), np.cos(a5)])
        return d_t2 * qdot[0] + d_t3 * qdot[1] + d_t5 * qdot[2]

    # --------------------------------------------------------------- reset
    def reset(self):
        """Move the arm to its home pose and re-grasp the object.
        [repo env/robot_env_dt.py:90-110]"""
        self.q = np.array(self.cfg.home_pose, dtype=np.float64)
        self.qdot = np.zeros(3)
        self.gripper_closed = True
        self.velocity = np.zeros(4)          # last commanded (for smoothing)
        self.curr_step = 0
        self.curr_time = 0.0
        _, _, tip = self._fk(self.q)
        self.object_position = np.array([tip[0], 0.0, tip[1]])
        self.object_height = tip[1]

    # -------------------------------------------------------------- target
    def update_target(self, target):
        """[repo env/robot_env_dt.py:252-267]"""
        self.target = np.asarray(target, dtype=np.float64)

    # --------------------------------------------------------------- state
    def get_state(self):
        """(theta_2, theta_3, theta_5, theta_gr); theta_gr binary as in the
        paper's state definition S4.1 (1 = closed, 0 = open)."""
        return np.array([self.q[0], self.q[1], self.q[2],
                         1.0 if self.gripper_closed else 0.0])

    # -------------------------------------------------------------- reward
    def reward_sparse(self, obj_pos=None, target=None):
        """Sparse reward, eq. (4.2).  Success also requires the object to
        land beyond its initial position [repo env/robot_env_dt.py:303-320]."""
        if target is None:
            target = self.target
        if obj_pos is None:
            obj_pos = self.object_position
        distance = np.sqrt((obj_pos[0] - target[0]) ** 2 +
                           (obj_pos[1] - target[1]) ** 2)
        if distance <= self.target_radius and obj_pos[0] > self.initial_pos[0]:
            return 1.0
        return -1.0

    # ------------------------------------------------------------- helpers
    def smooth_velocity(self, new_velocity):
        """Complementary filter [repo env/robot_env_dt.py:322-333]; the
        published configuration uses smooth_factor = 0 (pass-through)."""
        old = np.asarray(self.velocity)
        sm = old * self.smooth_factor + new_velocity * (1 - self.smooth_factor)
        sm[-1] = new_velocity[-1]
        return sm

    def proj_on_max_speed(self, velocity_vector):
        """[-1,1] action -> rad/s command; gripper element untouched.
        [repo env/robot_env_dt.py:335-343]"""
        v = velocity_vector * self.max_speed * self.max_speed_factor
        v = v * np.pi / 180.0
        v[-1] = velocity_vector[-1]
        return v

    def _integrate_control_step(self, qdot_cmd):
        """Integrate one 1/f control period.  Position command follows the
        paper's Taylor approximation theta_t = theta_{t-1} + w_t / f
        (eq. 5.1); joints track it with a first-order actuator lag whose
        time constant stands in for the arbitrary PID gains of the
        simulated arm (S4.4, S5.1)."""
        c = self.cfg
        h = c.dt / c.substeps
        q_cmd = self.q + qdot_cmd * c.dt
        for _ in range(c.substeps):
            qdot_des = (q_cmd - self.q) / c.tau_act
            qdot_des = np.clip(qdot_des, -self.max_speed_rad_arm, self.max_speed_rad_arm)
            self.qdot = qdot_des
            self.q = self.q + self.qdot * h
        _, _, tip = self._fk(self.q)
        return tip

    @property
    def max_speed_rad_arm(self):
        return self.cfg.max_speed_rad * self.max_speed_factor

    def _ballistic_landing(self, pos, vel):
        """First landing point of a projectile released at `pos` with
        velocity `vel` (paper S4.1: x_land is the first landing point)."""
        g = self.cfg.gravity
        z0 = pos[1] - self.cfg.object_half_diag
        if z0 <= 0.0:
            return pos[0]
        vz = vel[1]
        t_land = (vz + np.sqrt(vz ** 2 + 2.0 * g * z0)) / g
        return pos[0] + vel[0] * t_land

    # ----------------------------------------------------------------- step
    def step(self, velocity_vector):
        """One 10 Hz control step; mirrors [repo env/robot_env_dt.py:352-420].

        Returns (reward, done, termination_reason, obj_pos, success).
        """
        velocity_vector = np.asarray(velocity_vector, dtype=np.float64)
        velocity_vector = self.proj_on_max_speed(velocity_vector)
        velocity_vector = self.smooth_velocity(velocity_vector)
        self.velocity = velocity_vector

        qdot_cmd = velocity_vector[:3]
        gripper = velocity_vector[-1]

        tip = self._integrate_control_step(qdot_cmd)
        self.curr_time += self.cfg.dt
        self.curr_step += 1

        if self.gripper_closed:
            self.object_position = np.array([tip[0], 0.0, tip[1]])
            self.object_height = tip[1]

        # --- termination logic (same order as the official env) -----------
        if self.curr_step >= self.number_steps or gripper < self.gripper_thresh:
            done = True
            if gripper < self.gripper_thresh:
                # Release: object leaves the gripper with the instantaneous
                # end-effector velocity and flies ballistically.
                termination_reason = f"Gripper was opened with value: {gripper}"
                tip_vel = self._tip_velocity(self.q, self.qdot)
                x_land = self._ballistic_landing(tip, tip_vel)
                self.gripper_closed = False
                self.object_position = np.array(
                    [x_land, 0.0, self.cfg.object_half_diag])
                self.object_height = self.object_position[2]
                reward = self.reward_sparse()
                success = True
            else:
                termination_reason = f"Time is up: {self.curr_time}"
                reward = -1.0
                success = False
        else:
            if self.object_height <= self.cfg.object_half_diag:
                termination_reason = ("Object is too close to ground: "
                                      f"{self.object_height}")
                done = True
                reward = -1.0
                success = False
            else:
                termination_reason = None
                done = False
                reward = 0.0
                success = False

        return reward, done, termination_reason, self.object_position.copy(), success
