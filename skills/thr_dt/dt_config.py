"""Published conditions of:

    M. Monastirsky, O. Azulay and A. Sintov,
    "Learning to Throw With a Handful of Samples Using Decision Transformers,"
    IEEE Robotics and Automation Letters, vol. 8, no. 2, pp. 576-583, 2023.
    doi: 10.1109/LRA.2022.3229266  (IEEE Xplore arnumber 9984828)

Every constant below is annotated with its source:
  [paper SX.Y]  -- the RA-L paper / MSc thesis text (identical work, more detail)
  [repo <file>] -- the authors' official code, github.com/MaxorPaxor/ThrowBot

Where the paper text and the released code disagree, the paper value is the
default and the code value is noted in a comment.
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Task / environment
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    # --- control loop -----------------------------------------------------
    update_rate: float = 10.0        # Hz     [paper S5.1] communication frequency
    total_time: float = 1.0          # sec    [paper S5.1] max trajectory length
    #   => upper bound of T_i <= 10 timesteps per trajectory [paper S5.1]

    # --- joints -----------------------------------------------------------
    # Planar throw: only joints 2 (L), 3 (U), 5 (B) of the Motoman GP8 move;
    # remaining joints fixed at zero, throw direction set analytically by
    # joint 1 [paper S5.1].  Gripper is the 4th "joint".
    joint_names: tuple = ('joint_2_l', 'joint_3_u', 'joint_5_b', 'finger_joint')
    max_speed_deg: tuple = (385.0, 520.0, 550.0)  # deg/s  [repo env/robot_env_dt.py:40]
    max_speed_factor: float = 1.0                 # [repo env/robot_env_dt.py:47]
    smooth_factor: float = 0.0                    # complementary velocity filter
    #                                               [repo env/robot_env_dt.py:20] (disabled)

    # --- goal / reward ----------------------------------------------------
    dg_min: float = 0.5              # m  [paper S5.1] d_g in [50, 200] cm
    dg_max: float = 2.0              # m
    target_radius: float = 0.1       # m (rho)  [repo env/robot_env_dt.py:31]
    # NOTE: thesis S5.1 collected data with rho = 0.5 cm and reports that
    # rho in [0.5, 20] cm makes no significant difference (S5.3).  With the
    # default K_her = 0 relabeling, rho does not affect training labels at all.

    # --- gripper ----------------------------------------------------------
    # The gripper opens when action a_gr <= tau; tau is the mean of all
    # gripper actions in the training set [paper S5.1].  0.8385 is the
    # authors' default before a dataset is available [repo env/robot_env_dt.py:48].
    gripper_thresh: float = 0.8385

    # --- planar kinematics (x-z plane) ------------------------------------
    # The paper models the GP8 in Gazebo with *arbitrary* dynamic parameters
    # "while in the same scale of the real one" [paper S5.1].  We therefore
    # use a self-contained planar model with GP8-scale link lengths.  The
    # angle convention was calibrated against two poses of the official code:
    #   home pose (0.5, -0.3, -1.5)      -> tip ~ (0.26, 0.22); object spawn
    #                                       in Gazebo is (0.28, 0.20)
    #   release pose (0.9, 0.7, -0.586)  -> tip ~ (0.76, 0.32); the authors'
    #                                       ballistic model uses (0.813, 0.349)
    shoulder_height: float = 0.330   # m, base to joint-2 axis (GP8 scale)
    l_upper: float = 0.345           # m, joint-2 -> joint-3
    l_fore: float = 0.340            # m, joint-3 -> joint-5
    l_tool: float = 0.220            # m, joint-5 -> object in gripper

    home_pose: tuple = (0.5, -0.3, -1.5)  # rad  [repo env/robot_env_dt.py:112]

    # --- integration / actuator dynamics ----------------------------------
    substeps: int = 10               # physics substeps per control step
    tau_act: float = 0.05            # sec, 1st-order actuator lag ("PID" speed).
    #   Arbitrary by design [paper S5.1]; this is the knob the paper's
    #   simulation-tuning step (Bayesian optimization, S4.4) would optimize.
    gravity: float = 9.81

    # --- object -----------------------------------------------------------
    # cube 1.5x1.5x1.5 cm, 15 g [paper S5.1]
    object_half_diag: float = 0.5 * 0.015 * np.sqrt(2)  # ground-contact height
    #                                    [repo env/robot_env_dt.py:347,406]

    @property
    def dt(self) -> float:
        return 1.0 / self.update_rate

    @property
    def number_steps(self) -> int:
        return int(self.total_time * self.update_rate)  # = 10

    @property
    def max_speed_rad(self) -> np.ndarray:
        return np.asarray(self.max_speed_deg) * np.pi / 180.0


# A stand-in for the physical robot used by the sim2real fine-tuning demo:
# same task, perturbed (unknown-to-the-model) dynamics = reality gap.
def real_robot_config() -> SimConfig:
    cfg = SimConfig()
    cfg.tau_act = 0.14               # much laggier actuators
    cfg.max_speed_factor = 0.85      # weaker than the ideal model
    cfg.l_tool = 0.24                # slightly different tool geometry
    return cfg


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    n_trajectories: int = 500        # [paper S5.3] performance saturates at 500;
    #                                  Table 5.1 DT row uses 500 random throws
    k_her: int = 0                   # [paper S5.3] K_her > 0 gives no significant
    #                                  improvement; K_her = 0 relabels the goal
    #                                  with the landing spot
    her_radius_factor: float = 2.0   # HER goals drawn in a circle of radius
    #                                  2*rho around x_land [paper S4.2]
    #                                  [repo agent/agent_dt.py:21]

    # Ornstein-Uhlenbeck exploration noise, eq. (4.3):
    #   x_{k+1} = x_k + (mu - x_k) * theta * dt + sigma * sqrt(dt) * N(0,1)
    ou_theta: float = 0.02           # [paper S5.1]
    ou_mu: float = 0.0               # [paper S5.1]
    ou_sigma: float = 0.2            # [paper S5.1]
    ou_dt: float = 0.1               # "dt is the sample time" [paper S4.2] = 1/10 Hz
    # NOTE: the released code used a stronger process instead
    # (theta=0, sigma=0.6, dt=0.2) [repo agent/agent_dt.py:45]; set
    # --ou-preset repo in collect_data.py to reproduce that variant.


# ---------------------------------------------------------------------------
# Decision Transformer model + training
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    state_dim: int = 5               # (theta_2, theta_3, theta_5, theta_gr || d_g)
    #                                  [paper S4.3 eq. 4.7, repo train_dt_offline.py:23]
    act_dim: int = 4                 # (w_2, w_3, w_5, a_gr) [paper S4.1]
    K: int = 20                      # context window fed to the model
    #                                  [repo train_dt_offline.py:25]; episodes are
    #                                  <= 10 steps so the DT always sees the full
    #                                  trajectory ("access all previous steps",
    #                                  paper S5.1)
    max_ep_len: int = 64             # timestep-embedding table size
    #                                  [repo train_dt_offline.py:148]
    embed_dim: int = 128             # [paper S5.2]
    n_layer: int = 1                 # [paper S5.2]
    n_head: int = 1                  # [paper S5.2]
    activation: str = 'relu'         # [paper S5.2]
    dropout: float = 0.1             # [paper S5.2]
    n_positions: int = 1024          # [repo train_dt_offline.py:147]
    # With these values the model has exactly 210,058 trainable parameters,
    # matching the count reported in the paper [paper S5.2].


@dataclass
class TrainConfig:
    batch_size: int = 128            # trajectories per mini-batch [paper S5.2]
    epochs: int = 100                # [paper S5.2] "100 epochs"
    steps_per_epoch: int = 100       # [paper S5.2] "each epoch consists of 100
    #                                  optimization steps"
    lr: float = 1e-4                 # [paper S5.2] (repo code used 3e-4)
    weight_decay: float = 1e-4       # [paper S5.2]
    warmup_steps: int = 10_000       # [paper S5.2] linear warm-up over the first
    #                                  10^4 gradient steps (i.e. whole training)
    grad_clip: float = 0.25          # [repo agent/trainer.py:73]
    bce_weight: float = 0.5          # loss = MSE(motors) + 0.5*BCE(gripper)
    #                                  [repo agent/trainer.py:39]; paper S5.2 says
    #                                  "sum of MSE and BCE" without the factor
    timestep_noise: int = 3          # random timestep translation in [0, 2]
    #                                  [paper S5.2, repo train_dt_offline.py:116]
    eval_throws: int = 20            # goals evenly distributed in range, each
    #                                  epoch; snapshot best model [paper S5.2]
    eval_warmup_epochs: int = 20     # repo trains 2000 "warm-up" steps before
    #                                  its evaluated iterations
    #                                  [repo train_dt_offline.py:226]
    target_return: float = 1.0       # RTG conditioning at inference
    #                                  [repo evaluate_dt.py:43]


@dataclass
class FinetuneConfig:
    n_real_throws: int = 5           # "a handful (~5) of real throws" [abstract]
    alpha_max: float = 3.0           # action gain alpha ~ U(1, alpha_max) while
    #                                  collecting real throws [paper S5.4]
    iters: int = 10                  # [repo train_dt_offline.py:189]
    steps_per_iter: int = 100        # [repo train_dt_offline.py:190]
    warmup_steps: int = 5_000        # [repo train_dt_offline.py:191]
