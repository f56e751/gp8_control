"""Published conditions of:

    A. Zeng, S. Song, J. Lee, A. Rodriguez and T. Funkhouser,
    "TossingBot: Learning to Throw Arbitrary Objects with Residual Physics,"
    IEEE Transactions on Robotics (T-RO), 2020.
    arXiv:1903.11239v3

This package implements the **Physics-only** baseline of the paper
[paper SVI-B]:

    "Physics-only is also a variant of our approach where the throwing
     network is removed and completely replaced by velocity predictions
     made by the physics-based controller. In other words, this variant
     only learns grasping and uses physics for throwing (without
     learning a residual)."

Every constant below is annotated with its source:
  [paper SX.Y / Eq.N / Fig.N] -- the T-RO paper text
  [not in paper]              -- required for a runnable simulation but not
                                 specified by the paper; value chosen and
                                 documented here.

The simulated environment follows the paper's simulation setup [paper SVI-A]
(PyBullet, 4 seen + 4 unseen object types, 12 target boxes outside the
UR5's reach).  Known deviations from the paper are listed in README.md
(most notably: a free-floating parallel-jaw gripper executes the motion
primitives instead of a full UR5 arm; the paper's primitives are otherwise
reproduced verbatim -- top-down grasp, 10 cm lift, release at position r
with velocity v).
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Workspace / heightmap                                          [paper SIII-A]
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkspaceConfig:
    # "this area covers a 0.9 x 0.7 m tabletop surface" [paper SIII-A]
    # 0.9 m is lateral (y, image width 180 px), 0.7 m is depth (x, height 140).
    extent_x: float = 0.7            # m      [paper SIII-A]
    extent_y: float = 0.9            # m      [paper SIII-A]
    # Placement of the workspace in the robot frame (base at origin).
    # [not in paper] -- must keep every point within the UR5's 0.85 m reach.
    x_min: float = 0.15              # m      [not in paper]
    y_min: float = -0.45             # m      [not in paper] centered on x-axis

    # "heightmaps have a pixel resolution of 180 x 140, hence each pixel i
    #  represents a 5x5 mm vertical column of 3D space" [paper SIII-A]
    pixels_w: int = 180              # px     [paper SIII-A] (y / lateral axis)
    pixels_h: int = 140              # px     [paper SIII-A] (x / depth axis)
    resolution: float = 0.005        # m/px   [paper SIII-A]

    # Bin walls around the workspace.  Fig. 6 shows a shallow gray tray.
    wall_height: float = 0.08        # m      [not in paper]
    wall_thickness: float = 0.01     # m      [not in paper]

    @property
    def x_max(self) -> float:
        return self.x_min + self.extent_x

    @property
    def y_max(self) -> float:
        return self.y_min + self.extent_y


# ---------------------------------------------------------------------------
# Physics-based ballistic controller                    [paper SIII-C, SIV, Eq.1-2]
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BallisticConfig:
    # "imparts a downward acceleration a_z = -9.8 m/s^2" [paper SIII-C]
    gravity: float = 9.8             # m/s^2  [paper SIII-C] (magnitude)

    # "we select constant values of c_h and c_d such that all release
    #  positions are accessible by the robot: c_h = 0.04 m and c_d = 0.7 m
    #  in simulation" [paper SIII-C]
    release_height_ch: float = 0.04  # m      [paper SIII-C] r_z = c_h
    release_dist_cd: float = 0.70    # m      [paper SIII-C] |r_xy| = c_d

    # "we further constrain the direction of v to be angled 45 deg upwards
    #  in the direction of p ... ||v_{x,y}|| = v_z" [paper SIII-C]
    release_angle_deg: float = 45.0  # deg    [paper SIII-C]

    # If True, use Eq. 2 exactly as printed in the paper.  As printed, Eq. 2
    # (a) measures the horizontal distance from the robot BASE (sqrt(px^2+py^2))
    #     instead of from the release position (that distance minus c_d), and
    # (b) carries a sign typo on the (r_z - p_z) term.
    # The exact closed-form solution of the constraint system the paper
    # defines ("p = r + v*t + 1/2*a*t^2" with the 45 deg and collinearity
    # constraints [paper SIII-C]) is the default; tests/test_physics.py
    # demonstrates numerically that the literal Eq. 2 overshoots the target.
    use_literal_eq2: bool = False    # [see README "Eq. 2 discrepancy"]


# ---------------------------------------------------------------------------
# Target boxes                                                    [paper SVI-A]
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoxConfig:
    # "12 boxes located outside a UR5 robot arm's maximum reach range"
    # [paper SVI-A]; UR5 max reach = 0.85 m (UR5 spec, [not in paper]).
    n_boxes: int = 12                # [paper SVI-A]
    # "Each box is 20 cm tall with a 25 x 15 cm opening" [paper SVI-A]
    height: float = 0.20             # m      [paper SVI-A]
    opening_lateral: float = 0.25    # m      [paper SVI-A] 25 cm
    opening_throw: float = 0.15      # m      [paper SVI-A] 15 cm, along the
                                     #        throw direction (Fig. 1 layout)
    wall_thickness: float = 0.01     # m      [not in paper]

    # Layout [Fig. 1/6]: the bin sits in FRONT of the robot (+x) and the
    # box grid sits to the SIDE (+y), 3 rows (y, throw direction) x 4
    # columns (x).  This matches Fig. 6 and keeps the low-altitude release
    # run-up (r_z = c_h = 0.04 m) clear of the bin.       [Fig. 6; layout
    # coordinates not in paper]
    rows: int = 3                    # [Fig. 1/6]
    cols: int = 4                    # [Fig. 1/6]
    # Nearest box edge at y = 0.92 m > 0.85 m UR5 reach.  [not in paper]
    first_row_y: float = 1.00        # m      [not in paper]
    row_pitch: float = 0.16          # m      = opening_throw + wall
    col_pitch: float = 0.26          # m      = opening_lateral + wall

    # The robot + bin stand on a pedestal and the boxes stand on the ground
    # below, as in the paper's real setup (Fig. 1: arm on a pedestal, the
    # landing zone below/adjacent).  This is geometrically REQUIRED for the
    # paper's ballistic constants: with release height c_h = 0.04 m, a
    # 45 deg throw reaches near boxes whose 0.20 m-tall rims are ABOVE the
    # release point while still ascending and would strike the box's outer
    # wall.  With the box openings 0.10 m BELOW the release plane every
    # target is reached on descent.                        [not in paper;
    # see README "Pedestal geometry"]
    ground_offset: float = 0.30      # m      [not in paper]

    def centers(self) -> np.ndarray:
        """(12, 3) box-opening centers p = middle of the top opening, in
        the robot frame (z=0 at pedestal top / bin floor).
        "The middle of the top opening of each box is used as the input
         target landing position p" [paper SVI-A]."""
        ys = self.first_row_y + self.row_pitch * np.arange(self.rows)
        xs = (np.arange(self.cols) - (self.cols - 1) / 2.0) * self.col_pitch
        p_z = self.height - self.ground_offset
        centers = [(x, y, p_z) for y in ys for x in xs]
        return np.array(centers)


# ---------------------------------------------------------------------------
# Objects                                                  [paper SVI-A, Fig.5]
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObjectConfig:
    # "We use 8 different objects: 4 seen during training and 4 unseen for
    #  testing." [paper SVI-A]
    # Seen: "4cm-diameter ball, 4x4x4 cm cube, 3cm-diameter 16cm-long rod,
    #  and a 16cm-long hammer (union of 2cm-diameter 12cm-long rod with
    #  10x4x2.5 cm block)" [paper SVI-A]
    ball_diameter: float = 0.04              # m  [paper SVI-A]
    cube_size: float = 0.04                  # m  [paper SVI-A]
    rod_diameter: float = 0.03               # m  [paper SVI-A]
    rod_length: float = 0.16                 # m  [paper SVI-A]
    hammer_rod_diameter: float = 0.02        # m  [paper SVI-A]
    hammer_rod_length: float = 0.12          # m  [paper SVI-A]
    hammer_head: tuple = (0.10, 0.04, 0.025) # m  [paper SVI-A] 10x4x2.5 cm

    # Unseen simulation objects, approximated from silhouettes in
    # Fig. 5 (top-right) / Fig. 10: L-shape (elbow), cross (+), I-beam
    # (rod with a block on both ends), ball.  Dimensions [not in paper].
    unseen_L_arm_a: float = 0.12             # m  [not in paper]
    unseen_L_arm_b: float = 0.08             # m  [not in paper]
    unseen_arm_diameter: float = 0.02        # m  [not in paper]
    unseen_cross_length: float = 0.12        # m  [not in paper]
    unseen_ibeam_rod: float = 0.10           # m  [not in paper]
    unseen_ibeam_block: tuple = (0.06, 0.04, 0.025)  # m [not in paper]
    unseen_ball_diameter: float = 0.05       # m  [not in paper]

    # "Multiple copies of each object (12 in total) are randomly colored and
    #  dropped into the bin during training and testing." [paper SVI-A]
    n_objects: int = 12                      # [paper SVI-A/SVI] n = 12 in sim

    # Physical material parameters [not in paper]: wooden-toy density
    # ~600 kg/m^3; PyBullet friction defaults tuned for stable grasps.
    density: float = 600.0                   # kg/m^3 [not in paper]
    lateral_friction: float = 0.8            # [not in paper]
    spinning_friction: float = 0.02          # [not in paper]
    restitution: float = 0.2                 # [not in paper]


# ---------------------------------------------------------------------------
# Motion primitives                                             [paper SIII-B/C]
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrimitiveConfig:
    # Grasping primitive [paper SIII-B]: "executes a top-down parallel-jaw
    # grasp centered at a 3D location x oriented theta around the gravity
    # direction ... approaches x along the gravity direction until the 3D
    # position of the middle point between the gripper fingertips meets x,
    # at which point the gripper closes, and lifts upwards 10 cm."
    n_rotations: int = 16            # [paper SIII-B] "16 orientations
                                     #  (multiples of 22.5 deg)"
    lift_height: float = 0.10        # m  [paper SIII-B] "lifts upwards 10cm"

    # "checked by thresholding the distance between fingertips"
    # [paper SIII-C]; threshold value [not in paper].
    grasp_width_threshold: float = 0.005   # m  [not in paper]

    # Fingertip engagement depth below the heightmap surface value.  The
    # paper drives the fingertip midpoint exactly to the pixel's height;
    # a small extra descent makes finger-pad contact reliable in PyBullet.
    engage_depth: float = 0.02       # m  [not in paper]
    min_grasp_z: float = 0.01        # m  [not in paper] floor clearance

    # Floating-gripper geometry/actuation [not in paper] (stands in for the
    # UR5 + RG2; RG2 spec: 110 mm stroke, 40 N grip force).
    finger_max_open: float = 0.10    # m  [RG2 spec, not in paper]
    finger_force: float = 60.0       # N  [not in paper]
    finger_length: float = 0.05      # m  [not in paper]

    # Throwing primitive [paper SIII-C]: linear acceleration along the
    # 45 deg release direction, then a short constant-velocity cruise so
    # the held object's velocity converges to the commanded release
    # velocity (the contact solver tracks a changing velocity with a few-
    # timestep lag), releasing exactly at r.  Ramp lengths [not in paper].
    # The run-up start (r - v_dir*(d_acc+d_cruise), z ~= -0.10 m) lies in
    # the open gap beyond the pedestal edge.
    throw_accel_dist: float = 0.15   # m  [not in paper] acceleration ramp
    throw_cruise_dist: float = 0.05  # m  [not in paper] const-v before r
    # Follow-through long enough (~90 ms at release speed) for the freed
    # object to fall clear of the finger cage before the gripper stops --
    # stopping earlier slams the palm into the still-level projectile.
    throw_follow_dist: float = 0.22  # m  [not in paper] follow-through
    sim_timestep: float = 1.0 / 240  # s  [PyBullet default, not in paper]
    settle_timeout: float = 3.0      # s  [not in paper] wait for landing


# ---------------------------------------------------------------------------
# Network / training                                          [paper SIII, SV]
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    # Perception: "C(3,64)-MP-RB(128)-MP-RB(256)-RB(512)" [paper SV]
    # Grasping:   "RB(256)-RB(128)-UP-RB(64)-UP-C(1,2)"   [paper SV]
    # (The throwing module "RB(256)-RB(128)-UP-RB(64)-UP-C(1,1)" is REMOVED
    #  in the Physics-only variant [paper SVI-B].)
    # Input I: RGB-D heightmap; "RGB and D channels are normalized (mean-
    # subtracted and divided by standard deviation from a pre-recorded
    # dataset of 100 images)" [paper SIII-A].
    in_channels: int = 4             # RGB + D  [paper SIII-A; packing not
                                     #  specified -- see README]
    n_norm_images: int = 100         # [paper SIII-A]

    # "concatenating the visual feature representation mu with a k-channel
    #  image (k = 128) where each pixel holds the value of v_hat" [paper SIV]
    v_channels: int = 128            # [paper SIV]

    # "trained end-to-end ... stochastic gradient descent with momentum,
    #  using fixed learning rates of 1e-4, momentum of 0.9, and weight
    #  decay 2e-5" -- NOTE the paper typesets "2^-5" [paper SV].
    lr: float = 1e-4                 # [paper SV]
    momentum: float = 0.9            # [paper SV]
    weight_decay: float = 2 ** -5    # = 0.03125 [paper SV]

    # "Our models are trained from scratch (i.e. random Xavier
    #  initialization)" [paper SV]
    # -> models.py applies Xavier init to every conv layer.

    # "we train our models via trial and error for 15,000 steps, then test
    #  each model for 1,000 steps" [paper SVI-C]
    train_steps: int = 15000         # [paper SVI-C]
    test_steps: int = 1000           # [paper SVI-C]

    # "exploration strategy is epsilon-greedy, with epsilon initialized at
    #  0.5 then annealed over training to 0.1" [paper SV]
    epsilon_init: float = 0.5        # [paper SV]
    epsilon_final: float = 0.1       # [paper SV]
    # Annealing schedule shape [not in paper]: linear over all train steps.

    # "prioritized experience replay [32] using stochastic rank-based
    #  prioritization, approximated with a power-law distribution" [paper SV].
    # Implemented as in the authors' released VPG code (surprise-sorted
    # buffer sampled with np.random.power(replay_power)).
    replay_power: float = 2.0        # [not in paper; VPG practice]
    # Gradient iterations per executed action [not in paper].  The real
    # system trains asynchronously "while robot.is_grasping/is_throwing"
    # [paper Alg.1] (an action takes seconds -> many iterations at their
    # 220 ms inference/train cadence); simulation needs an explicit count.
    train_iters_per_step: int = 8    # [not in paper]
    batch_size: int = 1              # [paper SV] "gradients only through the
                                     #  single pixel i" -- per-sample updates

    device: str = "cuda"
    seed: int = 0                    # [not in paper]
    log_every: int = 50
    ckpt_every: int = 1000


# ---------------------------------------------------------------------------
WORKSPACE = WorkspaceConfig()
BALLISTIC = BallisticConfig()
BOXES = BoxConfig()
OBJECTS = ObjectConfig()
PRIMITIVE = PrimitiveConfig()
TRAIN = TrainConfig()
