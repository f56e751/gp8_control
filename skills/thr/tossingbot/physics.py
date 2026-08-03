"""Physics-based ballistic controller.                [paper SIII-C, SIV, Eq.1-2]

"The physics-based controller uses the standard equations of linear
 projectile motion, by assuming a grasp on the center of mass of the
 object, to analytically solve back for the release velocity v_hat given
 the target landing location p and release position r of the throwing
 primitive: p = r + v*t + 1/2*a*t^2."                        [paper SIII-C]

Constraints [paper SIII-C]:
  (i)   the aerial trajectory is linear on the xy-plane and in the same
        direction as v_xy:  (r_xy - p_xy) x v_xy = 0
  (ii)  sqrt(r_x^2 + r_y^2) = c_d  and  r_z = c_h
  (iii) v is angled 45 deg upwards in the direction of p:  ||v_xy|| = v_z

Under (i)-(iii) the only unknown is ||v_xy||               [paper SIV].

NOTE on Eq. 1 as printed: "r_x = c_d*sin(theta), r_y = c_d*cos(theta)"
with theta = arctan(p_y / p_x).  Taken literally this places r rotated
away from p, contradicting collinearity constraint (i).  We implement the
collinear placement r_xy = c_d * p_xy / ||p_xy|| (equivalently
r_x = c_d*cos(theta), r_y = c_d*sin(theta)), which is what (i) requires.

NOTE on Eq. 2 as printed: see BallisticConfig.use_literal_eq2 and
tests/test_physics.py -- the literal formula measures the horizontal
distance from the robot base instead of from the release position and
flips the sign of (r_z - p_z); the exact solution below is used by
default and is verified against simulated projectiles.
"""

from __future__ import annotations

import numpy as np

from .config import BALLISTIC, BallisticConfig


class BallisticController:
    """Maps a target landing location p -> (release position r,
    release velocity v_hat, scalar speed ||v_xy||)."""

    def __init__(self, cfg: BallisticConfig = BALLISTIC):
        self.cfg = cfg

    # -- Eq. 1 ------------------------------------------------------------
    def release_position(self, p: np.ndarray) -> np.ndarray:
        """Release position r on the circle of radius c_d at height c_h,
        collinear with p in the xy-plane [paper SIII-C, Eq.1]."""
        p = np.asarray(p, dtype=np.float64)
        d_xy = np.linalg.norm(p[:2])
        if d_xy < 1e-9:
            raise ValueError("target directly above robot base")
        r_xy = self.cfg.release_dist_cd * p[:2] / d_xy
        return np.array([r_xy[0], r_xy[1], self.cfg.release_height_ch])

    # -- Eq. 2 ------------------------------------------------------------
    def speed(self, p: np.ndarray) -> float:
        """||v|| (full 3D speed) that lands the projectile on p."""
        p = np.asarray(p, dtype=np.float64)
        g = self.cfg.gravity
        r = self.release_position(p)
        if self.cfg.use_literal_eq2:
            # Eq. 2 exactly as printed, with a = -9.8 m/s^2 [paper SIII-C]:
            #   ||v|| = sqrt( a*(p_x^2+p_y^2) /
            #                 (r_z - p_z - sqrt(p_x^2+p_y^2)) )
            d2 = p[0] ** 2 + p[1] ** 2
            denom = r[2] - p[2] - np.sqrt(d2)
            return float(np.sqrt((-g) * d2 / denom))
        # Exact solution of p = r + v*t + 1/2*a*t^2 under the 45 deg
        # constraint.  With D = horizontal release->target distance and
        # v_z = ||v_xy|| = v_h:
        #   t = D / v_h ;  p_z - r_z = D - g*D^2 / (2*v_h^2)
        #   => v_h^2 = g*D^2 / (2*(D + r_z - p_z)),  ||v||^2 = 2*v_h^2
        D = np.linalg.norm(p[:2] - r[:2])
        denom = D + r[2] - p[2]
        if denom <= 0:
            raise ValueError(f"target not reachable at 45 deg: p={p}")
        return float(np.sqrt(g * D ** 2 / denom))

    def release(self, p: np.ndarray):
        """Full controller output for target p.

        Returns (r, v_hat, v_xy_mag):
          r        (3,) release position                       [Eq. 1]
          v_hat    (3,) release velocity vector, 45 deg up     [SIII-C]
          v_xy_mag scalar ||v_hat_{x,y}|| -- the quantity the throwing
                   module would predict residuals on top of    [paper SIV];
                   in Physics-only it is used as-is and fed to the
                   grasping network as conditioning input.
        """
        p = np.asarray(p, dtype=np.float64)
        r = self.release_position(p)
        v = self.speed(p)                       # ||v||
        v_h = v / np.sqrt(2.0)                  # ||v_xy|| = v_z  [SIII-C]
        dir_xy = (p[:2] - r[:2])
        dir_xy = dir_xy / np.linalg.norm(dir_xy)
        v_hat = np.array([v_h * dir_xy[0], v_h * dir_xy[1], v_h])
        return r, v_hat, float(v_h)

    # -- utilities ---------------------------------------------------------
    def flight_time(self, p: np.ndarray) -> float:
        r = self.release_position(p)
        _, v_hat, v_h = self.release(p)
        return float(np.linalg.norm(p[:2] - r[:2]) / v_h)

    def landing_xy(self, r: np.ndarray, v: np.ndarray, z_land: float):
        """Analytic landing point of a point mass released at r with
        velocity v when it crosses height z_land on the way down."""
        g = self.cfg.gravity
        a, b, c = -0.5 * g, v[2], r[2] - z_land
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None
        t = (-b - np.sqrt(disc)) / (2 * a)      # later (descending) root
        return np.array([r[0] + v[0] * t, r[1] + v[1] * t]), float(t)
