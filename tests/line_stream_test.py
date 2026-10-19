"""EE-line IK streaming test — validate the rule-based Cartesian control path.

Why: push's DESCENT (``move_through_via``) and CHAIN are per-joint trapezoids
in JOINT space, so the EE path between two poses is uncontrolled in Cartesian
space and can dip into the belt (z≈0). The push STROKE already avoids this by
sampling a Cartesian straight line, solving seeded IK per waypoint, and
streaming at 250 Hz. This test exercises exactly that stroke-style control on
an arbitrary line so it can be validated (RViz sim first, then hardware) and
its speed limits measured, BEFORE replacing descent/chain with it.

What it runs, per speed (default line: (0.4,0,0.1) -> (0.6,0,0.1)):

  1. joint-space move to a hover above the start (safe, high — at
     ``--speed-scale`` of the joint limits),
  2. EE-LINE vertical descent   hover -> start      (``--descent-speed``),
  3. EE-LINE stroke             start -> goal       (the measured segment),
  4. EE-LINE vertical ascent    goal  -> hover',
     all with the stroke-facing push orientation (constant ``--swing``).

Reported per segment: IK failures, max joint-velocity ratio vs M1 (>1.0 ==
Yaskawa alarm 4414 — the segment is time-stretched like the production
clamp), planned vs wall-clock duration (soft-RT overrun), PATH-matched
tracking from /joint_states (``path-dev`` = residual to the closest point on
the commanded path — the latency-free belt-safety number, ≈0 vs the echo
mock; ``lag`` = transport(+servo) delay), and the commanded EE z range.

Also printed once: the BELT-DIP EVIDENCE — the same descent/stroke endpoints
interpolated the CURRENT way (joint-space ``trajectory()`` at production
M1/M2), FK-sampled, reporting min EE z and max deviation from the straight
line. This quantifies the collision risk the EE-line path removes.

Run against the sim (RViz):

  ros2 launch gp8_control debug_robot.launch.py         # terminal 1 (HW)
  ros2 run gp8_control line_stream_test --sweep 0.5,1.0,1.5,2.0,2.5   # terminal 2

or with the venv python (NOTE ``:$PYTHONPATH`` — a bare assignment clobbers
the ROS paths and rclpy stops importing; ROS must be sourced):

  PYTHONPATH=$HOME/ros2_ws/src:$PYTHONPATH \\
    ~/ros2_ws/src/gp8_control/.venv/bin/python \\
    -m gp8_control.tests.line_stream_test --sweep 0.5,1.0,1.5,2.0,2.5

Run against hardware (adv4ncr stream stack up, pendant REMOTE, no alarms,
1 m clearance — same prereqs as tests/push_height_test.py). Start LOW:

  ... -m gp8_control.tests.line_stream_test --speed 0.3

⚠️ SIM ON THE ROBOT PC: if the real stack may be up, isolate the sim run
(`export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1` in BOTH terminals) — this
test publishes to /JointGroupPositionController/commands, which the REAL
controller executes if it is reachable on the same domain.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from gp8_control.config import Config
from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.robots.gp8 import GP8
from gp8_control.skills.push_skill import PushSkill
from gp8_control.trajectory.trajectory_primitive import trajectory

# Sampling rate (Hz) of the /joint_states tracker thread during a streamed
# segment. 200 Hz ≈ every other 4 ms stream tick — enough to bound the error.
TRACK_HZ = 200.0

# Max time-stretch passes when a segment exceeds joint velocity limits
# (mirrors PushSkill._clamp_stroke_velocity).
CLAMP_PASSES = 2


# =========================================================================
# Build: Cartesian line -> seeded-IK joint trajectory
# =========================================================================

def _arc_profile(dist: float, speed: float, accel: float, hz: float):
    """Trapezoidal arc-length profile s(t) for a line of length ``dist``.

    Ramps 0 -> v -> 0 at ``accel``; degrades to a triangular profile (peak
    v = sqrt(accel*dist) < speed) when the line is too short to reach cruise.
    A constant-speed line (the original build) has a velocity STEP at both
    ends that the controller's acceleration limiter cannot follow — while it
    smooths the step the joints lag unevenly and the EE sags below the line
    (observed: stroke path-dev 37->59 mm at 0.5->1.0 m/s while the ramped
    descent/ascent tracked at 1-2 mm).

    Returns ``(s(n+1,), ts(n+1,), v_peak)``.
    """
    accel = max(accel, 1e-6)
    v = min(speed, float(np.sqrt(accel * dist)))
    t_ramp = v / accel
    d_ramp = 0.5 * accel * t_ramp ** 2
    d_cruise = max(0.0, dist - 2.0 * d_ramp)
    T = 2.0 * t_ramp + d_cruise / max(v, 1e-9)
    n = max(2, int(round(T * hz)))
    ts = np.linspace(0.0, T, n + 1)
    s = np.empty_like(ts)
    for i, t in enumerate(ts):
        if t <= t_ramp:
            s[i] = 0.5 * accel * t * t
        elif t <= T - t_ramp:
            s[i] = d_ramp + v * (t - t_ramp)
        else:
            dt_end = T - t
            s[i] = dist - 0.5 * accel * dt_end * dt_end
    return s, ts, v


def build_ee_line(
    robot: GP8,
    p0: np.ndarray,
    p1: np.ndarray,
    R: np.ndarray,
    speed: float,
    hz: float,
    q_seed: np.ndarray,
    accel: float = 5.0,
):
    """Straight EE line p0 -> p1 at fixed orientation ``R``, trapezoid speed.

    Arc length follows :func:`_arc_profile` (ramp-cruise-ramp at ``accel``,
    cruising at ``speed``); each knot is solved with seeded IK (branch
    continuity — same scheme as PushSkill._build_push_stroke), and joint
    velocities come from central finite differences (boundaries at rest).

    Returns ``(traj(6,n), vel(6,n), ts(n,), info)``; ``info['ik_fail']`` is
    the failing step index or None, ``info['v_peak']`` the profile's actual
    peak TCP speed (< ``speed`` when the line is too short to reach cruise).
    On IK failure the trajectory is truncated at the last good waypoint.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    dist = float(np.linalg.norm(p1 - p0))
    s_arr, ts_arr, v_peak = _arc_profile(dist, max(speed, 1e-6), accel, hz)
    n_steps = len(s_arr) - 1
    dt = ts_arr[1] - ts_arr[0] if n_steps >= 1 else 1.0 / hz

    waypoints = []
    seed = np.asarray(q_seed, float)
    ik_fail = None
    for i in range(n_steps + 1):
        a = s_arr[i] / max(dist, 1e-9)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p0 + a * (p1 - p0)
        q = robot.inverse_kinematics(T, q_init=seed)
        if q is None:
            ik_fail = i
            break
        seed = np.asarray(q, float)
        waypoints.append(seed)

    if len(waypoints) < 2:
        raise RuntimeError(
            f"EE line degenerate: IK failed at step {ik_fail}/{n_steps} "
            f"(<2 waypoints) for {p0} -> {p1}"
        )

    n = len(waypoints)
    traj = np.column_stack(waypoints)
    ts = ts_arr[:n].copy()          # uniform grid from the arc profile
    vel = np.zeros_like(traj)
    if n > 2:
        vel[:, 1:-1] = (traj[:, 2:] - traj[:, :-2]) / (2.0 * dt)
    vel[:, 0] = 0.0
    vel[:, -1] = 0.0

    # Max inter-waypoint joint jump — a ~pi jump means an IK branch flip.
    max_jump = float(np.max(np.abs(np.diff(traj, axis=1)))) if n > 1 else 0.0
    info = {"ik_fail": ik_fail, "n": n, "max_jump": max_jump, "dist": dist,
            "v_peak": v_peak}
    return traj, vel, ts, info


def clamp_velocity(traj, vel, ts, m1):
    """Time-stretch the segment until no joint exceeds ``m1`` (4414 guard).

    Same uniform-stretch approach as PushSkill._clamp_stroke_velocity.
    Returns ``(traj, vel, ts, raw_ratio, stretch)`` — ``raw_ratio`` is the
    pre-clamp worst ratio, ``stretch`` the total time scale applied (1.0 =
    untouched).
    """
    m1 = np.asarray(m1, float)[: traj.shape[0]]
    raw_ratio = None
    stretch = 1.0
    for _ in range(CLAMP_PASSES):
        dt_seg = np.maximum(np.diff(ts), 1e-9)
        seg_vel = np.abs(np.diff(traj, axis=1)) / dt_seg[None, :]
        ratio = float(np.max(seg_vel / m1[:, None]))
        if raw_ratio is None:
            raw_ratio = ratio
        if ratio <= 1.0:
            break
        scale = ratio * 1.05
        ts = ts * scale
        stretch *= scale
        n = traj.shape[1]
        new_dt = ts[1] - ts[0] if n > 1 else 1e-9
        vel = np.zeros_like(traj)
        if n > 2:
            vel[:, 1:-1] = (traj[:, 2:] - traj[:, :-2]) / (2.0 * new_dt)
        vel[:, 0] = 0.0
        vel[:, -1] = 0.0
    return traj, vel, ts, raw_ratio, stretch


# =========================================================================
# Analysis helpers
# =========================================================================

def ee_z_range(robot: GP8, traj: np.ndarray) -> tuple[float, float]:
    """(min, max) commanded EE z across the trajectory knots."""
    zs = [float(robot.forward_kinematics(traj[:, i])[2, 3])
          for i in range(traj.shape[1])]
    return min(zs), max(zs)


def line_deviation(robot: GP8, traj: np.ndarray, p0, p1) -> float:
    """Max EE distance (m) from the p0->p1 straight line across knots."""
    p0 = np.asarray(p0, float)
    d = np.asarray(p1, float) - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return 0.0
    u = d / L
    worst = 0.0
    for i in range(traj.shape[1]):
        p = robot.forward_kinematics(traj[:, i])[:3, 3]
        r = p - p0
        along = float(np.clip(r @ u, 0.0, L))
        worst = max(worst, float(np.linalg.norm(r - along * u)))
    return worst


class Tracker:
    """Samples measured joints from /joint_states while a segment streams."""

    def __init__(self, ctrl: TrajectoryController) -> None:
        self._ctrl = ctrl
        self._samples: list[tuple[float, list]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        period = 1.0 / TRACK_HZ
        while not self._stop.is_set():
            q = self._ctrl.current_joints
            if q is not None:
                self._samples.append((time.time(), list(q)))
            time.sleep(period)

    def error_vs(self, robot: GP8, t0: float, ts, traj):
        """PATH-matched tracking metrics, separating lag from deviation.

        A wall-time comparison folds the whole pub/sub transport latency into
        the "error" (latency x joint speed reads as tens of degrees at stroke
        speeds, even against the ideal echo mock). Instead each measured
        sample is matched to its CLOSEST point on the reference path (2 ms
        resample of ``traj``):

          * ``path_dev`` — residual to that closest point (max joint rad, max
            TCP m): did the arm move ALONG the commanded path? Latency-free;
            the real belt-safety number.
          * ``lag`` — median (sample time - matched reference time): transport
            latency in sim; transport + servo lag on hardware.
        """
        ts = np.asarray(ts, float)
        grid_t = np.arange(0.0, float(ts[-1]) + 2e-3, 2e-3)
        q_grid = np.stack([np.interp(grid_t, ts, traj[j]) for j in range(6)])
        max_dq = 0.0
        max_dp = 0.0
        sag = 0.0            # worst BELOW-path z excursion (m, >=0) — belt safety
        lags = []
        for t, q in self._samples:
            rel = t - t0
            if rel < 0.0 or rel > ts[-1] + 0.2:
                continue
            q = np.asarray(q, float)
            d = np.linalg.norm(q_grid - q[:, None], axis=0)
            k = int(np.argmin(d))
            lags.append(rel - grid_t[k])
            dq = float(np.max(np.abs(q - q_grid[:, k])))
            p = robot.forward_kinematics(q)[:3, 3]
            p_ref = robot.forward_kinematics(q_grid[:, k])[:3, 3]
            dp = float(np.linalg.norm(p - p_ref))
            sag = max(sag, float(p_ref[2] - p[2]))   # measured BELOW reference
            if dq > max_dq:
                max_dq = dq
                max_dp = dp
        lag = float(np.median(lags)) if lags else 0.0
        return max_dq, max_dp, sag, lag


# =========================================================================
# Test harness
# =========================================================================

class LineStreamTest:
    def __init__(self, node: Node, args) -> None:
        self.node = node
        self.args = args
        self.robot = GP8()
        self.ctrl = TrajectoryController(node)
        cfg = Config()
        self.hz = args.hz if args.hz else cfg.TRAJ_HZ
        # Production limits (what the deployed skill runs at) — used for the
        # 4414 clamp and the joint-space comparison.
        self.M1 = np.asarray(self.robot.velocity_limits, float) * cfg.JOINT_VEL_LIMIT_SCALE
        self.M2 = self.M1 * cfg.JOINT_ACCEL_LIMIT_SCALE
        # Slow limits for the positioning (non-measured) joint moves.
        self.M1_slow = self.M1 * args.speed_scale
        self.M2_slow = self.M1_slow * 4.0

        self.p_start = np.asarray(args.start, float)
        self.p_goal = np.asarray(args.goal, float)
        d = self.p_goal - self.p_start
        d_xy = np.array([d[0], d[1], 0.0])
        if np.linalg.norm(d_xy) < 1e-9:
            d_xy = np.array([1.0, 0.0, 0.0])   # vertical test line: face +X
        self.direction = d_xy / np.linalg.norm(d_xy)
        self.R = PushSkill._push_orientation(self.direction, np.radians(args.swing))
        self.hover_start = self.p_start.copy()
        self.hover_start[2] = args.hover_z
        self.hover_goal = self.p_goal.copy()
        self.hover_goal[2] = args.hover_z
        self.results: list[dict] = []

    # ---------------- low-level moves ----------------
    def _current(self) -> np.ndarray:
        q = self.ctrl.current_joints
        if q is None:
            raise RuntimeError("no /joint_states — is the driver/mock up?")
        return np.asarray(q, float)

    def _ik(self, p: np.ndarray, seed: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = p
        q = self.robot.inverse_kinematics(T, q_init=seed)
        if q is None:
            raise RuntimeError(f"IK failed for point {p}")
        return np.asarray(q, float)

    def _settle(self, q_target: np.ndarray, timeout: float = 3.0) -> float:
        """Wait until the MEASURED joints reach ``q_target``.

        send_trajectory_queue returns when the last COMMAND is published, not
        when the arm arrives — on the real robot the servo lags the stream by
        ~0.1-0.2 s. Without this wait consecutive segments blend: the next
        stream starts while the arm is still short of the previous target and
        the lagging servo cuts the corner (the observed diagonal path).
        Returns the settle time (s); warns on timeout.
        """
        t0 = time.time()
        ok = self.ctrl._wait_for_position(q_target, tolerance=0.03,
                                          timeout_sec=timeout)
        dt = time.time() - t0
        if not ok:
            print(f"  !! settle timeout ({timeout:.1f}s): arm never reached "
                  f"the segment target within 0.03 rad")
        return dt

    def _joint_move(self, q_to: np.ndarray) -> None:
        """Slow joint-space positioning move (not part of the measurement)."""
        q_from = self._current()
        if float(np.max(np.abs(q_to - q_from))) < 1e-3:
            return
        zero = np.zeros(6)
        traj, vel, ts = trajectory(q_from, zero, q_to, zero,
                                   self.M1_slow, self.M2_slow, hertz=self.hz)
        self.ctrl.send_trajectory_queue(traj, vel, ts, final_joint=q_to)
        self._settle(q_to)

    def _run_segment(self, label: str, p0, p1, speed: float) -> dict:
        """Build + stream one EE line; return its metrics row."""
        q_seed = self._current()
        traj, vel, ts, info = build_ee_line(
            self.robot, p0, p1, self.R, speed, self.hz, q_seed,
            accel=self.args.accel)
        traj, vel, ts, raw_ratio, stretch = clamp_velocity(traj, vel, ts, self.M1)
        z_lo, z_hi = ee_z_range(self.robot, traj)
        planned = float(ts[-1]) + 0.05          # +final_joint settle point

        with Tracker(self.ctrl) as trk:
            t0 = time.time()
            self.ctrl.send_trajectory_queue(traj, vel, ts, final_joint=traj[:, -1])
            wall = time.time() - t0
        # Wait for the ARM (not just the command stream) to reach the segment
        # end before the caller starts the next segment — otherwise segments
        # blend and the lagging servo cuts corners (diagonal path on HW).
        settle = self._settle(traj[:, -1])
        max_dq, max_dp, sag, lag = trk.error_vs(self.robot, t0, ts, traj)

        row = {
            "label": label, "speed": speed, "n": info["n"],
            "v_peak": info["v_peak"],
            "ik_fail": info["ik_fail"], "max_jump": info["max_jump"],
            "raw_ratio": raw_ratio, "stretch": stretch,
            "planned": planned, "wall": wall, "settle": settle,
            "path_dq_deg": np.degrees(max_dq), "path_dp_mm": max_dp * 1e3,
            "sag_mm": sag * 1e3, "lag_ms": lag * 1e3,
            "z_lo": z_lo, "z_hi": z_hi,
        }
        self._print_row(row)
        return row

    def _print_row(self, r: dict) -> None:
        flags = []
        if r["ik_fail"] is not None:
            flags.append(f"IK-FAIL@{r['ik_fail']}")
        if r["max_jump"] > 1.0:
            flags.append("BRANCH-FLIP?")
        if r["raw_ratio"] > 1.0:
            flags.append(f"4414 (x{r['stretch']:.2f} stretched)")
        if r["v_peak"] < r["speed"] - 1e-6:
            flags.append(f"line too short: peak {r['v_peak']:.2f} m/s")
        print(
            f"  {r['label']:<9} v={r['speed']:4.2f}  n={r['n']:3d}  "
            f"vel-ratio={r['raw_ratio']:4.2f}  "
            f"T plan/wall={r['planned']:5.3f}/{r['wall']:5.3f}s  "
            f"settle {r['settle'] * 1e3:4.0f}ms  "
            f"path-dev {r['path_dq_deg']:5.2f}deg/{r['path_dp_mm']:5.1f}mm  "
            f"sag {r['sag_mm']:4.1f}mm  lag {r['lag_ms']:4.0f}ms  "
            f"z[{r['z_lo']:+.3f},{r['z_hi']:+.3f}]"
            + ("  << " + ", ".join(flags) if flags else "")
        )

    # ---------------- one full cycle ----------------
    def run_speed(self, speed: float) -> None:
        print(f"-- speed {speed:.2f} m/s "
              f"(line {self.p_start} -> {self.p_goal}, knots {self.hz:.0f} Hz)")
        q_hover = self._ik(self.hover_start, self._current())
        self._joint_move(q_hover)
        rows = [
            self._run_segment("descent", self.hover_start, self.p_start,
                              self.args.descent_speed),
            self._run_segment("stroke", self.p_start, self.p_goal, speed),
            self._run_segment("ascent", self.p_goal, self.hover_goal,
                              self.args.descent_speed),
        ]
        self.results.extend(rows)
        time.sleep(0.2)

    # ---------------- belt-dip evidence (offline, no motion) ----------
    def compare_jointspace(self) -> None:
        """FK-sample the CURRENT joint-space interpolation between the same
        endpoints (production M1/M2) and report min EE z + line deviation."""
        print("-- joint-space interpolation comparison (offline, no motion):")
        q_hover = self._ik(self.hover_start, self._current())
        q_start = self._ik(self.p_start, q_hover)
        q_goal = self._ik(self.p_goal, q_start)
        zero = np.zeros(6)
        for label, qa, qb, pa, pb in (
            ("descent", q_hover, q_start, self.hover_start, self.p_start),
            ("stroke", q_start, q_goal, self.p_start, self.p_goal),
            ("chain-up", q_goal, q_hover, self.p_goal, self.hover_start),
        ):
            traj, _, _ = trajectory(qa, zero, qb, zero, self.M1, self.M2,
                                    hertz=50.0)
            z_lo, z_hi = ee_z_range(self.robot, traj)
            dev = line_deviation(self.robot, traj, pa, pb)
            dip = z_lo - min(float(pa[2]), float(pb[2]))
            print(
                f"  {label:<9} joint-space: min z={z_lo:+.3f} "
                f"({'DIPS ' + format(-dip, '.3f') + ' m BELOW endpoints' if dip < -1e-3 else 'no dip'}), "
                f"max line deviation={dev * 1e3:6.1f} mm"
            )
        print("  (EE-line path holds z within its endpoints by construction)")


# =========================================================================
# main
# =========================================================================

def _xyz(s: str):
    v = [float(x) for x in s.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=_xyz, default=[0.4, 0.0, 0.1],
                    help="line start x,y,z (m); default 0.4,0,0.1")
    ap.add_argument("--goal", type=_xyz, default=[0.6, 0.0, 0.1],
                    help="line goal x,y,z (m); default 0.6,0,0.1")
    ap.add_argument("--speed", type=float, default=None,
                    help="single stroke speed (m/s)")
    ap.add_argument("--sweep", type=str, default="0.5,1.0,1.5,2.0,2.5",
                    help="comma list of stroke speeds (m/s); ignored if --speed")
    ap.add_argument("--hz", type=float, default=None,
                    help="knot rate (default: Config.TRAJ_HZ)")
    ap.add_argument("--descent-speed", type=float, default=0.3,
                    help="descent/ascent TCP speed (m/s)")
    ap.add_argument("--accel", type=float, default=5.0,
                    help="TCP accel/decel for the trapezoid arc profile "
                         "(m/s^2); a velocity-step line is untrackable")
    ap.add_argument("--hover-z", type=float, default=0.25,
                    help="hover height above the line (m)")
    ap.add_argument("--swing", type=float, default=0.0,
                    help="fixed swing tilt (deg) for the whole run")
    ap.add_argument("--speed-scale", type=float, default=0.15,
                    help="joint-limit fraction for positioning moves")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the joint-space belt-dip comparison")
    ap.add_argument("--yes", action="store_true",
                    help="skip the safety confirmation prompt")
    args = ap.parse_args()

    speeds = ([args.speed] if args.speed is not None
              else [float(s) for s in args.sweep.split(",") if s.strip()])

    rclpy.init()
    node = Node("line_stream_test")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def _spin() -> None:
        try:
            executor.spin()
        except Exception:
            pass   # context torn down on shutdown — teardown noise only

    spin = threading.Thread(target=_spin, daemon=True)
    spin.start()

    def _teardown() -> None:
        # Orderly stop: shut rclpy down FIRST (unblocks executor.spin), then
        # join the spin thread — exiting the process with the thread still in
        # rcl wait aborts with "terminate called without an active exception".
        rclpy.shutdown()
        spin.join(timeout=2.0)

    test = LineStreamTest(node, args)

    print("waiting for /joint_states ...")
    t0 = time.time()
    while test.ctrl.current_joints is None:
        if time.time() - t0 > 10.0:
            print("no /joint_states after 10 s — launch debug_robot.launch.py "
                  "(sim) or the adv4ncr driver stack (HW) first, and check "
                  "ROS_DOMAIN_ID matches it")
            _teardown()
            return
        time.sleep(0.1)

    if not args.yes:
        input(
            "This MOVES the robot (real swing on hardware — clear 1 m around "
            "the arm). ENTER to start, Ctrl-C to abort ... "
        )

    try:
        if not args.no_compare:
            test.compare_jointspace()
        for v in speeds:
            test.run_speed(v)
    finally:
        print("\n==== summary (stroke rows) " + "=" * 40)
        for r in test.results:
            if r["label"] == "stroke":
                verdict = ("STRETCHED" if r["raw_ratio"] > 1.0 else "ok")
                print(
                    f"  v={r['speed']:4.2f} m/s  vel-ratio={r['raw_ratio']:4.2f} "
                    f"[{verdict}]  overrun={(r['wall'] - r['planned']) * 1e3:+6.0f} ms  "
                    f"path-dev {r['path_dq_deg']:.2f} deg / {r['path_dp_mm']:.1f} mm  "
                    f"sag {r['sag_mm']:.1f} mm  lag {r['lag_ms']:.0f} ms"
                )
        _teardown()


if __name__ == "__main__":
    main()
