"""End-to-end smoke test for the MuJoCo backends (NO rclpy / NO torch needed).

Drives the RobotBackend/WorldSource seams directly — the same 250 Hz stream
engine the real hardware runs — against the physics twin, and checks the whole
chain: stepper realtime lock, synthesized schema-v2 perception, the
encoder-equivalent belt distance, a timed ambush pick (suction weld), a lift,
and a timed-release throw (the box flies ballistically off the released weld).

Run (Windows or Linux; needs numpy/scipy/mujoco — ``uv sync --extra sim``):

    python -m gp8_control.tests.sim_smoke [--viewer]

with the repo's parent on PYTHONPATH so ``gp8_control.*`` imports resolve.
Prints PASS/FAIL per stage; exit code 0 iff all pass.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from gp8_control.backends.mujoco_sim import (
    MujocoRobotBackend,
    MujocoWorldSource,
    SimConfig,
    SimCore,
)
from gp8_control.robots.gp8 import GP8
from gp8_control.trajectory.trajectory_primitive import trajectory

# Tool-down grasp orientation (== perception/detection_intake._R_GRASP_DEFAULT).
_R_GRASP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
_HOME = [0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]


def _make_T(Rm, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


class _Stage:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            self.failures += 1
        return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sim_smoke")
    parser.add_argument("--viewer", action="store_true",
                        help="open the MuJoCo passive viewer window")
    args = parser.parse_args(argv)

    st = _Stage()
    cfg = SimConfig(spawn_interval=3.0)
    if args.viewer:
        cfg.viewer = True
    core = SimCore(cfg)
    backend = MujocoRobotBackend(core)
    world = MujocoWorldSource(core)

    st.check("backend ready (wait_for_servers)", backend.wait_for_servers(10.0))

    # -- stage 1: stepper realtime lock ------------------------------------
    t0_wall = time.monotonic()
    t0_sim = core.data.time
    time.sleep(2.0)
    ratio = (core.data.time - t0_sim) / (time.monotonic() - t0_wall)
    st.check("stepper realtime ratio ~= 1.0", 0.9 < ratio < 1.1, f"ratio={ratio:.3f}")
    st.check("current_joints fed by stepper", backend.current_joints is not None)

    # -- stage 2: synthesized perception -----------------------------------
    deadline = time.time() + 2.0 * cfg.spawn_interval + 5.0
    snap = None
    while time.time() < deadline:
        snap = world.latest_snapshot()
        if snap and snap["detections"]:
            break
        time.sleep(0.1)
    ok = bool(snap and snap["detections"])
    st.check("snapshot with detections", ok)
    if not ok:
        core.stop()
        return 1
    det = snap["detections"][0]
    st.check("schema-v2 keys present", all(
        k in det for k in ("class", "confidence", "cam", "cam_bbox",
                           "base_grasp", "base_aim", "base_bbox_grasp",
                           "base_bbox_aim", "in_workspace")))
    st.check("box in workspace", bool(det["in_workspace"]),
             f"cam x={det['cam'][0]:+.3f}")
    r1 = snap["receipt_time"]
    time.sleep(0.3)
    snap2 = world.latest_snapshot()
    st.check("receipt_time strictly increasing", snap2["receipt_time"] > r1)

    # -- stage 3: encoder-equivalent belt distance -------------------------
    belt = world.belt
    d0, t0 = belt.distance_at(time.time()), time.time()
    time.sleep(1.0)
    d1, t1 = belt.distance_at(time.time()), time.time()
    rate = (d1 - d0) / (t1 - t0)
    st.check("belt.distance_at advances ~ belt_speed",
             abs(rate - cfg.belt_speed) < 0.02 * max(1.0, cfg.belt_speed / 0.1),
             f"rate={rate:.4f} vs {cfg.belt_speed:.4f}")

    # -- stage 4: timed ambush pick (suction weld) -------------------------
    robot = GP8()
    M1 = np.asarray(robot.velocity_limits, dtype=float) * 0.9
    M2 = M1 * 6.0

    # Track the freshest upstream box via the world seam and pick an intercept
    # it will reach in a few seconds. Detections give base X/Y; grasp height is
    # the app's GRASP_Z convention == cfg.grasp_z (the detection z is the
    # camera-plane placeholder, exactly like hardware).
    deadline = time.time() + 30.0
    target = None
    while time.time() < deadline:
        snap = world.latest_snapshot()
        cands = [d for d in (snap["detections"] if snap else [])
                 if d["in_workspace"] and d["base_grasp"][1] > 0.35]
        if cands:
            target = max(cands, key=lambda d: d["base_grasp"][1])
            t_snap = snap["receipt_time"]
            break
        time.sleep(0.1)
    if not st.check("upstream box found for pick", target is not None):
        core.stop()
        return 1

    x_b, y_b = float(target["base_grasp"][0]), float(target["base_grasp"][1])
    y_int = 0.15                                  # ambush line (reachable)
    eta = t_snap + (y_b - y_int) / cfg.belt_speed  # box centre crosses y_int
    grasp_p = [x_b, y_int, cfg.grasp_z]
    hover_p = [x_b, y_int, cfg.grasp_z + 0.10]
    q_hover = robot.inverse_kinematics(_make_T(_R_GRASP, hover_p),
                                       q_init=np.asarray(_HOME))
    q_grasp = robot.inverse_kinematics(_make_T(_R_GRASP, grasp_p),
                                       q_init=q_hover)
    if not st.check("IK for hover/grasp", q_hover is not None and q_grasp is not None):
        core.stop()
        return 1

    qi = np.asarray(backend.current_joints)
    tr, vl, ts = trajectory(qi, np.zeros(6), q_hover, np.zeros(6), M1, M2)
    backend.send_trajectory_queue(tr, vl, ts, final_joint=q_hover)
    tr, vl, ts = trajectory(q_hover, np.zeros(6), q_grasp, np.zeros(6), M1 * 0.4, M2 * 0.4)
    backend.send_trajectory_queue(tr, vl, ts, final_joint=q_grasp)
    st.check("arrived at grasp before the box", time.time() < eta - 0.2,
             f"margin={eta - time.time():+.2f}s")

    # Hold at the grasp pose (the real ambush wait), priming suction at ETA
    # from the hold loop's tick — the same pq_hold_until pattern the skills use.
    def _prime() -> None:
        if time.time() >= eta:
            backend.suction_on()   # coalesced: repeated ticks are no-ops

    backend.pq_hold_until(q_grasp, eta + 0.3, tick_fn=_prime)
    st.check("suction primed at ETA", backend.last_suction_on_t is not None)
    time.sleep(0.2)
    grabbed = core._grabbed
    st.check("suction weld grabbed a box", grabbed is not None)
    if grabbed is None:
        core.stop()
        return 1
    st.check("weld constraint active", int(core.data.eq_active[core._weld_ids[grabbed]]) == 1)

    # -- stage 5: lift — the welded box must follow the cup ----------------
    box_z0 = float(core._box_joints[grabbed].qpos[2])
    tr, vl, ts = trajectory(q_grasp, np.zeros(6), q_hover, np.zeros(6), M1 * 0.4, M2 * 0.4)
    backend.send_trajectory_queue(tr, vl, ts, final_joint=q_hover)
    time.sleep(0.2)
    box_z1 = float(core._box_joints[grabbed].qpos[2])
    st.check("welded box lifted with the cup", box_z1 - box_z0 > 0.05,
             f"dz={box_z1 - box_z0:+.3f}m")

    # -- stage 6: timed-release throw --------------------------------------
    # Hand-rolled upward swing (no torch): sweep S while raising the arm; the
    # timed release fires mid-swing on the 4 ms grid, the weld drops, and the
    # box leaves with the swing's velocity.
    q_throw_end = np.asarray(q_hover) + np.array([0.8, -0.35, 0.25, 0.0, 0.6, 0.0])
    tr, vl, ts = trajectory(np.asarray(q_hover), np.zeros(6), q_throw_end,
                            np.zeros(6), M1, M2)
    rel = int(tr.shape[1] * 0.6)
    box_p_rel = None
    backend.send_trajectory_queue_with_timed_release(
        tr, vl, ts, final_joint=q_throw_end, release_index=rel)
    box_p_rel = np.array(core._box_joints[grabbed].qpos[0:3])
    lt = backend.last_throw or {}
    st.check("last_throw recorded", lt.get("release_wall") is not None,
             f"release_index={lt.get('release_index')}")
    st.check("weld released", int(core.data.eq_active[core._weld_ids[grabbed]]) == 0)
    time.sleep(0.5)
    box_p_now = np.array(core._box_joints[grabbed].qpos[0:3])
    disp = float(np.linalg.norm(box_p_now - box_p_rel))
    st.check("box flew ballistically after release", disp > 0.15,
             f"displacement={disp:.3f}m in 0.5s")

    core.stop()
    print(f"\n{'ALL PASS' if st.failures == 0 else f'{st.failures} FAILURE(S)'}")
    return 0 if st.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
