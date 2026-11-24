"""Cycle mode without the robot: planner (local or robot6) + the real runner, fake arm.

Runs CycleThrowRunner exactly as the app does, but nothing reaches a robot and
there is no ROS. Objects come from a vendored schedule file, placed the way the
vendored rt_dry.py places them (y = 1.40 m at t + 3.6 s).

  --output buffer (default, = hardware): a fake SampleBufferController with the
      C++ controller's rules (2 ms tick on this PC's clock, min_lead 3, splice
      velocity/acceleration check incl. tol, gap = hold, buffer end = hold)
      behind the real BufferClient. It also checks that every tick put out
      exactly the planner's command.
  --output stream: the app's own 2 ms stream into a recording arm.
  --ros: instead of the fake, the real TrajectoryController + the real C++
      SampleBufferController on ros2_control MOCK hardware. Start it first, on
      its own domain (never the robot's):
        ROS_DOMAIN_ID=77 ros2 launch motoman_bringup gp8.launch.py \
            use_mock_hardware:=true sample_buffer:=true
      and run this with ROS_DOMAIN_ID=77 GP8_ROBOT_IP=127.0.0.1 GP8_STREAM_HZ=500.

  PYTHONPATH=~/iitp_demo_ws/src ~/iitp_demo_ws/src/gp8_control/.venv/bin/python \\
      -m gp8_control.tools.cycle_dry_check --host robot6-planner --seconds 30

Pass: late pieces 0, no refused piece, solves cut at the deadline <= 20 %
(~10 % is normal), (buffer) ticks == planned commands.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import types

import numpy as np

from gp8_control.backends.robot_base import RobotBackend
from gp8_control.config import Config
from gp8_control.controllers.sample_buffer import BufferClient
from gp8_control.cycle_throw import runner as R
from gp8_control.robots.gp8 import GP8
from gp8_control.tracking.object_queue import TrackedObject, TrackedObjectQueue

from gp8_control.controllers.sample_buffer_rules import (  # noqa: E402  (one copy of the C++ rules)
    A_FACTOR,
    MIN_LEAD,
    TOL,
    TS,
    SampleBufferRules,
)


class _Log:
    def __init__(self):
        self.lines = []

    def _p(self, lvl, m):
        self.lines.append(str(m))
        if lvl != "D" and (lvl != "I" or not str(m).startswith("[cycle] commands")):
            print(f"{time.strftime('%H:%M:%S')} {lvl} {m}", flush=True)

    def info(self, m): self._p("I", m)
    def warn(self, m): self._p("W", m)
    def error(self, m): self._p("E", m)
    def debug(self, m): pass


class _FakeSampleBuffer(SampleBufferRules):
    """motoman_controllers/SampleBuffer on this PC's clock: the controller's
    rules (inherited, controllers/sample_buffer_rules.py) on a 2 ms wall tick."""

    def __init__(self, q0, a_factor=A_FACTOR, tol=TOL, min_lead=MIN_LEAD):
        super().__init__(q0, min_lead=min_lead, a_factor=a_factor, tol=tol)
        self.lock = threading.Lock()
        self.client = None
        self.out: dict[int, np.ndarray] = {}      # tick -> commanded (this activation)
        self.activate(np.asarray(q0, float))
        threading.Thread(target=self._loop, daemon=True).start()

    def activate(self, hold) -> None:
        with self.lock:
            self.t_act = time.monotonic()
            self.reset(hold, k=0)
            self.pending, self.last_seq, self.last_res = [], 0, 0
            self.out = {}

    def send_chunk(self, seq, start, Q):
        with self.lock:
            self.pending.append((int(seq), int(start), np.asarray(Q, float).reshape(-1, 6), False))

    def send_stop(self, seq):
        with self.lock:
            self.pending.append((int(seq), 0, None, True))

    def _merge(self, seq, s, Q, stop):
        self.last_seq = seq
        if stop:
            self.stop_now()
            self.last_res = 0
            return
        res, why = self.merge(s, Q)
        self.last_res = res
        if why is not None:
            self.rejects[-1] = (seq, why)

    def _loop(self):
        while True:
            with self.lock:
                t_next = self.t_act + (self.k + 1) * TS
            dt = t_next - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            with self.lock:
                for c in self.pending:
                    self._merge(*c)
                self.pending = []
                idx, cmd, holding = self.output()
                self.out[idx] = cmd
                args = (idx, time.monotonic(), self.end, holding, self.last_seq, self.last_res,
                        cmd.tolist(), self.t_act + idx * TS)
            if self.client is not None:
                self.client.on_progress(*args[:6], command=args[6], t_stamp=args[7])


class _FakeArm(RobotBackend):
    """JGPC stream + (optional) SampleBuffer, both into nowhere."""

    def __init__(self, buffer: bool):
        super().__init__()
        self.samples = []
        self.current_joints = list(np.asarray(R.CYCLE_HOME) + 0.01)
        self.fake = None
        self.buf = None
        self.on_buffer = False
        if buffer:
            self.fake = _FakeSampleBuffer(self.current_joints)
            self.buf = BufferClient(self.fake.send_chunk, self.fake.send_stop,
                                    fire=lambda on: self.suction_on() if on else self.suction_off(),
                                    name="fake-buf")
            self.fake.client = self.buf

    def wait_for_servers(self, timeout_sec=10.0):
        return True

    def _emit_sample(self, positions):
        self.samples.append(time.monotonic())
        self.current_joints = list(positions)

    def _set_suction(self, on, requested_at):
        pass

    def buf_switch(self, to_buffer: bool, timeout_sec: float = 5.0) -> bool:
        self.on_buffer = bool(to_buffer)
        return True

    def buf_activate(self, timeout_sec: float = 5.0) -> bool:
        self.fake.activate(self.current_joints)
        self.on_buffer = True
        t_end = time.monotonic() + timeout_sec
        n0 = self.buf.n_progress
        while time.monotonic() < t_end:
            if self.buf.n_progress > n0 + 5 and self.buf.ready():
                return True
            time.sleep(0.01)
        return False


class _Belt:
    current = R.PLANNER_BELT_SPEED
    distance_m = None

    def check_freshness(self):
        pass


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cycle_dry_check")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--schedule", default="normal",
                    help="dense|tight|normal|burst|bigburst or a path to a schedule json")
    ap.add_argument("--host", default=None, help="planner host (default: GP8_CYCLE_PLANNER_HOST)")
    ap.add_argument("--output", choices=("buffer", "stream"), default="buffer")
    ap.add_argument("--mode", choices=("cycle", "pieces"), default="cycle",
                    help="cycle: CycleThrowRunner (push slots); pieces: PieceRunner")
    ap.add_argument("--push-test", type=float, default=None, metavar="T",
                    help="pieces: at T s, with the arm at rest, plan a stand-in push step "
                         "(J1 +0.15 rad and back) through the push path")
    ap.add_argument("--slot-test", type=float, default=None, metavar="T",
                    help="force one push slot at planner time T: a JGPC move of J1 +0.15 rad "
                         "stands in for the push (tests drain, switch, return, re-activation)")
    ap.add_argument("--ros", action="store_true",
                    help="real TrajectoryController + SampleBufferController on mock hardware")
    args, _ = ap.parse_known_args(argv)
    os.environ.setdefault("GP8_CYCLE_DRAIN_S", "6")
    os.environ["GP8_CYCLE_OUTPUT"] = args.output
    if args.host is not None:
        os.environ["GP8_CYCLE_PLANNER_HOST"] = args.host
    path = args.schedule
    if not os.path.isfile(path):
        path = str(R.find_planner_dir().parent / "examples" / f"schedule_{args.schedule}.json")
    sched = json.load(open(path))

    cfg = Config()
    cfg.BACKEND = "hw"
    robot = GP8()
    log = _Log()
    ticks_out: dict[int, np.ndarray] = {}
    if args.ros:
        if os.environ.get("ROS_DOMAIN_ID") != "77" or os.environ.get("GP8_ROBOT_IP") != "127.0.0.1":
            raise SystemExit("--ros needs ROS_DOMAIN_ID=77 and GP8_ROBOT_IP=127.0.0.1 (mock only)")
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        os.environ["GP8_MODE"] = args.mode
        from gp8_control.controllers.trajectory_controller import TrajectoryController
        rclpy.init()
        node = rclpy.create_node("gp8_cycle_dry")
        arm = TrajectoryController(node)
        ex = MultiThreadedExecutor(num_threads=3)
        ex.add_node(node)
        threading.Thread(target=ex.spin, daemon=True).start()
        t_w = time.monotonic() + 10
        while arm.current_joints is None and time.monotonic() < t_w:
            time.sleep(0.05)
        if arm.current_joints is None:
            raise SystemExit("no /joint_states on domain 77: is the mock bringup up?")
        if arm.buf is not None:
            gen = {"last": -1}

            def _hook(k, t, cmd):
                if int(k) < gen["last"]:          # re-activation: the tick restarted at 0
                    ticks_out.clear()
                gen["last"] = int(k)
                ticks_out[int(k)] = np.asarray(cmd, float)

            arm.buf.tick_hooks.append(_hook)
    else:
        arm = _FakeArm(buffer=args.output == "buffer")
    app = types.SimpleNamespace(
        cfg=cfg, robot=robot, traj_ctrl=arm, conveyor=_Belt(),
        queue=TrackedObjectQueue(cfg.MAX_REACH, drop_below_y=-cfg.MAX_REACH),
        _node=types.SimpleNamespace(get_logger=lambda: log), _spinner_dead=False)
    app.M1 = np.asarray(robot.velocity_limits) * cfg.JOINT_VEL_LIMIT_SCALE
    app.M2 = app.M1 * cfg.JOINT_ACCEL_LIMIT_SCALE
    if args.mode == "pieces":
        from gp8_control.cycle_throw.piece_runner import PieceRunner
        from gp8_control.skills.planner_throw_skill import PlannerThrowSkill
        runner = PieceRunner(app, ctx=None, skills={"throw": PlannerThrowSkill(None)},
                             skill_for=None)
    else:
        runner = R.CycleThrowRunner(app)
    if R.STREAM_HZ != 500.0:
        runner.cfg.BACKEND = "mujoco"                 # skip the HW 500 Hz guard: nothing is sent
    planned: dict[int, np.ndarray] = {}
    on_commit = runner._on_commit

    def record(k, rows):
        for i, q in enumerate(np.asarray(rows, float)):
            planned[int(k) + i] = q
        on_commit(k, rows)

    runner._on_commit = record
    if args.slot_test is not None:
        slot = {"done": False}

        def _due():
            if not slot["done"] and runner.t0 is not None and runner.now() > args.slot_test:
                return [types.SimpleNamespace(track_id=-1, class_name="slot-test")]
            return []

        def _fake_push(obj):
            slot["done"] = True
            q = np.asarray(arm.current_joints, float) + np.array([0.15, 0, 0, 0, 0, 0])
            runner._move_to(q)
            return "moved J1 +0.15 rad (stand-in push)"

        runner._push_due = _due
        runner.push_handler = _fake_push
    push_test = {"sent": False}
    if args.push_test is not None and args.mode == "pieces":
        from gp8_control.backends.robot_base import resample_trajectory
        from gp8_control.trajectory.trajectory_primitive import trajectory
        decide = runner._decide

        def _decide():
            a = runner.arm
            if not push_test["sent"] and a["kind"] == "rest" and runner.now() > args.push_test:
                # A stand-in push step: "stroke" commands J1 +0.15 rad and back
                # from the rest pose, fired 3 s from now (planned through
                # F(U) + the fit, like a real push).
                push_test["sent"] = True
                q0 = np.asarray(a["q"], float)
                q1 = q0 + np.array([0.15, 0, 0, 0, 0, 0])
                z = np.zeros(6)
                U = []
                for qa, qb in ((q0, q1), (q1, q0)):
                    tr, ve, tt = trajectory(qa, z, qb, z, app.M1 * 0.5, app.M2 * 0.5, hertz=cfg.TRAJ_HZ)
                    _g, S_ = resample_trajectory(tr, ve, tt, qb, R.PLANNER_DT)
                    U.append(S_ if not U else S_[1:])
                U = np.vstack(U)
                step = {"kind": "push", "id": -7, "U": U.round(9).tolist(),
                        "t_f": max(runner.now(), float(a["t_free"])) + 3.0,
                        "i_contact": len(U) // 2, "i_cut": len(U) - 1, "q_hover": q0.tolist()}
                runner._request([step], {-7: "push"}, ["push-test"])
                log.info(f"[dry] push-test: {len(U)} stroke samples")
                return
            decide()

        runner._decide = _decide
    spawned, t_end = set(), [float("inf")]

    def intake():
        if runner.t0 is None:
            return
        if t_end[0] == float("inf"):
            t_end[0] = time.monotonic() + args.seconds
        tp = runner.now()
        for d in sched:
            ts = d["t"] + 3.6
            if d["id"] in spawned or tp < ts:
                continue
            spawned.add(d["id"])
            T = np.eye(4)
            T[:3, 3] = (float(d["x"]), 1.40, 0.02)
            app.queue._objects.append(TrackedObject(
                T_aim_base=T.copy(), T_grasp_base=T, class_name=d["cls"],
                detect_time=time.time() - (tp - ts), n_hits=10))
        app.queue.update(time.time(), R.PLANNER_BELT_SPEED)   # the schedule belt runs at the plan's speed

    err = None
    try:
        runner.run(intake=intake, ok=lambda: time.monotonic() < t_end[0])
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    solves = [m for m in log.lines
              if re.search(r"plan (s\d+ cycle \d+|r\d+ throw|stop throw)[^:]*:", m)]
    cut = [m for m in solves if "User_Requested_Stop" in m]
    sync = next((m for m in log.lines if "clock sync" in m), "(no clock sync line)")
    late = runner.stats.get("late_pieces", 0)
    if args.slot_test is not None:
        sp = runner.splices[0] if runner.splices else None
        parts_slot = (f"slot: drain {sp['t_rest'] - sp['t_drain']:.1f} s, rest->resume "
                      f"{sp['t_resume'] - sp['t_rest']:.1f} s, return err {sp['return_err_rad']:.4f} rad"
                      if sp else "slot: NOT RUN")
    else:
        parts_slot = None
    parts = [f"host {runner.planner_host or 'local'}", sync.replace("[cycle] ", ""),
             f"cycle solves {len(solves)}, cut at the deadline {len(cut)}", f"late pieces {late}",
             f"suction switches {runner.stats.get('suction_on', 0) + runner.stats.get('suction_off', 0)}"]
    ok = err is None and late == 0 and len(cut) <= max(1, 0.2 * len(solves))
    if args.push_test is not None and args.mode == "pieces":
        ok_c = runner.stats.get("push_planned", 0) >= 1
        parts.append(f"push-test: {'planned' if ok_c else 'NOT RUN / refused'}")
        ok = ok and ok_c
    if parts_slot is not None:
        parts.append(parts_slot)
        ok = ok and bool(runner.splices) and bool(runner.splices[0]["pushed"])
    fake = getattr(arm, "fake", None)
    if runner.out is not None:
        out = dict(fake.out if fake is not None else ticks_out)   # the buffer thread still ticks
        shift = runner.out.shift
        ticks = sorted(k - shift for k in out if (k - shift) in planned)
        diff = max((float(np.max(np.abs(out[k + shift] - planned[k]))) for k in ticks),
                   default=float("nan"))
        acc = (f"pieces accepted {fake.n_acc}, refused {fake.n_rej} {fake.rejects[:3]}"
               if fake is not None else f"refused {'yes: ' + str(runner._fault) if runner._fault else 0}")
        parts += [acc,
                  f"min write lead {runner.out.min_lead_s:.3f} s" if runner.out.min_lead_s is not None
                  else "no write",
                  f"ticks vs planned: {len(ticks)} ticks, max |diff| {diff:.2e} rad"]
        ok = ok and (fake is None or fake.n_rej == 0) and bool(ticks) and diff < 1e-9
    else:
        gaps = np.diff(arm.samples[-15000:]) if len(arm.samples) > 2 else []
        parts.append(f"max send lag {1e3 * runner.max_lag:.1f} ms"
                     + (f", gap p99 {1e3 * float(np.percentile(gaps, 99)):.2f} ms" if len(gaps) else ""))
    print(f"\nSUMMARY {'PASS' if ok else 'FAIL'} ({args.output}{', ROS mock' if args.ros else ''}) | "
          + " | ".join(parts) + f" | error {err}")
    if args.ros:
        arm.close()            # back to JointGroupPositionController
        rclpy.shutdown()


if __name__ == "__main__":
    main()
