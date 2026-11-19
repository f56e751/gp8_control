"""RobotBackend — the ONE robot-control seam app/skills drive, HW **or** sim.

Why this exists
---------------
Everything the orchestrator and the manipulation skills need from "the robot"
is a small, stable set of calls. The *real* control path
(``controllers/trajectory_controller.py``, adv4ncr 500 Hz stream) and the *sim*
path historically drifted because the old sim mocked the ROS **wire** protocol
(Point Queue Mode services, ``/write_single_io``, ``/joint_states_urdf``) — and
that wire changed under the sim when the backend migrated to ros2_control. The
wire was never the real contract; THIS class is.

Crucially, the **hardware operation logic itself lives here**, moved verbatim
from ``trajectory_controller.py``:

  * the 500 Hz stream engine (:meth:`_stream_trajectory` / :meth:`_stream_hold`):
    cubic-Hermite resample onto the 2 ms grid honoring knot velocities, clamped
    to the bracketing-knot envelope (floor-dive / alarm-4315 fix), grid-time
    suction release, prime-after-release, the ``tick_fn`` early-stop protocol,
    and wall-clock pacing to an absolute monotonic schedule;
  * suction request **coalescing** (a second same-state call is a silent no-op
    *including the telemetry stamp*) and the fire-instant telemetry every
    skill's diagnostics read;
  * the blocking motion surface (``send_trajectory_queue*``, ``pq_*``,
    ``_wait_for_position``) and the queue-mode lifecycle shims.

So the MuJoCo twin executes the *identical* resample/clamp/release/pacing code
as the real arm — a backend only supplies three primitives:

  * :meth:`wait_for_servers` — become ready;
  * :meth:`_emit_sample`     — deliver ONE 2 ms joint-position command
    (hardware publishes ``Float64MultiArray`` to the JGPC; the twin writes the
    position-actuator ctrl targets);
  * :meth:`_set_suction`     — physically toggle the vacuum (hardware enqueues
    a Simple-Message TCP write; the twin welds/releases the grasped box).

plus keeps :attr:`current_joints` fresh (a ``/joint_states`` callback on
hardware; the physics stepper in the twin), and may override :meth:`_ok`
(hardware: ``rclpy.ok()``), :meth:`set_motion_op`, :meth:`close`.

This module imports NO rclpy — backends decide their own transport.

NOTE ON POLICY vs. MECHANISM: *when* to release (``release_index``) and *when*
to prime the next pick (``suction_on_at``) are the skill's policy, passed per
call. This class only provides the mechanism; it bakes in no timing of its own.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class TrackingFault(RuntimeError):
    """명령과 컨트롤러 목표(joint_states)의 차가 track_trip_rad 를 TRIP_N 샘플 연속 넘음 — 스트림을 멈춘다 (11-03 실물 lock)."""


TRIP_N = 10                                    # 연속 샘플 수 (20 ms) — joint_states 한 번 늦게 온 것으로는 멈추지 않게

# YRC1000micro interpolation cycle is 2 ms (telnet '[RT] Starting control loop with cycle
# time: 2 ms', 2026-09-18). Streaming at 500 Hz left every other controller cycle without a
# command and capped the arm at exactly half its datasheet speed. Must match
# controllers.yaml update_rate and MotomanHardware nominal_dt.
STREAM_HZ = 500.0
STREAM_DT = 1.0 / STREAM_HZ   # 2 ms
# 송신 마감 정밀 대기 (11-04 실물): time.sleep 은 0.05~0.1 ms 늦게 깨고, 한 주기 증분 감시 (_win_time) 아래에서는 계획 속도가 한계의 50 %
# 를 넘는 구간에서 그 늦은 몫을 따라잡을 수 없어 한 판 (샘플 ~5000) 에 수백 ms 가 쌓였다 (재정렬 8~18 회, 합 0.15~0.68 s → 잡기가
# 상류로 ~11 mm). 마감 SPIN_S 전까지만 sleep 하고 나머지는 sleep(0) (GIL 을 놓는 양보) 로 돌아 마감에 맞춘다. 0 이면 종전 sleep.
SPIN_S = float(os.environ.get("GP8_STREAM_SPIN_S", "0.0006"))


def _sleep_until(t_mono: float) -> None:
    """monotonic 시각 t_mono 까지 대기 — 마지막 SPIN_S 는 양보하며 돌아 늦게 깨지 않게."""
    r = t_mono - time.monotonic()
    if r > SPIN_S:
        time.sleep(r - SPIN_S)
    if SPIN_S <= 0.0:
        return
    while time.monotonic() < t_mono:
        time.sleep(0)


def _max_abs_diff(a, b) -> float:
    n = min(len(a), len(b))
    return max((abs(float(a[i]) - float(b[i])) for i in range(n)), default=0.0)


def _traj_total(timestep) -> float:
    """Total trajectory duration. ``timestep`` from trajectory() is CUMULATIVE time_from_start
    (arange(L+1)/hertz), so the duration is the LAST value — NOT np.sum(timestep) (which is
    ~T*(n+1)/2 and grossly inflates timeouts/deadlines). Review finding #3/#4."""
    t = np.asarray(timestep, dtype=float).ravel()
    return float(t[-1]) if t.size else 0.0


class RobotBackend(ABC):
    """Common robot-control surface skills/app depend on; HW and sim subclass it.

    Subclasses MUST keep :attr:`current_joints` (and optionally
    :attr:`current_jointvels`) fresh from their own state source and implement
    the three abstract methods. Everything else is provided here.
    """

    def __init__(self) -> None:
        # --- live state (the subclass keeps these fresh) ---------------------
        #: Latest measured joint positions [rad], GP8 order [S,L,U,R,B,T];
        #: ``None`` until the first sample arrives.
        self.current_joints: list | None = None
        #: Latest measured joint velocities [rad/s] (same order), or ``None``.
        self.current_jointvels: list | None = None
        #: 던지기 실행 추적 버퍼 (2026-09-09). None = 꺼짐. track_start() 로 켜면 _emit_tracked 가
        #: 2 ms 마다 (monotonic, 명령 6, 측정 6, 측정속도 6) 튜플 하나를 append 한다 — I/O·계산 없음.
        #: 분석·출력은 실험이 끝난 뒤 skills.thr_track.report() 에서 한 번에 한다.
        self.track_buf: list | None = None
        #: 추종 오차 감시 (11-03): None = 끔. 앱이 실물에서 라디안으로 넣는다 (sequence_app GP8_TRACK_TRIP_DEG).
        #: 11-03 실물: 예전 던지기 NLP 의 준비점 던지기(가속 14,000~28,000°/s²)를 팔이 못 따라가 오차 60° 넘게 9 s 동안 명령을 보내다 lock.
        self.track_trip_rad: float | None = None
        self._trip_n = 0
        #: 최근 송신 (monotonic, 자세) — 한 주기 증분 감시 (_win_time). _emit_tracked 가 채운다.
        self._win_hist = collections.deque(maxlen=8)
        self._win_over = 0                     # 샘플 하나로 이미 한 주기 한계를 넘은 횟수 (계획 문제 — 늦춰도 못 고침)

        # --- pick-cycle timing telemetry (kept for the skills' logs) ---------
        self.last_suction_on_t: float | None = None
        self.last_suction_on_ack_t: float | None = None
        # Joint state snapshotted at the instant suction turned ON, so a skill can
        # FK it to the EE pose the vacuum actually fired at (diagnostic).
        self.last_suction_on_joints: list | None = None
        self.last_suction_off_t: float | None = None
        self.last_suction_off_ack_t: float | None = None
        self.last_throw: dict | None = None
        self._last_throw_ok: bool = True   # #R3: last timed-release dispatch accepted?

        # Kept for API compatibility with the skills' timeline model
        # (push_skill.t_to_contact reads qmode_ms_avg for T_setup). On a modern
        # controller there is NO queue-mode re-entry; the only fixed overhead is
        # dispatch + ~40 ms controller dead-time.
        self.last_qmode_ms: float | None = None
        self.qmode_ms_avg: float = 40.0

        # Persistent-session state (pq_*): just tracks "a pick+hold+throw
        # sequence is in progress" (no queue to keep alive on any backend).
        self._pq_active: bool = False

        # Suction request coalescing: suction_on/off record the REQUESTED state
        # under this lock and no-op on duplicates (telemetry stamp included) —
        # the hardware IO worker also resets `_io_requested_value` to None after
        # a failed write so a later same-state call can retry (it shares these
        # exact attribute names).
        self._io_state_lock = threading.Lock()
        self._io_requested_value: int | None = None

    # ------------------------------------------------------------------
    # Liveness hook — hardware overrides with rclpy.ok() so a ROS shutdown
    # breaks the paced loops; the sim has no external liveness source.
    # ------------------------------------------------------------------
    def _ok(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # 500 Hz stream engine (moved verbatim from trajectory_controller;
    # only rclpy.ok() -> self._ok() and the JGPC publish -> _emit_sample)
    # ------------------------------------------------------------------
    def _stream_trajectory(self, traj, vel, timestep, final_joint=None, *,
                           release_index: int | None = None,
                           suction_on_at: float | None = None,
                           tick_fn=None,
                           suction_events=None,
                           start_mono: float | None = None,
                           catchup: float | None = None,
                           vel_limit=None,
                           reanchor_s: float = 0.1,
                           prev_emit=None) -> dict:
        """Stream a trajectory at ~500 Hz through :meth:`_emit_sample`.

        The app's trajectories are sampled at ~20 Hz (``trajectory()``: ``timestep`` is the
        cumulative time_from_start = arange(L+1)/hertz). The 500 Hz control loop wants a
        fresh position every 2 ms, so we RESAMPLE onto a 2 ms grid and emit each sample,
        paced to wall clock.

        ⚠️ SOFT real-time: this is a Python paced loop (GIL / scheduler jitter), NOT hard 500 Hz.
        The controller runs the true 500 Hz (2 ms); the receiver zero-order-holds our last command
        between emits, so jitter degrades smoothness, not safety.

        ``release_index`` → fire suction_off when the stream reaches that waypoint's time.
        ``suction_on_at`` (wall clock) → after release, fire suction_on once to prime the next pick.
        ``tick_fn`` → called every sample (e.g. to prime suction on a positioning move).
        ``suction_events`` → [(t_traj, on), ...] fired when the streamed sample time reaches t_traj
        (2026-09-21, sequence_app full plans with several ON/OFF). Tied to the SAMPLE clock, not wall
        time, so loop jitter cannot move a release to a different point of the arc.
        Returns {'fired': bool, 'primed_next': bool}.

        Absolute pacing (2026-10-27, sequence_app.execute_plan): ``start_mono`` = monotonic due time of the first sample on the
        caller's plan clock (waits for it if early). With ``catchup`` set, a late stream is caught up gradually instead of in a
        burst: the gap between two samples may shrink to STREAM_DT / (1 + catchup) but never so far that the commanded joint
        speed exceeds ``vel_limit`` (rad/s) × 0.95 — fast arcs run at the datasheet limit, so catch-up happens in slow parts
        (hover, descent, pick). A lag above ``reanchor_s`` is accepted instead (the plan clock shifts; returned as st['shift_s']).
        In this mode the stream does not sleep after its last sample — the next call's ``start_mono`` paces the boundary, and
        ``prev_emit`` = st['last_emit'] of the previous call ((monotonic time, joint sample)) keeps the same gap limit across it.
        Without ``catchup`` the old behaviour is kept (clock starts at the call, late samples go out back to back).
        """
        arr = np.asarray(traj, dtype=float)                       # (n_joints, n_steps)
        varr = np.asarray(vel, dtype=float)                       # velocity profile (same shape)
        n_joints, n_steps = arr.shape
        times = np.asarray(timestep, dtype=float).ravel()         # cumulative time_from_start
        t_release = float(times[int(release_index)]) if release_index is not None else None
        if final_joint is not None:
            arr = np.concatenate(
                [arr, np.asarray(final_joint, dtype=float).reshape(n_joints, 1)], axis=1)
            varr = np.concatenate([varr, np.zeros((n_joints, 1))], axis=1)   # settle at rest
            times = np.append(times, times[-1] + 0.05)
        T = float(times[-1])
        # 정수 개수로 격자를 만든다: np.arange(0, T+dt, dt) 는 부동소수 때문에 T 가 격자 위여도 8% 의 길이에서 한 점이 더 생겨
        # on_grid 판정이 깨지고 불필요한 리샘플(보간)이 일어났다 (2026-09-22 시뮬 'wait' 세그먼트 경고). floor+eps 라 격자 밖 T 도 안전.
        grid = np.arange(int(np.floor(T / STREAM_DT + 1e-6)) + 1) * STREAM_DT
        # 격자 정렬 궤적 (2026-09-21 사용자: 보간 제거): 웨이포인트가 정확히 k·2 ms 에 놓여 있으면 그 값을 그대로
        # 내보낸다 — Hermite 도 클램프도 타지 않는다. sequence_app 의 FullPlan 과 NLP ThrowPlan 이 이 형식이다.
        on_grid = (len(times) == len(grid) and np.abs(times - grid).max() < 1e-6)
        if on_grid:
            samples = np.ascontiguousarray(arr.T)
        # Resample onto the 2ms grid HONORING the knot velocities (#R3: a plain linear resample of
        # the coarse ~20Hz arc yields chord-slope velocity, not the designed NN release speed —
        # matters because throw range ~ v^2). Cubic-Hermite uses positions+velocities; fall back to
        # linear if times aren't strictly increasing (degenerate/segment-join) or scipy is missing.
        elif varr.shape == arr.shape and np.all(np.diff(times) > 0):
            try:
                from scipy.interpolate import CubicHermiteSpline
                samples = np.column_stack(
                    [CubicHermiteSpline(times, arr[j], varr[j])(grid) for j in range(n_joints)])
                # SAFETY: cubic-Hermite can OVER/UNDERSHOOT beyond the knot values — a descent
                # decelerating to belt contact dips BELOW the final pose. The receiver is a POSITION
                # controller, so a below-surface command drives the tool INTO the belt -> shock
                # (alarm 4315). Clamp each sample to the envelope of its two bracketing knots:
                # keeps the cubic velocity shape WITHIN bounds but never past a commanded waypoint
                # (linear never overshoots; this makes cubic equally floor-safe).
                seg = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 2)
                lo = np.minimum(arr[:, seg], arr[:, seg + 1]).T   # (n_grid, n_joints)
                hi = np.maximum(arr[:, seg], arr[:, seg + 1]).T
                samples = np.clip(samples, lo, hi)
            except Exception:
                samples = np.column_stack([np.interp(grid, times, arr[j]) for j in range(n_joints)])
        else:
            samples = np.column_stack([np.interp(grid, times, arr[j]) for j in range(n_joints)])

        st = {"fired": False, "primed_next": False, "resampled": not on_grid, "n_samples": int(samples.shape[0]),
              "ev_fired": 0, "ev_total": len(suction_events or [])}   # 2026-09-27: 못 쏜 이벤트를 호출자가 알 수 있게
        events = sorted(((float(t), bool(on)) for t, on in (suction_events or [])), key=lambda e: e[0])
        ev_i = 0
        vl = None if vel_limit is None else np.maximum(np.asarray(vel_limit, dtype=float).ravel()[:samples.shape[1]], 1e-9)
        inc = None if vl is None else vl * STREAM_DT                  # 컨트롤러 한 주기 증분 한계 [rad]
        n_win = 0                                                     # 한 주기 증분 감시로 늦춘 샘플 수

        def _gaps(dq):
            """(목표 간격, 실제 간격 하한): 목표 = 계획보다 최대 (1+catchup) 배 빠르게·속도 한계 95 % 안 (늦을 때 따라잡는 일정),
            하한 = 속도 한계 100 %·0.5 STREAM_DT — sleep 이 늦게 깨 다음 간격이 줄어도 이 아래로는 안 보낸다."""
            tg = np.full(dq.shape[0], STREAM_DT / (1.0 + float(catchup)))
            lo = np.full(dq.shape[0], 0.5 * STREAM_DT)
            if vl is not None:
                need = np.max(np.abs(dq) / vl, axis=1)
                tg = np.maximum(tg, need / 0.95); lo = np.maximum(lo, need)
            return np.minimum(tg, STREAM_DT), np.minimum(lo, STREAM_DT)
        sched = None
        if start_mono is not None:
            t_first = sched = float(start_mono)
            if catchup is not None and prev_emit is not None:    # 앞 세그먼트 마지막 샘플과의 간격도 같은 한계
                g, lo0 = _gaps((samples[0] - np.asarray(prev_emit[1], dtype=float))[None, :])
                sched = max(sched, float(prev_emit[2]) + float(g[0]))
                t_first = max(sched, float(prev_emit[0]) + float(lo0[0]))
            if inc is not None:
                t_w = self._win_time(samples[0], t_first, inc)
                n_win += int(t_w > t_first + 1e-9); t_first = t_w
            _sleep_until(t_first)
            start = float(start_mono)
        else:
            start = time.monotonic()      # pacing clock (monotonic)
        wall_start = time.time()      # telemetry clock (wall) — last_throw must stay wall-clock (#3)
        pace = catchup is not None
        shift = 0.0
        if pace:
            min_int, lo_int = _gaps(np.diff(samples, axis=0)) if samples.shape[0] > 1 else (np.zeros(0), np.zeros(0))
            if sched is None:
                sched = start
        n_s = samples.shape[0]
        t_emit, k_last = None, -1
        for k in range(n_s):
            if not self._ok():
                break
            t_go = time.monotonic()
            self._emit_due = sched if pace else (start + k * STREAM_DT)   # 이 샘플의 예정 송신 시각 (시뮬 명령 시각 표시용)
            self._emit_tracked(samples[k])
            t_emit = time.monotonic()                             # 실제 송신이 끝난 시각 — 다음 간격 하한의 기준 (송신이 막혔다 늦게 나가도 다음 샘플이 바짝 붙지 않게)
            k_last = k
            if pace and t_go - sched > 3e-4:                      # 일정보다 0.3 ms 넘게 늦게 나감(멈춤) → 일정을 실제로 당긴다 (그 늦음은 catchup 비율로만 갚는다;
                sched = t_go                                       #  0.3 ms 이하는 sleep 이 늦게 깬 몫이라 다음 간격에서 바로 보정)

            while ev_i < len(events) and grid[k] >= events[ev_i][0]:     # 샘플 시각 기준 suction 이벤트
                (self.suction_on if events[ev_i][1] else self.suction_off)()
                st.setdefault("ev_log", []).append((events[ev_i][0], bool(events[ev_i][1]),
                                                    float(grid[k]), time.time()))
                ev_i += 1
                st["ev_fired"] = ev_i

            if (t_release is not None and not st["fired"] and grid[k] >= t_release):
                t_io = time.time()
                self.suction_off()
                st["fired"] = True
                self.last_throw = {
                    "throw_start": wall_start, "release_wall": self.last_suction_off_t,
                    "io_ms": (time.time() - t_io) * 1000.0,
                    "release_index": int(release_index), "n_waypoints": int(n_steps),
                }
            elif (st["fired"] and suction_on_at is not None and not st["primed_next"]
                  and time.time() >= suction_on_at):
                self.suction_on()
                st["primed_next"] = True
            if tick_fn is not None:
                # Existing callbacks return None and continue as before.  An
                # explicit False is reserved for supervised debug motions that
                # must stop at the current streamed command (for example the
                # suction attachment-range measurement).
                if tick_fn() is False:
                    st["stopped_early"] = True
                    st["stop_index"] = int(k)
                    break

            if pace:
                if k + 1 >= n_s:
                    break                                         # 경계 박자는 다음 호출의 start_mono 가 맞춘다
                due = start + shift + (k + 1) * STREAM_DT
                nxt = max(due, sched + float(min_int[k]))          # 늦었으면 일정 위에서 조금씩 따라잡는다 (늦게 깬 몫이 쌓이지 않게)
                if nxt - due > reanchor_s:                         # 너무 늦음 → 따라잡지 않고 시간축을 민다
                    shift += nxt - due - reanchor_s
                sched = nxt
                t_ok = max(nxt, t_emit + float(lo_int[k]))                # 실제 간격은 속도 한계 100 % 아래로
                if inc is not None:                                    # + 어느 2 ms 구간에도 증분 합이 한 주기 한계 안
                    t_w = self._win_time(samples[k + 1], t_ok, inc)
                    n_win += int(t_w > t_ok + 1e-9); t_ok = t_w
                t_wake = t_ok
            else:
                # pace to the next 2 ms tick (drop no samples; sleep the remainder)
                t_wake = start + (k + 1) * STREAM_DT
            _sleep_until(t_wake)
        st["shift_s"] = shift
        st["win_delayed"] = n_win
        st["end_due"] = start + shift + n_s * STREAM_DT              # 다음 샘플(다음 세그먼트 첫 샘플)의 마감
        if pace and k_last >= 0:
            st["last_emit"] = (t_emit, samples[k_last].copy(), sched)
        return st

    def _stream_hold(self, hold_joint, deadline_wall: float, tick_fn=None) -> None:
        """Keep emitting ``hold_joint`` at ~500 Hz until wall-clock ``deadline_wall``
        (feeds the servo during an ambush hold). tick_fn runs each cycle (e.g. prime suction)."""
        pose = [float(x) for x in hold_joint]
        nxt = time.monotonic()
        while time.time() < deadline_wall and self._ok():
            self._emit_tracked(pose)
            if tick_fn is not None:
                tick_fn()
            nxt += STREAM_DT
            dt_sleep = nxt - time.monotonic()
            if dt_sleep > 0:
                time.sleep(dt_sleep)
            else:
                nxt = time.monotonic()

    # ------------------------------------------------------------------
    # send_trajectory family (stream semantics; the hardware subclass
    # overrides these to branch to its JTC action path when configured)
    # ------------------------------------------------------------------
    def send_trajectory(self, traj, vel, timestep, final_joint=None) -> bool:
        self._stream_trajectory(traj, vel, timestep, final_joint)
        return True

    def send_trajectory_queue(self, traj, vel, timestep, final_joint=None) -> bool:
        """(MotoROS2 name kept.) Blocks for the whole motion."""
        self._stream_trajectory(traj, vel, timestep, final_joint)
        return True

    def send_trajectory_queue_interruptible(
        self, traj, vel, timestep, final_joint, stop_requested,
    ) -> bool:
        """Stream until ``stop_requested()`` becomes true.

        Returns True when the motion stopped early.
        """
        state = self._stream_trajectory(
            traj, vel, timestep, final_joint,
            tick_fn=lambda: not bool(stop_requested()),
        )
        return bool(state.get("stopped_early", False))

    def send_trajectory_queue_with_timed_release(self, traj, vel, timestep, final_joint,
                                                 release_index: int,
                                                 suction_on_at: float | None = None) -> bool:
        """Throw path: fire suction_off when the stream reaches the release waypoint's grid
        time, then (optionally) prime the next pick's vacuum once ``suction_on_at`` passes.
        Returns True iff the next-pick suction was primed (NOT success — call sites discard it).
        """
        arr = np.asarray(traj)
        rel = int(max(0, min(release_index, arr.shape[1] - 1)))
        st = self._stream_trajectory(traj, vel, timestep, final_joint,
                                     release_index=rel, suction_on_at=suction_on_at)
        if not st["fired"]:
            self.suction_off()   # fallback: never carry the object past release
        self._last_throw_ok = True   # stream emit never "rejects" (#R3)
        return st["primed_next"]

    def send_trajectory_queue_timed_suction(self, traj, vel, timestep, final_joint,
                                            suction_on_at: float) -> bool:
        """Positioning move that fires suction_ON once wall-clock ``suction_on_at`` passes
        (priming the vacuum before grasp). Returns True iff suction fired before the move ended."""
        fired = {"v": False}

        def _tick() -> None:
            if not fired["v"] and time.time() >= suction_on_at:
                self.suction_on()
                fired["v"] = True

        self._stream_trajectory(traj, vel, timestep, final_joint, tick_fn=_tick)
        if not fired["v"] and time.time() >= suction_on_at:
            self.suction_on()
            fired["v"] = True
        return fired["v"]

    # ------------------------------------------------------------------
    # Queue-mode shims (no-ops on any modern backend — kept for call-site
    # stability with the MotoROS2-era app/skills)
    # ------------------------------------------------------------------
    def enter_queue_mode(self) -> bool:
        """No-op: the backend is always ready (no Point Queue Mode)."""
        return True

    def exit_queue_mode(self) -> bool:
        """No-op."""
        return True

    # ------------------------------------------------------------------
    # Persistent-session shims (pq_*). No queue keep-alive on any backend,
    # so these are thin: a segment is one stream; a hold feeds the servo.
    # ------------------------------------------------------------------
    @property
    def pq_active(self) -> bool:
        """True while a persistent session is open. **@property** (read as an attribute,
        not called) — a plain method would evaluate truthy always and defeat the
        session-live guard (#A)."""
        return self._pq_active

    def pq_begin(self) -> None:
        self._pq_active = True

    def pq_segment(self, traj, vel, ts, final_joint, *, is_last: bool = False,
                   join_tol: float = 0.05, between_fn=None) -> tuple:
        """Execute one segment; call ``between_fn`` each cycle (used to fire suction_on
        when its wall-clock instant passes). Returns (ok, codes)."""
        self._stream_trajectory(traj, vel, ts, final_joint, tick_fn=between_fn)
        return (True, [])

    def pq_hold_until(self, hold_joint, deadline_wall: float, *,
                      dt: float = 0.12, lead: float = 0.35, tick_fn=None) -> bool:
        """Wait until wall-clock ``deadline_wall`` at the hold pose, feeding the servo;
        ``tick_fn`` runs each cycle (e.g. to prime suction on time)."""
        self._stream_hold(hold_joint, deadline_wall, tick_fn=tick_fn)
        return True

    def pq_throw_segment(self, traj, vel, timestep, final_joint, *,
                         release_index, suction_on_at=None,
                         release_tol: float = 0.05) -> tuple:
        """Throw with a timed suction release (same logic as
        send_trajectory_queue_with_timed_release). Returns ``(ok, primed_next)`` — a 2-tuple to
        match the caller's ``ok_throw, primed_next = ...`` unpack (throw_skill.py) and the sibling
        pq_segment's (ok, ...) contract (#1)."""
        primed_next = self.send_trajectory_queue_with_timed_release(
            traj, vel, timestep, final_joint, release_index, suction_on_at=suction_on_at,
        )
        return (self._last_throw_ok, bool(primed_next))

    def pq_finish(self, *, wait: bool = True, tail_buffer: float = 0.3,
                  settle_tol: float = 0.03) -> None:
        """End the session. (Each segment already blocks, so nothing to drain.)"""
        self._pq_active = False

    # ------------------------------------------------------------------
    # Waiting helpers
    # ------------------------------------------------------------------
    def _wait_for_position(self, target_joint, tolerance: float = 1e-4,
                           timeout_sec: float = 5.0) -> bool:
        """Block until the arm is within ``tolerance`` (max-abs) of ``target_joint``.
        (Private name kept — push_skill2 / push_skill_floor_gated call it.)"""
        target = [float(x) for x in target_joint]
        t_end = time.time() + timeout_sec
        while time.time() < t_end and self._ok():
            if self.current_joints is not None and _max_abs_diff(self.current_joints, target) <= tolerance:
                return True
            time.sleep(0.005)
        return False

    def _wait_trajectory_end(self, total_duration: float, t_start: float | None = None) -> None:
        t0 = t_start if t_start is not None else time.time()
        remaining = total_duration - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Suction — CONCRETE: coalescing + the telemetry every skill's
    # diagnostics read; the physical toggle is the backend's _set_suction.
    # ------------------------------------------------------------------
    def suction_on(self) -> None:
        """Request suction ON once; duplicate requested states are coalesced."""
        requested_at = time.time()
        with self._io_state_lock:
            if self._io_requested_value == 0:
                return
            self._io_requested_value = 0
        self.last_suction_on_t = requested_at
        # Snapshot the joint state AT the fire instant (this may run inside the
        # stream loop mid-move) so callers can FK the true EE pose the vacuum
        # fired at. list() to freeze it against a concurrent state update.
        self.last_suction_on_joints = (
            list(self.current_joints) if self.current_joints is not None else None
        )
        self._set_suction(True, requested_at)

    def suction_off(self) -> None:
        """Request suction OFF once; duplicate requested states are coalesced."""
        requested_at = time.time()
        with self._io_state_lock:
            if self._io_requested_value == 1:
                return
            self._io_requested_value = 1
        self.last_suction_off_t = requested_at
        self._set_suction(False, requested_at)

    # ------------------------------------------------------------------
    # Lifecycle — no-ops by default; hardware overrides.
    # ------------------------------------------------------------------
    def set_motion_op(self, label: str) -> None:
        """Tag the current cycle's motion with a skill label (push/throw) for
        diagnostic logging. No-op by default; a backend with a motion logger
        overrides it."""
        return None

    def close(self, timeout_sec: float = 5.0) -> None:
        """Release backend resources on shutdown. No-op by default."""
        return None

    # ------------------------------------------------------------------
    # Abstract seam — the ONLY things a backend must implement.
    # ------------------------------------------------------------------
    @abstractmethod
    def wait_for_servers(self, timeout_sec: float = 10.0) -> bool:
        """Block until the backend can accept motion (action server up / model
        loaded + stepper running). Return True on ready, False on timeout. The
        app calls this once in ``setup()`` before it starts the control loop."""
        raise NotImplementedError

    def track_start(self) -> None:
        """실행 추적 시작 (버퍼 비움)."""
        self.track_buf = []

    def track_stop(self) -> list:
        """추적 종료 → 수집된 버퍼 반환."""
        buf, self.track_buf = (self.track_buf or []), None
        return buf

    def _emit_tracked(self, positions) -> None:
        """_emit_sample + (추적 중이면) 명령/측정 한 줄 기록. 500 Hz 루프 안에서 돌므로
        append 한 번 외에는 아무것도 하지 않는다. track_trip_rad 가 있으면 보내기 전에 |명령 − 컨트롤러 목표| 를 보고, TRIP_N 샘플 연속
        넘으면 이 샘플을 보내지 않고 TrackingFault (호출측이 그 자리 정지)."""
        lim = self.track_trip_rad
        if lim is not None:
            cur = self.current_joints
            if cur:
                e = max(abs(float(a) - float(b)) for a, b in zip(positions, cur))
                if e > lim:
                    self._trip_n += 1
                    if self._trip_n >= TRIP_N:
                        self._trip_n = 0
                        raise TrackingFault(f"추종 오차 {np.degrees(e):.1f}° > {np.degrees(lim):.0f}° ({TRIP_N} 샘플 연속) — 명령 {np.round(np.degrees(positions), 1).tolist()}°, "
                                            f"컨트롤러 목표 {np.round(np.degrees(cur), 1).tolist()}°")
                else:
                    self._trip_n = 0
        self._emit_sample(positions)
        self._win_hist.append((time.monotonic(), positions))
        buf = self.track_buf
        if buf is not None:
            try:
                fdb = getattr(self, "current_joints_fdb", None)
                buf.append((time.monotonic(), tuple(float(x) for x in positions),
                            tuple(self.current_joints) if self.current_joints else None,
                            tuple(self.current_jointvels) if self.current_jointvels else None,
                            tuple(fdb) if fdb else None))
            except Exception:                      # noqa: BLE001 — 추적이 제어를 방해하면 안 된다
                pass

    def _win_time(self, q, t0: float, inc) -> float:
        """한 주기 증분 감시 (11-03 실물 CODE 112): 컨트롤러는 자기 2 ms 주기마다 그때까지 받은 마지막 목표로의 증분만 보고, 그 증분이
        한계(속도 한계 × 2 ms)를 넘으면 INC_MOVE_ERROR 로 선다. Python 송신 간격은 1.9~2.6 ms 로 흔들려, 늦었다 따라잡을 때 두 샘플이
        한 주기에 들어가면 속도 100 % 구간에서 증분이 최대 2 배가 됐다. 송신 시각 t 에서 't − 2 ms 까지 마지막으로 보낸 명령'과 q 의 차가
        관절마다 inc 안이면 주기 위상과 무관하게 안전 — 그렇게 되는 가장 이른 t ≥ t0 를 준다 (늦어진 몫은 느린 구간에서 따라잡는다).
        샘플 하나로 이미 넘으면 (계획이 한계 밖) 마지막 송신 + 2 ms 에 그대로 보낸다 (_win_over 로 센다)."""
        h = self._win_hist
        if not h or inc is None:
            return t0
        q = np.asarray(q, dtype=float)
        lim = np.asarray(inc, dtype=float) * (1.0 + 1e-6)
        for t in [t0] + [te + STREAM_DT for te, _ in h if te + STREAM_DT > t0]:
            ref = h[0][1]
            for te, qe in h:
                if te <= t - STREAM_DT + 1e-9:
                    ref = qe
                else:
                    break
            if np.all(np.abs(q - np.asarray(ref, dtype=float)) <= lim):
                return t
        self._win_over += 1
        return h[-1][0] + STREAM_DT

    @abstractmethod
    def _emit_sample(self, positions: Sequence[float]) -> None:
        """Deliver ONE 2 ms joint-position command [S,L,U,R,B,T] rad.

        Hardware publishes a ``Float64MultiArray`` to the JGPC command topic;
        the twin writes the position-actuator ctrl targets. MUST be fast and
        non-blocking — it runs inside the 500 Hz paced loop.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_suction(self, on: bool, requested_at: float) -> None:
        """Physically toggle the vacuum: True = grip, False = release.

        Called only on requested-STATE CHANGES (coalescing happens in
        suction_on/off above). Hardware enqueues a Simple-Message TCP write on
        its IO worker; the twin welds/releases the nearest box. MUST be
        non-blocking enough not to stall the 500 Hz loop at the release
        instant. Errors are logged, never raised — a killed drive loop is worse
        than a missed toggle (though a failed *release* is a real safety
        concern to catch in bring-up).
        """
        raise NotImplementedError
