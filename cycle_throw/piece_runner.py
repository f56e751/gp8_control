"""Piece mode (GP8_MODE=pieces): skills plan pieces, one realized timeline runs them.

The app decides, object by object and in belt order, what the arm does next
(``skill_for``, the ``ActionSelector`` seam that an RL policy replaces later).
Each skill returns a :class:`~gp8_control.skills.base.MotionPiece` instead of
moving the arm:

  * throw (PlannerThrowSkill): what to throw into which bin; the teammate's
    cycle planner computes the motion,
  * push  (PushSkill.plan_piece): the stroke commands as the plain stream would
    send them from the backswing, and their fire time.

They go to ``cycle_throw/piece_link.py`` (robot6) as one list of steps:
``throw`` (the held object), then the pushes due before the next throw object,
then ``pick`` of that object. The link plans all of it as ONE planned realized
motion: moves between steps leave from the state the arm is in (moving or
not), a push is the controller filter applied to its stroke commands (what the
real arm does with them today), and one ``rt_inv`` fit makes the commands. No
rest hand-off, settle hold, controller switch or command smoothing between a
throw and a push. Command output, suction timing, the remote link and clock
sync are the cycle runner's (``runner.CycleThrowRunner``).

Steps are decided ``RT_AHEAD`` s before the held object's throw cycle starts
(as rt_cont chooses its next object), or at once when the arm is at rest.
"""

from __future__ import annotations

import glob
import math
import os
import queue
import signal
import threading
import time

import numpy as np

from gp8_control.cycle_throw.runner import (
    PLANNER_BELT_SPEED,
    CycleThrowRunner,
    _env_float,
)
from gp8_control.skills.base import MotionPiece
from gp8_control.skills.context import PickRequest
from gp8_control.tracking.object_queue import is_established


class PieceRunner(CycleThrowRunner):
    LINK = "piece_link.py"

    def __init__(self, app, *, ctx, skills: dict, skill_for, top_z: float | None = None,
                 on_execute=None, vtime=None, decide_fn=None) -> None:
        """``skills``: skill name -> skill with ``plan_piece`` ("throw" ->
        PlannerThrowSkill, "push" -> PushSkill). ``on_execute(obj, skill)`` runs
        when the link has planned ``obj`` (the sim marks it manipulated)."""
        super().__init__(app, skill_for=skill_for, push_handler=None, top_z=top_z, vtime=vtime)
        self.ctx = ctx
        self.skills = dict(skills)
        self.on_execute = on_execute
        self.arm: dict | None = None
        self._replies: "queue.Queue" = queue.Queue()
        self._pending: dict | None = None
        self._rid = 0
        self._taken: set[int] = set()
        self._retry_at: dict[int, float] = {}
        self._skips: dict[int, int] = {}
        self._reroute: set[int] = set()
        self.log_rows: list[dict] = []
        self.ahead = _env_float("RT_AHEAD", 3.0)
        # Pushes per request: a run of cans goes into ONE request, so the arm
        # flows from can to can instead of ending at rest after each request.
        # Two per request: the next request carries on from the open end, so
        # more buys nothing and costs planning time before the request goes.
        self.push_max = int(_env_float("GP8_PIECE_PUSH_MAX", 2))
        # A push the skill or the link refuses (squeezed backswing, too late
        # twice) is thrown instead when the throw skill takes the class.
        self.push_fallback = os.environ.get("GP8_PIECE_PUSH_FALLBACK", "throw").strip().lower()
        self.vy = -PLANNER_BELT_SPEED
        self.r_reach = float(self.cfg.MAX_REACH)
        self.H = None                                  # the planner's travel height (link "ready")
        # decide_fn(runner, candidates, t_from) -> (item, "throw" | "push") | None:
        # what to do next, instead of the belt-order rule (an RL policy). One
        # object per request; None = nothing now (asked again after DECIDE_WAIT).
        self.decide_fn = decide_fn
        self.decide_wait = _env_float("GP8_PIECE_DECIDE_WAIT", 0.2)
        # Audit of every policy decision: what was chosen, what piece mode did
        # with it (planned as chosen / the documented fallbacks), whether the
        # link then took it. tools/piece_twin and the app log it.
        self.decisions: list[dict] = []
        self._ask_after = 0.0
        # Virtual time: idle stretches are warped over in steps of WARP_S (the
        # link commits whatever falls due first); planning runs at wall speed.
        self.warp_s = _env_float("GP8_PIECE_WARP_S", 0.05)
        self._warp_seq = 0
        self._warped = threading.Event()
        self._warp_ack = -1
        self._vready = False
        self._link_paused = threading.Event()

    # ------------------------------------------------------------------
    def run(self, intake=None, ok=None) -> None:
        if ok is None:
            import rclpy

            def ok():
                return rclpy.ok() and not self.app._spinner_dead

        self._start_planner()
        self.H = float((self._ready_msg or {}).get("H", 0.12))
        self.arm = {"kind": "rest", "q": self.home.tolist(), "t_free": 0.3, "t_rest": 0.0}
        period = 1.0 / max(1.0, _env_float("GP8_PIECE_DECIDE_HZ", 50.0))
        last_report = self.mono()
        try:
            while ok() and self._fault is None and not self._done.is_set():
                if intake is not None:
                    intake()
                self._record_tracks()
                self._take_replies()
                if self._pending is None:
                    self._decide()
                if self.mono() - last_report > 10.0:
                    last_report = self.mono()
                    self.log.info(f"[pieces] {dict(self.stats)} arm {self._arm_str()}")
                self._advance(period)
        finally:
            self.close()
        if self._fault:
            raise RuntimeError(f"piece mode stopped: {self._fault}")

    def close(self) -> None:
        """No new step; the link throws a held object; let everything play out."""
        if self._child is None:
            return
        self._send({"t": "stop"})
        deadline = self.mono() + _env_float("GP8_CYCLE_DRAIN_S", 12.0)
        while self.mono() < deadline and self._fault is None and not self._done.is_set():
            self._take_replies()
            self._sleep(0.05)
        while self.mono() < deadline and self._fault is None and not self._out_idle(0.2):
            self._sleep(0.05)
        if not self._done.is_set() and self._fault is None:
            self.log.warn("[pieces] stop: the link did not finish in time; holding where the arm is")
        self._teardown()

    # ------------------------------------------------------------------
    # link replies
    # ------------------------------------------------------------------
    def _on_msg(self, kind, msg: dict) -> None:
        if kind == "planned":
            self._replies.put(msg)
        elif kind == "vready":
            self._vready = True
        elif kind == "warped":
            self._warp_ack = int(msg.get("seq", -1))
            self._warped.set()
        elif kind == "paused":
            self._link_paused.set()

    # ------------------------------------------------------------------
    # time
    # ------------------------------------------------------------------
    def _advance(self, period: float) -> None:
        """Let ``period`` pass: wall time, or (virtual) a warp while the link
        computes nothing for us."""
        if self.vt is None or self._pending is not None or self.t0 is None or not self._vready:
            self._sleep(period)
            return
        self._warp_seq += 1
        seq = self._warp_seq
        self._warped.clear()
        self._send({"t": "warp", "dt": float(period), "seq": seq})
        t_end = time.monotonic() + 10.0
        while self._warp_ack < seq:
            if self._fault is not None:
                return
            if not self._warped.wait(0.5) and time.monotonic() > t_end:
                self._fault = "virtual time: the link did not ack a warp"
                return
            self._warped.clear()
        if self.out is not None:
            self.out._q.join()            # every command the link sent before the ack is written
        self.vt.warp(period)

    def _take_replies(self) -> None:
        while True:
            try:
                msg = self._replies.get_nowait()
            except queue.Empty:
                return
            p = self._pending
            if p is None or msg.get("rid") != p["rid"]:
                self.log.warn(f"[pieces] unexpected reply {msg}")
                continue
            self._pending = None
            self.arm = msg["arm"]
            taken_ids = {int(t) for t in msg.get("taken", [])}
            if self.arm.get("kind") == "obj":
                taken_ids.add(int(self.arm["id"]))
            skipped_ids = {int(t): why for t, why in msg.get("skipped", [])}
            for d in self.decisions:
                if d.get("rid") == p["rid"] and d.get("result") in (
                        "planned push", "planned throw", "push refused by the skill -> throw"):
                    d["link"] = ("took it" if d["id"] in taken_ids
                                 else f"skipped ({skipped_ids[d['id']]})" if d["id"] in skipped_ids
                                 else "not taken")
                    if self.decide_fn is not None and hasattr(self.decide_fn, "log") and self.decide_fn.log:
                        self.decide_fn.log(f"[policy] result: {d['chose']} {d['id']} -> "
                                           f"{d['result']}, link {d['link']}")
            for tid in msg.get("taken", []):
                self._accept(int(tid), p["kinds"].get(int(tid), "throw"))
            for tid, why in msg.get("skipped", []):
                tid = int(tid)
                self._skips[tid] = self._skips.get(tid, 0) + 1
                self._retry_at[tid] = self.mono() + 0.5
                self.stats["skipped"] += 1
                self.log.info(f"[pieces] id={tid} skipped by the link: {why}")
                if p["kinds"].get(tid) == "push" and self._skips[tid] >= 2:
                    obj = self._find(tid)
                    if obj is not None:
                        self._fallback(obj, f"push skipped twice ({why})")
            self.stats["plans_ok" if msg.get("ok") else "plans_refused"] += 1
            if not msg.get("ok"):
                self.log.warn(f"[pieces] plan refused: {msg.get('msg')}")
            self.log_rows.append(dict(t=round(self.now(), 3), ok=bool(msg.get("ok")),
                                      steps=p["desc"], taken=msg.get("taken"),
                                      skipped=msg.get("skipped"), arm=self.arm,
                                      plan_s=round(self.mono() - p["t_sent"], 3)))

    def _find(self, track_id: int):
        return next((o for o in self.app.queue.snapshot() if int(o.track_id) == track_id), None)

    def _accept(self, track_id: int, skill: str) -> None:
        if track_id in self._taken:
            return
        self._taken.add(track_id)
        self.stats[f"{skill}_planned"] += 1
        obj = self._find(track_id)
        if obj is None:
            return
        if skill == "push":
            self.app.queue.remove(obj)
        if self.on_execute is not None:
            self.on_execute(obj, skill)

    def _arm_str(self) -> str:
        a = self.arm or {}
        if a.get("kind") == "obj":
            return f"on object {a.get('id')} (contact {a.get('tc', 0):.2f})"
        return f"rest from {a.get('t_free', 0):.2f}"

    # ------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------
    def _items(self) -> list[dict]:
        """Routed, established, untaken objects with their planner-frame dicts."""
        tw = self._wall()
        tm = self.now()
        v = float(self.app.conveyor.current)
        dist = self.app.conveyor.distance_m
        mono = self.mono()
        out = []
        for obj in self.app.queue.snapshot():
            tid = int(obj.track_id)
            if tid in self._taken or self._retry_at.get(tid, 0.0) > mono:
                continue
            if not is_established(obj, self.cfg.TRACK_MIN_HITS):
                continue
            skill = self.skill_for(obj) if self.skill_for is not None else "throw"
            if tid in self._reroute:
                skill = "throw"
            if skill not in self.skills:
                continue
            y = float(obj.y_at(tw, v, dist))
            if not self.vis_y[0] <= y <= self.vis_y[1]:
                continue
            x = float(obj.T_grasp_base[0, 3])
            o = {"id": tid, "cls": obj.class_name, "x": x, "y0": y + PLANNER_BELT_SPEED * tm}
            if self.top_z == "track":
                o["z"] = float(obj.T_grasp_base[2, 3])
            elif self.top_z:
                o["z"] = float(self.top_z)
            out.append({"obj": obj, "o": o, "skill": skill, "x": x})
        return out

    def _y_at(self, item: dict, t: float) -> float:
        return item["o"]["y0"] + self.vy * t

    def _in_order(self, items: list[dict], t: float, exclude: int | None = None) -> list[dict]:
        """Objects still inside the reach window at time t, most downstream first."""
        keep = []
        for it in items:
            if it["o"]["id"] == exclude:
                continue
            yb = math.sqrt(max(self.r_reach ** 2 - it["x"] ** 2, 0.0))
            if self._y_at(it, t) >= -yb + 0.05:
                keep.append(it)
        return sorted(keep, key=lambda it: self._y_at(it, t))

    def _decide(self) -> None:
        arm = self.arm
        tp = self.now()
        items = self._items()
        if arm["kind"] == "obj":
            ts = float(arm["ts"])
            if tp < ts - self.ahead:
                return
            steps, kinds, desc = self._steps(items, ts + 0.6, exclude=int(arm["id"]))
            self._request([{"kind": "throw"}] + steps, kinds, ["throw"] + desc)
            return
        if not items:
            return
        if self.decide_fn is not None and self.mono() < self._ask_after:
            return
        # The link carries on from the unsent end of its last plan when it can
        # (t_open), else it starts from rest (t_free, plus its command lead).
        t_open = arm.get("t_open")
        t_from = float(t_open) if t_open is not None else max(tp + 1.6, float(arm["t_free"]) + 0.3)
        steps, kinds, desc = self._steps(items, t_from)
        if steps:
            self._request(steps, kinds, desc)
        elif self.decide_fn is not None:
            self._ask_after = self.mono() + self.decide_wait

    def _steps(self, items: list[dict], t_from: float, exclude: int | None = None):
        """Pushes due before the next throw object, then its pick (belt order)."""
        if self.decide_fn is not None:
            return self._steps_decided(items, t_from, exclude)
        steps, kinds, desc = [], {}, []
        t_est = t_from
        n_push = 0
        for it in self._in_order(items, t_from, exclude):
            tid = it["o"]["id"]
            if it["skill"] == "throw":
                piece = self._piece(it, t_est)
                if piece is None:
                    continue
                o = dict(it["o"])
                o["b"] = int(piece.request["bin"])
                steps.append({"kind": "pick", "obj": o})
                kinds[tid] = "throw"
                desc.append(f"pick {tid}")
                break
            if n_push >= self.push_max:
                break
            r = self._push_step(it, t_est)
            if r == "throw":                     # rerouted: throw it as the next pick
                self._reroute.add(tid)
                it = dict(it, skill="throw")
                piece = self._piece(it, t_est)
                if piece is not None:
                    o = dict(it["o"])
                    o["b"] = int(piece.request["bin"])
                    steps.append({"kind": "pick", "obj": o})
                    kinds[tid] = "throw"
                    desc.append(f"pick {tid} (push refused)")
                break
            if r is None:
                continue
            step, t_after = r
            steps.append(step)
            kinds[tid] = "push"
            desc.append(f"push {tid}")
            n_push += 1
            t_est = t_after
        return steps, kinds, desc

    def _steps_decided(self, items: list[dict], t_from: float, exclude: int | None = None):
        """Steps chosen by ``decide_fn`` (an RL policy), one decision at a time,
        bundled like the rule's: up to ``push_max`` pushes, then at most one
        pick. After a push the policy is asked again from the time that push
        frees the arm; a pick (or None) ends the request."""
        steps, kinds, desc = [], {}, []
        chosen = {exclude} if exclude is not None else set()
        t_est = t_from
        n_push = 0
        while True:
            cands = [c for c in self._in_order(items, t_est) if c["o"]["id"] not in chosen]
            choice = self._ask(cands, t_est) if cands else None
            if choice is None:
                break
            it, skill = choice
            tid = it["o"]["id"]
            chosen.add(tid)
            rec = {"n": len(self.decisions) + 1, "id": tid, "chose": skill, "rid": self._rid + 1}
            self.decisions.append(rec)
            if skill == "push":
                if n_push >= self.push_max:
                    rec["result"] = "deferred: push limit of this request"
                    break                     # the next request takes it
                r = self._push_step(it, t_est)
                if r is None:
                    rec["result"] = "push not possible now (asked again)"
                    continue                  # not possible now; ask again
                if r != "throw":
                    step, t_est = r
                    steps.append(step)
                    kinds[tid] = "push"
                    desc.append(f"push {tid} (policy)")
                    rec["result"] = "planned push"
                    n_push += 1
                    continue
                self._reroute.add(tid)        # refused: thrown instead (as the rule does)
                rec["result"] = "push refused by the skill -> throw"
            piece = self._piece(dict(it, skill="throw"), t_est)
            if piece is not None:
                o = dict(it["o"])
                o["b"] = int(piece.request["bin"])
                steps.append({"kind": "pick", "obj": o})
                kinds[tid] = "throw"
                desc.append(f"pick {tid} (policy)")
                rec.setdefault("result", "planned throw")
            else:
                rec["result"] = "no throw piece"
            break
        return steps, kinds, desc

    def _ask(self, cands, t_est):
        """``decide_fn`` with virtual time standing still while it blocks: an RL
        agent in a vec env waits for the other envs, a policy on the robot
        answers in ms. The twin only moves when this thread drives it; the
        link's clock is paused and moved back by the same wait, and the link's
        process tree is stopped (SIGSTOP) so a solve in flight gains no time."""
        if self.vt is None or not self._vready:
            return self.decide_fn(self, cands, t_est)
        self.vt.pause()
        self._link_paused.clear()
        self._send({"t": "pause"})
        stopped = self._signal_planner(signal.SIGSTOP) if self._link_paused.wait(2.0) else []
        try:
            return self.decide_fn(self, cands, t_est)
        finally:
            self._send({"t": "resume", "dt": self.vt.resume()})
            self._signal_planner(signal.SIGCONT, stopped)

    def _signal_planner(self, sig, pids=None) -> list:
        """Send ``sig`` to the link and every process under it (planner server,
        worker pools; read from /proc); returns the pids signalled."""
        if pids is None:
            pids, todo = [], [self._child.pid]
            while todo:
                p = todo.pop()
                pids.append(p)
                for f in glob.glob(f"/proc/{p}/task/*/children"):
                    try:
                        with open(f) as fh:
                            todo += [int(c) for c in fh.read().split()]
                    except OSError:
                        pass
        for p in pids:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        return pids

    def decision_summary(self) -> dict:
        """Counts over the policy decisions (see ``self.decisions``)."""
        out: dict = {}
        for d in self.decisions:
            k = d.get("result", "?")
            if "link" in d or k.startswith("planned"):
                k = f"{k}, link: {d.get('link', 'no reply yet')}"
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    def _piece(self, item: dict, t_start_p: float):
        """The throw skill's piece (planner request) for an object, or None."""
        skill = self.skills["throw"]
        req = PickRequest(target=item["obj"], current_joint=self.home, T_aim=None, T_grasp=None,
                          aim_joint=None, grasp_joint=None, secondary=None)
        piece = skill.plan_piece(req, self._to_wall(t_start_p))
        if not isinstance(piece, MotionPiece):
            self._retry_at[item["o"]["id"]] = self.mono() + 1.0
            return None
        return piece

    def _push_step(self, item: dict, t_start_p: float):
        """("push" step, planner time the arm is free after it), "throw" (reroute) or None."""
        obj = item["obj"]
        tid = int(obj.track_id)
        skill = self.skills[item["skill"]]
        tp = self.now()
        q_from = np.asarray(self.home, float)               # move-time estimate only
        v = float(self.app.conveyor.current)
        it = self.ctx.earliest_reachable_intercept(obj, q_from, v, self.ctx.clock.time(),
                                                   pre_delay=max(0.0, t_start_p - tp), skill=skill)
        if it is None:
            self._taken.add(tid)
            self.app.queue.remove(obj)
            self.stats["push_uncatchable"] += 1
            self.log.info(f"[pieces] push id={tid}: dropped, uncatchable")
            return None
        veto = skill.placement_veto(obj, it)
        if veto is not None:
            self.stats["push_veto"] += 1
            if self._fallback(obj, f"push refused ({veto})"):
                return "throw"
            self._retry_at[tid] = self.mono() + 0.5
            return None
        req = PickRequest(target=obj, current_joint=q_from, T_aim=it.T_aim, T_grasp=it.T_grasp,
                          aim_joint=it.aim_joint, grasp_joint=it.grasp_joint, secondary=None)
        piece = skill.plan_piece(req, self._to_wall(t_start_p))
        if not isinstance(piece, MotionPiece) or piece.kind != "filtered":
            self.stats["push_declined"] += 1
            why = getattr(piece, "detail", piece)
            if self._fallback(obj, f"push declined ({why})"):
                return "throw"
            self._retry_at[tid] = self.mono() + 0.3
            return None
        U = np.asarray(piece.Q, float)
        # Above the backswing and above the stroke end, at the planner's travel
        # height, in the paddle's own orientation (the planner's IK is for its
        # cup-down tool and does not reach every stroke end).
        q_hover = self._above(U[0])
        q_lift = self._above(U[min(int(piece.meta["i_cut"]), len(U) - 1)])
        if q_hover is None or q_lift is None:
            self.stats["push_no_ik"] += 1
            if self._fallback(obj, "no pose above the backswing / stroke end"):
                return "throw"
            self._retry_at[tid] = self.mono() + 0.5
            return None
        t_f = self._to_planner(piece.t0)
        t_after = t_f + piece.meta["i_cut"] * piece.dt + 0.4
        self.log.info(f"[pieces] {piece.detail}; fire at {t_f:.2f} (now {tp:.2f})")
        return ({"kind": "push", "id": tid, "U": U.round(9).tolist(), "t_f": t_f,
                 "i_contact": piece.meta["i_contact"], "i_cut": piece.meta["i_cut"],
                 "q_hover": np.asarray(q_hover, float).tolist(),
                 "q_lift": np.asarray(q_lift, float).tolist()}, t_after)

    def _above(self, q: np.ndarray):
        """The pose ``q`` raised to the planner's travel height, same orientation."""
        robot = self.app.robot
        T = robot.forward_kinematics(np.asarray(q, float)[:6])
        T[2, 3] = self.H
        q_up = robot.inverse_kinematics(T, q_init=np.asarray(q, float)[:6])
        return None if q_up is None else np.asarray(q_up, float)

    def _fallback(self, obj, why: str) -> bool:
        """Route a refused push object to the throw skill, if allowed and able."""
        thrower = self.skills.get("throw")
        if self.push_fallback != "throw" or thrower is None or not thrower.can_handle(obj):
            return False
        self._reroute.add(int(obj.track_id))
        self.stats["push_to_throw"] += 1
        self.log.info(f"[pieces] id={obj.track_id} {obj.class_name}: {why}; thrown instead")
        return True

    def _request(self, steps: list, kinds: dict, desc: list) -> None:
        self._rid += 1
        self._pending = {"rid": self._rid, "kinds": kinds, "desc": desc,
                         "t_sent": self.mono()}
        self._send({"t": "plan", "rid": self._rid, "steps": steps})

    # planner time (monotonic - t0) <-> ctx clock (wall)
    def _wall(self) -> float:
        return self.ctx.clock.time() if self.ctx is not None else time.time()

    def _to_wall(self, t_p: float) -> float:
        return self._wall() + (t_p - self.now())

    def _to_planner(self, t_w: float) -> float:
        return self.now() + (t_w - self._wall())
