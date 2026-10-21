"""Physical-step safety telemetry for the native MuJoCo simulator.

The monitor reads solver contacts, rather than simulated sensors.  Normal
tool--target contact is exempt only during suction pickup/loading or the
protected Push stroke.  Other robot contacts and contacts caused by a handled
or previously disturbed object are counted.  Observing does not alter physics.
"""

from __future__ import annotations

from collections import Counter
import math

import numpy as np


class SafeContactMonitor:
    """Accumulate contact events and collateral falls after every physics step.

    ``force_reader(model, data, contact_index, output6)`` is optional dependency
    injection for tests; ordinary execution uses ``mujoco.mj_contactForce``.
    An event is one geom/spawn pair whose qualifying contact samples have no
    gap greater than ``event_gap``.  Its impulse includes all contact points.
    """

    TOOL_NAMES = frozenset(("pusher_collision", "gripper_geom", "suction_gripper_visual"))

    def __init__(self, core, *, force_threshold=.1, event_gap=.03,
                 fall_depth=.08, force_reader=None):
        if (not math.isfinite(force_threshold) or force_threshold < 0
                or not math.isfinite(event_gap) or event_gap < 0
                or not math.isfinite(fall_depth) or fall_depth <= 0):
            raise ValueError("Safety thresholds must be finite and nonnegative")
        self.core = core
        self.force_threshold = float(force_threshold)
        self.event_gap = float(event_gap)
        self.fall_depth = float(fall_depth)
        self.events: list[dict] = []
        self.falls: list[dict] = []
        self._force_reader = force_reader
        self._force = np.zeros(6, dtype=float)
        self._geom_to_box = {int(g): i for i, g in enumerate(core._box_geom_ids)}
        self._robot_geoms, self._tool_geoms = self._find_robot_geoms()
        self._last_events: dict[tuple, dict] = {}
        self._disturbed_ids: set[int] = set()
        self._collateral_ids: set[int] = set()
        self._collateral_causes: dict[int, dict] = {}
        self._intended_ids: set[int] = set()
        self._fallen_ids: set[int] = set()
        self._impulse = 0.
        self._last_time: float | None = None
        self.normal_contact_samples = 0

    def _find_robot_geoms(self):
        model = self.core.model
        robot, tools = set(), set()
        try:
            root = int(model.body("yaskawa_robot").id)
        except (KeyError, ValueError, AttributeError):
            root = -1
        for gid in range(int(model.ngeom)):
            name = str(model.geom(gid).name or "")
            if name in self.TOOL_NAMES:
                tools.add(gid)
            body = int(model.geom_bodyid[gid])
            while body > 0 and body != root:
                body = int(model.body_parentid[body])
            if (root >= 0 and body == root) or gid in tools:
                robot.add(gid)
        return robot, tools

    def _active(self):
        runner = getattr(self.core, "safe_runner", None)
        target = getattr(runner, "_active_target", None)
        target_id = getattr(target, "sim_object_id", None)
        task = None
        if target is not None and runner is not None:
            try:
                task = runner._skill_name_for_object(target)
            except (AttributeError, KeyError):
                pass
        return runner, None if target_id is None else int(target_id), task

    def phase_context(self):
        """Return task/phase/target metadata at the current physical instant."""
        runner, target_id, task = self._active()
        context = getattr(self.core, "safe_residual_context", None) or {}
        if context:
            target_id = context.get("target_id", target_id)
            target_id = None if target_id is None else int(target_id)
            task = context.get("task", task)
            elapsed = float(self.core.sim_time()) - float(context["start_sim_time"])
            if task == "throw" and context.get("release_time") is not None:
                phase = "throw_loaded" if elapsed <= float(context["release_time"]) + 1e-9 else "throw_return"
            elif task == "push" and context.get("prepare_end") is not None:
                prepare_end, stroke_end = context.get("prepare_end"), context.get("stroke_end")
                if prepare_end is None or elapsed < float(prepare_end) - 1e-9:
                    phase = "push_prepare"
                elif stroke_end is not None and elapsed <= float(stroke_end) + 1e-9:
                    phase = "push_stroke"
                else:
                    phase = "push_transfer"
            else:
                phase = str(context.get("phase", "idle"))
        else:
            status = str(getattr(runner, "_status", "IDLE")).upper()
            grabbed = getattr(self.core, "_grabbed", None)
            if task == "throw":
                if grabbed is not None:
                    phase = "throw_loaded"
                elif bool(getattr(self.core, "_vacuum_on", False)):
                    phase = "throw_pick"
                elif status == "THROWING":
                    phase = "throw_return"
                else:
                    phase = "throw_approach"
            elif task == "push":
                phase = {"POSITIONING": "push_route", "WAITING": "push_wait",
                         "PUSHING": "push_unknown"}.get(status, "idle")
            else:
                phase = "idle"
        # The app chains a Throw pickup into the end of the previous skill.
        # Only the tool touching that declared next Throw target in the final
        # 120 ms of a return/transfer is an intended grasp. Earlier contact,
        # arm-link contact, and a queued Push target remain safety events.
        next_grasp_target_id = None
        chain = getattr(runner, "_chain_action", None)
        if (context and phase in ("throw_return", "push_transfer")
                and getattr(chain, "skill_name", None) == "throw"
                and context.get("duration") is not None
                and elapsed >= float(context["duration"]) - .12 - 1e-9):
            next_target = getattr(chain, "target", None)
            serial = getattr(next_target, "sim_object_id", None)
            if serial is not None:
                next_grasp_target_id = int(serial)
        return {"task": task, "phase": phase, "target_id": target_id,
                "next_grasp_target_id": next_grasp_target_id}

    def phase(self):
        return self.phase_context()["phase"]

    def _box_id(self, index):
        if self.core._box_state[index] == self.core._FREE:
            return None
        serial = int(self.core._box_serial[index])
        return serial if serial > 0 else None

    def _handled(self, index, serial, context):
        return (bool(self.core._box_manipulated[index])
                or getattr(self.core, "_grabbed", None) == index
                or serial == context["target_id"]
                or serial == context.get("next_grasp_target_id"))

    def classify_pair(self, geom1, geom2, context=None):
        """Classify a geom pair without reading forces or changing counters.

        Return ``(kind, object_ids, collateral_ids)`` or ``None``.  A normal
        tool--target pair has kind ``normal_target`` and is never a safety event.
        """
        context = self.phase_context() if context is None else context
        geom1, geom2 = int(geom1), int(geom2)
        box1, box2 = self._geom_to_box.get(geom1), self._geom_to_box.get(geom2)
        if box1 is not None and box2 is not None:
            id1, id2 = self._box_id(box1), self._box_id(box2)
            if id1 is None or id2 is None:
                return None
            handled1, handled2 = self._handled(box1, id1, context), self._handled(box2, id2, context)
            if not (handled1 or handled2 or id1 in self._disturbed_ids or id2 in self._disturbed_ids):
                return None
            collateral = tuple(s for s, handled in ((id1, handled1), (id2, handled2)) if not handled)
            return "object_other", tuple(sorted((id1, id2))), collateral
        if box1 is None and box2 is not None:
            geom1, geom2, box1 = geom2, geom1, box2
        if box1 is None or geom2 not in self._robot_geoms:
            return None
        serial = self._box_id(box1)
        if serial is None:
            return None
        normal_phase = context["phase"] in ("throw_pick", "throw_loaded", "push_stroke")
        if geom2 in self._tool_geoms and serial == context["target_id"] and normal_phase:
            return "normal_target", (serial,), ()
        if geom2 in self._tool_geoms and serial == context.get("next_grasp_target_id"):
            return "normal_target", (serial,), ()
        collateral = () if self._handled(box1, serial, context) else (serial,)
        kind = "robot_target" if serial == context["target_id"] else "robot_other"
        return kind, (serial,), collateral

    def observe(self):
        """Read one just-completed MuJoCo step; repeated timestamps are ignored."""
        now = float(self.core.sim_time())
        if self._last_time is not None and now <= self._last_time + 1e-12:
            return
        self._last_time = now
        context = self.phase_context()
        target_id = context["target_id"]
        if target_id is not None:
            self._intended_ids.add(target_id)
        for i, state in enumerate(self.core._box_state):
            if state != self.core._FREE and self.core._box_manipulated[i]:
                self._intended_ids.add(int(self.core._box_serial[i]))
        if self._force_reader is None:
            import mujoco
            self._force_reader = mujoco.mj_contactForce
        dt = float(self.core._dt)
        pairs = {}
        object_contacts = []

        def record(classified, geom1, geom2, magnitude):
            kind, serials, collateral = classified
            if kind == "normal_target":
                self.normal_contact_samples += 1
                self._intended_ids.update(serials)
                return
            key = (kind, tuple(sorted((geom1, geom2))), serials)
            pair = pairs.setdefault(key, {"force": 0., "peak": 0., "samples": 0})
            pair["force"] += magnitude
            pair["peak"] = max(pair["peak"], magnitude)
            pair["samples"] += 1
            self._disturbed_ids.update(serials)
            self._collateral_ids.update(collateral)
            for serial in collateral:
                self._collateral_causes.setdefault(serial, dict(
                    cause_time=now, cause_task=context["task"],
                    cause_phase=context["phase"], cause_target_id=target_id))

        for index in range(int(self.core.data.ncon)):
            contact = self.core.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            box1, box2 = self._geom_to_box.get(geom1), self._geom_to_box.get(geom2)
            object_pair = box1 is not None and box2 is not None
            classified = None if object_pair else self.classify_pair(geom1, geom2, context)
            if classified is None and not object_pair:
                continue
            self._force[:] = 0.
            self._force_reader(self.core.model, self.core.data, index, self._force)
            magnitude = float(np.linalg.norm(self._force[:3]))
            if not math.isfinite(magnitude) or magnitude <= self.force_threshold:
                continue
            if object_pair:
                object_contacts.append((geom1, geom2, magnitude))
            else:
                record(classified, geom1, geom2, magnitude)
        # Solver contact ordering must not decide whether a collision chain is
        # counted.  First seed direct robot disturbances, then close the graph
        # of simultaneous object contacts connected to those or handled objects.
        while object_contacts:
            pending = []
            for geom1, geom2, magnitude in object_contacts:
                classified = self.classify_pair(geom1, geom2, context)
                if classified is None:
                    pending.append((geom1, geom2, magnitude))
                else:
                    record(classified, geom1, geom2, magnitude)
            if len(pending) == len(object_contacts):
                break
            object_contacts = pending
        for key, pair in pairs.items():
            impulse = pair["force"] * dt
            self._impulse += impulse
            event = self._last_events.get(key)
            if event is None or now - event["end_time"] > self.event_gap + 1e-9:
                event = dict(event_id=len(self.events), kind=key[0],
                    geom_ids=list(key[1]), object_ids=list(key[2]),
                    sim_time=now, start_time=now, end_time=now,
                    task=context["task"], phase=context["phase"], target_id=target_id,
                    impulse=0., peak_force=0., samples=0)
                self.events.append(event)
                self._last_events[key] = event
            event["end_time"] = now
            event["impulse"] += impulse
            event["peak_force"] = max(event["peak_force"], pair["peak"])
            event["samples"] += pair["samples"]
        self._observe_falls(now, context)

    def _observe_falls(self, now, context):
        base_p = np.asarray(self.core._base_p, dtype=float)
        threshold = float(self.core._belt_top_w) - float(base_p[2]) - self.fall_depth
        runner = getattr(self.core, "safe_runner", None)
        reach = float(getattr(getattr(runner, "cfg", None), "MAX_REACH", .65))
        min_y = float(getattr(self.core.cfg, "despawn_y", -.8))
        for i, state in enumerate(self.core._box_state):
            if state == self.core._FREE:
                continue
            serial = int(self.core._box_serial[i])
            if (serial not in self._collateral_ids or serial in self._fallen_ids
                    or serial in self._intended_ids or self.core._box_manipulated[i]):
                continue
            base = np.asarray(self.core._to_base(self.core._box_joints[i].qpos[:3]), dtype=float)
            in_zone = abs(float(base[0])) <= 1.2 and min_y <= float(base[1]) <= reach + .25
            if in_zone and float(base[2]) < threshold:
                self._fallen_ids.add(serial)
                self.falls.append(dict(sim_time=now, object_id=serial,
                    task=context["task"], phase=context["phase"],
                    target_id=context["target_id"], base_xyz=base.tolist(),
                    **self._collateral_causes.get(serial, {})))

    def snapshot(self):
        """Return JSON-serializable cumulative metrics for collector deltas."""
        count, falls = len(self.events), len(self.falls)
        return dict(contact_events=count, contacts=count, impulse=float(self._impulse),
            falls=falls, fall_events=falls, phase_counts=dict(Counter(e["phase"] for e in self.events)),
            kind_counts=dict(Counter(e["kind"] for e in self.events)),
            normal_contact_samples=int(self.normal_contact_samples))
