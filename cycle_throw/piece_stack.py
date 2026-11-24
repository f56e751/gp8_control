"""Piece mode, assembled in ONE place for the robot, the twin and RL.

``PieceStack`` owns the wiring every user of piece mode needs: the skills
(planner throw + PushSkill), which classes each skill takes, the decision
(``rl.piece_policy.PieceDecider``: observation, mask, policy) and the
``PieceRunner``. Subclasses only supply what differs:

* ``AppPieceStack``  -- the running app (robot or the app's own MuJoCo
  backend): wall clock, the controller buffer or the 2 ms stream, the planner
  link on robot6; ``GP8_PIECE_POLICY`` picks a policy.
* ``TwinPieceStack`` -- the MuJoCo twin without ROS (RL env, tools/piece_twin):
  ``rl.sim_runner.SimRlRunner`` with the controller filter
  (``cycle_sim_config``), a local link, the piece-mode settings
  (``piece_settings``), real time or VIRTUAL time (``backends/sim_tick.py``).

Change a rule here (or in the classes it uses) and the robot, the twin and the
RL env change together.
"""

from __future__ import annotations

import os
import types

import numpy as np

from gp8_control.cycle_throw.piece_runner import PieceRunner


class PieceStack:
    def __init__(self, *, app, ctx, push_skill, top_z: float | None = None, vtime=None,
                 on_execute=None, max_objects: int | None = None,
                 bbox_observation: str | None = None, log=None) -> None:
        from gp8_control.skills.planner_throw_skill import PLANNER_BIN_INDEX, PlannerThrowSkill

        self.app, self.ctx, self.push_skill = app, ctx, push_skill
        self.top_z, self.vtime, self.on_execute = top_z, vtime, on_execute
        self.log = log
        self.throw_skill = PlannerThrowSkill(ctx)
        self.skills = {"throw": self.throw_skill, "push": push_skill}
        #: which classes each skill takes (the policy's action mask uses this)
        self.can = {"throw": lambda o: o.class_name in PLANNER_BIN_INDEX,
                    "push": push_skill.can_handle}
        self.max_objects = int(max_objects or os.environ.get("GP8_PIECE_POLICY_OBJECTS", "6"))
        self.bbox_observation = bbox_observation or os.environ.get("GP8_RL_BBOX_OBSERVATION", "size")
        self.runner: PieceRunner | None = None
        self.decider = None

    # -- what differs per subclass ----------------------------------------
    def joints(self):
        q = self.app.traj_ctrl.current_joints
        return None if q is None else np.asarray(q, float)

    # -- the shared wiring ------------------------------------------------
    def make_decider(self, policy=None, remaining_fn=None, log=None):
        """The decision piece mode asks for (rl/piece_policy.py), on this stack."""
        from gp8_control.rl.piece_policy import PieceDecider, episode_remaining

        if remaining_fn is None:
            remaining_fn = episode_remaining(
                lambda: self.runner.now(),
                lambda: None if self.runner is None or self.runner.t0 is None else 0.0)
        self.decider = PieceDecider(
            queue=self.app.queue, ctx=self.ctx, robot=self.app.robot, joints_fn=self.joints,
            conveyor=self.app.conveyor, can=self.can, clock=self.ctx.clock,
            max_objects=self.max_objects, bbox_observation=self.bbox_observation,
            policy=policy, remaining_fn=remaining_fn, log=log,
            taken_fn=lambda o: self.runner is not None and int(o.track_id) in self.runner._taken)
        return self.decider

    def policy_decider(self, spec: str):
        """A decider running a policy (``heuristic``, ``upstream`` or a .zip)."""
        from gp8_control.rl.piece_policy import load_policy

        return self.make_decider(policy=load_policy(spec, self.max_objects), log=self.log)

    def build(self, decide_fn=None) -> PieceRunner:
        self.runner = PieceRunner(
            self.app, ctx=self.ctx, skills=self.skills, skill_for=self.ctx.skill_for,
            top_z=self.top_z, on_execute=self.on_execute, vtime=self.vtime,
            decide_fn=decide_fn)
        return self.runner

    def run(self, *, intake, ok, policy_spec: str = "") -> PieceRunner:
        """Build (a policy decides if ``policy_spec``, else the belt-order rule) and run."""
        decide = self.policy_decider(policy_spec) if policy_spec else None
        runner = self.runner if self.runner is not None else self.build(decide)
        runner.run(intake=intake, ok=ok)
        return runner


class AppPieceStack(PieceStack):
    """Piece mode inside the app (GP8App), robot or the app's MuJoCo backend."""

    def __init__(self, gp8app) -> None:
        top_z = None
        if gp8app.cfg.BACKEND == "mujoco":
            from gp8_control.cycle_throw.runner import sim_object_top
            top_z = sim_object_top(gp8app.world._core.cfg)
        log = gp8app._node.get_logger().info
        super().__init__(app=gp8app, ctx=gp8app.ctx, push_skill=gp8app.push_skill,
                         top_z=top_z, log=log)
        self.policy_spec = os.environ.get("GP8_PIECE_POLICY", "").strip()
        if self.policy_spec:
            log(f"[pieces] decisions by policy {self.policy_spec!r}")


class TwinPieceStack(PieceStack):
    """Piece mode in the MuJoCo twin without ROS (see the module docstring)."""

    def __init__(self, *, virtual: bool = True, seed: int | None = None, cfg=None,
                 max_objects: int = 6, bbox_observation: str = "size") -> None:
        from gp8_control.cycle_throw import piece_settings

        os.environ["GP8_SIM_REALTIME"] = "0" if virtual else "1"
        os.environ["GP8_PIECE_VIRTUAL"] = "1" if virtual else "0"
        os.environ["GP8_CYCLE_PLANNER_HOST"] = ""            # a local link per twin
        from gp8_control.backends.mujoco_sim import class_bin_name_for
        from gp8_control.config import Config
        from gp8_control.cycle_throw import runner as R
        from gp8_control.rl.sim_profile import apply_sim_profile
        from gp8_control.rl.sim_runner import SimRlRunner

        apply_sim_profile()                                 # a GP8_SIM_PROFILE fills env defaults first
        cfg = cfg or Config()
        cfg.BACKEND = "mujoco"
        self.settings = piece_settings.apply(cfg)           # the robot's piece-mode settings
        sim_cfg = R.cycle_sim_config(cfg)
        if seed is not None:
            sim_cfg.seed = int(seed)
        sim = SimRlRunner(cfg, sim_cfg=sim_cfg, realtime=not virtual, max_objects=max_objects,
                          bbox_observation=bbox_observation, sim_seed=seed)
        sim.start()
        self.sim = sim
        vt = None
        if virtual:
            from gp8_control.backends.sim_tick import SimTickBuffer, VirtualTime

            vt = VirtualTime(sim.core)
            buf = SimTickBuffer(vt, sim.core._ctrl_target)
            sim.core.set_tick_source(buf)
            sim.traj_ctrl.buf = buf
            # the controller holds the last COMMAND when the buffer takes over
            sim.traj_ctrl.buf_activate = lambda timeout=5.0: buf.activate(sim.core._ctrl_target)
            self.buf = buf
        self.vt = vt
        app = types.SimpleNamespace(cfg=sim.cfg, robot=sim.robot, traj_ctrl=sim.traj_ctrl,
                                    conveyor=sim.conveyor, queue=sim.queue, _node=sim.node,
                                    _spinner_dead=False, M1=sim.M1, M2=sim.M2,
                                    world=types.SimpleNamespace(_core=sim.core))
        bin_names = {b.name for b in sim.core.bins}

        def on_execute(o, skill):
            # The twin's bins: the planner's class bins (thrown) and "push_<class>"
            # (pushed); the right one depends on the skill that handled the object.
            want = f"push_{o.class_name}" if skill == "push" else o.class_name
            if want not in bin_names:
                want = class_bin_name_for(o.class_name)
            sim.core.mark_manipulated(o.sim_object_id, skill, want)

        super().__init__(app=app, ctx=sim.ctx, push_skill=sim.skills["push"],
                         top_z=R.sim_object_top(sim.core.cfg), vtime=vt, on_execute=on_execute,
                         max_objects=max_objects, bbox_observation=bbox_observation,
                         log=sim.node.get_logger().info)

    def joints(self):
        return self.sim.current_joints()

    def intake(self) -> None:
        self.sim.ingest()

    def observe(self) -> np.ndarray:
        """The decider's observation after the sim-only models (truth tracks, noise)."""
        self.sim._refresh_truth_tracks(self.sim.clock.time())
        return self.decider.observation()

    def close(self) -> None:
        self.sim.close()
