"""Replay a recorded 250 Hz command stream through the MuJoCo twin's arm.

The training env drives the arm exactly this way: ``RobotBackend`` emits 4 ms
position samples and ``SimCore`` zero-order-holds the latest one per 2 ms
substep (``_step_once_locked``). Feeding the *hardware's* recorded samples —
with their real timestamps, jitter included — therefore yields what the sim
would have done for the identical action, which the analyzer compares against
the hardware's measured joint feedback.

Each motion window is replayed independently, re-seeded from the hardware's
measured state at the window start, so one window's discrepancy can't leak
into the next.
"""

from __future__ import annotations

import math
import os

import numpy as np


class ArmReplayer:
    def __init__(self) -> None:
        import mujoco

        from gp8_control.backends.mujoco_sim import MJ_JOINTS, SimConfig, SimCore

        self._mj = mujoco
        cfg = SimConfig(
            realtime=False,
            randomize=False,
            physical_belt=False,
            spawn_interval=1e9,   # no boxes: arm-only replay
            bins=(),
            viewer=False,
            record_path="",
            landing_log=False,
        )
        self.core = SimCore(cfg)
        self.kp = float(cfg.kp)
        self.kv = float(cfg.kv)
        self.dt = float(self.core.model.opt.timestep)
        m = self.core.model
        self._qadr = [int(m.jnt_qposadr[m.joint(n).id]) for n in MJ_JOINTS]
        self._vadr = [int(m.jnt_dofadr[m.joint(n).id]) for n in MJ_JOINTS]

    def replay(self, cmd_t, cmd_q, *, t0: float, t1: float, q0, qd0=None,
               hold_q=None) -> tuple[np.ndarray, np.ndarray]:
        """Simulate [t0, t1] (recorded timeline); returns (t, q) at every substep.

        ``hold_q`` is the command in force before the first sample of the
        window (the JGPC/ZOH keeps the previous command), default ``q0``.
        """
        core, data = self.core, self.core.data
        cmd_t = np.asarray(cmd_t, dtype=float)
        cmd_q = np.asarray(cmd_q, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        qd0 = np.zeros(6) if qd0 is None else np.asarray(qd0, dtype=float)
        with core.lock:
            data.qpos[self._qadr] = q0
            data.qvel[self._vadr] = qd0
            data.qacc[:] = 0.0
            core._ctrl_target = np.asarray(q0 if hold_q is None else hold_q, dtype=float)
            self._mj.mj_forward(core.model, data)
        n = int(math.ceil((t1 - t0) / self.dt))
        out_t = np.empty(n + 1)
        out_q = np.empty((n + 1, 6))
        out_t[0] = t0
        out_q[0] = data.qpos[self._qadr]
        k = int(np.searchsorted(cmd_t, t0, side="right"))
        sim_t0 = float(data.time)
        for i in range(n):
            # ZOH: the latest command stamped at or before this substep's time
            # (same rule as SimCore.advance_steps on the _pending_ctrl queue).
            t_sub = t0 + (float(data.time) - sim_t0)
            while k < len(cmd_t) and cmd_t[k] <= t_sub:
                core._ctrl_target = cmd_q[k]
                k += 1
            with core.lock:
                core._step_once_locked()
            out_t[i + 1] = t0 + (float(data.time) - sim_t0)
            out_q[i + 1] = data.qpos[self._qadr]
        return out_t, out_q

    def close(self) -> None:
        try:
            self.core.stop()
        except Exception:
            pass


def available() -> tuple[bool, str]:
    try:
        import mujoco  # noqa: F401
        from gp8_control.backends import mujoco_sim  # noqa: F401
    except Exception as exc:
        return False, repr(exc)
    if os.environ.get("GP8_S2R_NO_REPLAY", "").lower() in ("1", "true", "yes"):
        return False, "disabled by GP8_S2R_NO_REPLAY"
    return True, ""
