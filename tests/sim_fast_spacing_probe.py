"""Print non-randomized fast-mode object spacing diagnostics."""

from __future__ import annotations

import sys

from gp8_control.rl.gym_env import GP8RecyclingEnv


def _snapshot(env: GP8RecyclingEnv, label: str) -> None:
    assert env._runner is not None
    core = env._runner.core
    rows = []
    with core.lock:
        for i, state in enumerate(core._box_state):
            if state == core._FREE:
                continue
            base = core._to_base(core._box_joints[i].qpos[0:3])
            rows.append((
                int(core._box_serial[i]),
                state,
                core._box_class[i],
                float(base[0]),
                float(base[1]),
                float(base[2]),
            ))
    rows.sort(key=lambda r: r[4])
    expected_gap = core.cfg.spawn_interval * core.cfg.belt_speed
    print(
        f"\n[{label}] sim_t={core.sim_time():.3f} active={len(rows)} "
        f"expected_gap={expected_gap:.3f}",
        flush=True,
    )
    for serial, state, cls, x, y, z in rows:
        print(
            f"  serial={serial:02d} state={state:8s} cls={cls:11s} "
            f"x={x:+.3f} y={y:+.3f} z={z:+.3f}",
            flush=True,
        )
    if len(rows) >= 2:
        gaps = [rows[i + 1][4] - rows[i][4] for i in range(len(rows) - 1)]
        print("  y_gaps=" + ", ".join(f"{gap:.3f}" for gap in gaps), flush=True)


def main(argv=None) -> int:
    del argv
    env = GP8RecyclingEnv(max_objects=6, max_steps=4, realtime=False)
    try:
        _, info = env.reset(options={"startup_timeout": 8.0})
        _snapshot(env, "after_reset")
        assert env._runner is not None
        env._runner.core.warp_seconds(12.0)
        env._runner.ingest()
        _snapshot(env, "after_12s_passive_warp")
        for step_idx in range(3):
            action = env._runner.app_heuristic_action_indices(info["action_mask"])
            _, reward, _terminated, _truncated, info = env.step(action)
            print(
                f"step={step_idx} action={action.tolist()} "
                f"reward={reward:+.3f} pending={info.get('pending_action')}",
                flush=True,
            )
            _snapshot(env, f"after_step_{step_idx}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
