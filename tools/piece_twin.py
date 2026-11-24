"""Piece mode in the MuJoCo twin, with the robot's wiring (TwinPieceStack).

    python -m gp8_control.tools.piece_twin --seconds 60 [--realtime] [--policy heuristic]
        [--spawn-script objects.json] [--record DIR]

* ``--realtime``: the twin on the wall clock with the 2 ms command stream (as
  the app's MuJoCo backend); default: VIRTUAL time with the controller-tick
  buffer (as the RL env).
* ``--policy``: none = the belt-order rule; ``heuristic`` = the same rule
  through the policy path; ``upstream`` = a test policy (most upstream first);
  or a MaskablePPO .zip -- exactly what GP8_PIECE_POLICY does on the robot.
* ``--spawn-script``: a recorded object stream (``tools/real2sim spawn``).
* ``--record``: a run recording (``GP8_RUN_RECORD_DIR``) for ``tools/real2sim``.

Prints one RESULT line (JSON): runner stats, wall/sim seconds, object outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=60.0, help="planner seconds to run")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--policy", default="")
    ap.add_argument("--spawn-script", default="")
    ap.add_argument("--record", default="")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    if args.spawn_script:
        os.environ["GP8_SIM_SPAWN_SCRIPT"] = args.spawn_script
    if args.record:
        os.environ["GP8_RUN_RECORD_DIR"] = args.record
    # The real camera's field of view stays on (the twin's default): with it
    # off, a can left on the belt after a missed push is detected again in
    # front of the robot -- which the real camera can never see -- and a
    # policy goes after it again.
    os.environ.setdefault("GP8_SIM_CLASSES", "transparent,metal")
    os.environ.setdefault("GP8_SIM_SPAWN_INTERVAL", "1.6")

    from gp8_control.cycle_throw.piece_stack import TwinPieceStack

    w0 = time.monotonic()
    stack = TwinPieceStack(virtual=not args.realtime, seed=args.seed)
    t_end = [None]
    holder = {}

    def ok():
        r = holder.get("r")
        if r is None or r.t0 is None:
            return True
        if t_end[0] is None:
            t_end[0] = r.now() + args.seconds
        return r.now() < t_end[0]

    decide = stack.policy_decider(args.policy) if args.policy else None
    holder["r"] = stack.build(decide)
    err = None
    try:
        holder["r"].run(intake=stack.intake, ok=ok)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    runner = holder["r"]
    events = list(stack.sim.core._reward_events)
    out = dict(error=err, stats=dict(runner.stats), settings=stack.settings,
               virtual=not args.realtime, policy=args.policy or "rule",
               sim_s=round(stack.sim.core.sim_time(), 1), wall_s=round(time.monotonic() - w0, 1),
               warped_s=round(stack.vt.warped_s, 1) if stack.vt is not None else 0.0,
               decisions=getattr(decide, "n", None),
               buf_rejects=stack.buf.rejects[:5] if stack.vt is not None else None)
    stack.close()
    rows: dict = {}
    for e in events:
        if not (e["manipulated"] or e["collateral"]):
            k = "rode past"
        elif e["actual_bin_name"] and e["actual_bin_name"] == e["class_bin_name"]:
            k = "in its bin"
        else:
            k = f"missed ({e['actual_bin_name']})"
        key = f"{e['class_name']} {e.get('manipulated_skill') or '-'}: {k}"
        rows[key] = rows.get(key, 0) + 1
    out["outcomes"] = dict(sorted(rows.items()))
    print("RESULT", json.dumps(out, default=str), flush=True)
    return 0 if err is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
