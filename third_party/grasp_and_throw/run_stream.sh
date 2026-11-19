#!/bin/bash
# A whole stream of objects, planned in real time WITHOUT a robot and WITHOUT a simulator (planner/rt_dry.py): the next object is chosen, every cycle is
# optimized, the motions from and to rest are added, and the commands are computed while the clock runs — exactly the planning and command threads of the runtime.
# usage: ./run_stream.sh SCHEDULE.json [NAME] [SECONDS]
#   SCHEDULE.json  list of objects [{"id": 1, "cls": "metal|paper|transparent", "x": 0.40, "t": 1.0}, ...]  (x = lane [m], t = time [s] at which it is put on the belt)
#   NAME           name of the run (default: stream)          SECONDS  the arm throws until this time on the run's clock (default 40)
# example: ./run_stream.sh examples/schedule_tight.json demo 40
# Result: planner/logs/<date>_<time>_NAME/cmds.npz (commands, one row per 2 ms sample) and rt_run.json (events); the line "SUMMARY ..." is printed.
# Environment: GT_BELT=0.16 / GT_BIN_X=1.10 another belt speed / bin distance (README.md, section 7); RT_ORDER=fifo for first-in-first-out; RT_DRY_SLOW=4 slows the clock 4 times (use it when the machine is slow or busy: all times in the
#   result stay in the run's clock); RT_DRY_TOPS=-0.02,0.0,0.03,0.05 gives every object one of these top heights (default 0.02 m).
set -e; H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$H/env.sh"
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 2; }
S="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; N="${2:-stream}"; export RT_STOP_SIM="${3:-40}"
cd "$H/planner" && mkdir -p logs && "$GT_PY" -W ignore rt_dry.py "$N" "$S" 2>&1 | grep -E "^SUMMARY|^RUN=|FAILED|Traceback|segment .* ends|late commit" | cut -c1-1200
