#!/bin/bash
# One grasp-and-throw cycle (contact with A -> throw A -> contact with B): plan, command and realized motion into one .npz file.
# usage: ./run_cycle.sh XA,YA[,ZA] XB,YB[,ZB] BIN [OUT.npz]
#   positions = suction points in the base frame [m] at the instant the cup touches A; BIN = 0|1|2 or metal|paper|transparent (bin of A)
# example: ./run_cycle.sh 0.40,0.30 0.45,0.50 paper        -> out/cycle.npz and a line "CYCLE {...}"
# Another belt speed or bin distance: GT_BELT=0.16 ./run_cycle.sh ...  or  GT_BIN_X=1.10 ./run_cycle.sh ...  (README.md, section 7; env.sh stops when the set is missing)
# Details: planner/plan_cycle.py (docstring) and README.md.
set -e; H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$H/env.sh"
[ $# -ge 3 ] || { sed -n '2,6p' "$0"; exit 2; }
OUT="${4:-$H/out/cycle.npz}"; mkdir -p "$(dirname "$OUT")"; OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
cd "$H/planner" && "$GT_PY" -W ignore plan_cycle.py --a "$1" --b "$2" --bin "$3" --out "$OUT" 2>&1 | grep -E "^CYCLE|Traceback|Error"
