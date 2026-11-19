#!/bin/bash
# build_set.sh — builds the set of learned models for another belt speed and/or bin position (README.md, section 7).
# usage: build/build_set.sh V X [N]      V = belt speed [m/s], X = bin distance [m], N = solver processes (default: 70 % of the CPU threads)
#   e.g.  setsid nohup build/build_set.sh 0.12 1.475 14 > /dev/null 2>&1 < /dev/null &        progress: planner/logs/build_<T>.log, ends with BUILDDONE
# The set is named T = [b<VVV>_][x<XXX>_]zm<ZZZ> (e.g. x1475_zm011) and lands in planner/logs, where env.sh finds it (GT_BELT=V GT_BIN_X=X source env.sh).
# Stages — each one is skipped when its result exists, so the script can be started again after a stop, and the library can be split over machines:
#   1) keys            logs/real_keys_a6_<T>.txt (the 5 cm grid of cell keys for belt V), logs/tm_rand_a6_<T>_keys.txt (GT_RAND random off-grid tasks)
#   2) library         logs/real_A6_<T>/c_*.npz, one solved cycle per cell.  Pass 1 starts every cell from GT_FROM; pass 2 re-solves only the failures of pass 1,
#                      starting from solved neighbours.  Takes ~3 h with 14 processes on an i9-11900K (68,811 cells, ~6.5 cells/s)
#   3) off-grid tasks  logs/tm_rand_a6_<T> (validation of the models; with heights also logs/tm_randz_a6_<T>, random object tops)
#   4) cycle-time model  logs/tmodel_a6_<T>.npz
#   5) order network   logs/ord_a6_<T>/m1.npz (exhaustive expert on 600 streams x 2 views, one more round on the network's own decisions) + evaluation vs FIFO
#   6) start networks  logs/nnws_<T>/w2_n2c.npz (durations network + trajectory network; GPU with torch, ~20-40 min)
# Environment:
#   GT_BIN_Z=-0.11     bin height [m] (default -0.11, the real bins)
#   GT_HEIGHTS=1       models that take the tops of the objects as inputs (time model, order network, start networks; the default set zm011_z is built so).
#                      Only for belt 0.12 (the random-height tasks of mk_keys_z.py are for 0.12).  Adds GT_RANDZ (12000) solves and the suffix _z
#   GT_FROM=PATH       start of pass 1: a start network (.npz, default planner/logs/nnws_zm011_z/w2_n2c.npz) or a library folder of another set (each cell
#                      starts from the nearest cell there, e.g. robot6 ~/rge_jc/logs/real_A6_zm011)
#   GT_PART=i/n | i,j/n   only stage 2, only the cells whose line number is i (or i, j) modulo n, then stop.  Run one part per machine, copy the folders
#                      logs/real_A6_<T> together onto one machine and start build_set.sh there without GT_PART: it solves what is missing and goes on
#   GT_RAND=1500       number of off-grid tasks (validation only when < 2000)        GT_TORCH_PY=python with torch and CUDA for stage 6
#   GT_TEST=n          smoke test: n cells, n tasks, 4 streams, few epochs, set name <T>_test (delete it afterwards)
# The solver processes run at nice 19.  CPU stages wait while a real-robot planner or an Isaac world runs on this machine.
set -o pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; GT_ENV_BASE=1 source "$H/env.sh" || exit 1
[ $# -ge 2 ] || { sed -n '2,4p' "$0"; exit 2; }
gt_num() { awk -v v="$1" 'BEGIN { s = sprintf("%.3f", v); sub(/0$/, "", s); print s }'; }
V=$(gt_num "$1"); X=$(gt_num "$2"); Z=$(gt_num "${GT_BIN_Z:--0.11}"); N=${3:-$(( $(nproc) * 7 / 10 ))}; [ "$N" -ge 1 ] || N=1
export RGE_BIN_Z=$Z; unset RGE_BELT_V RGE_BIN_X ORD_Z ORD_TMODEL RN_NN RN_DB
T=""; [ "$V" = 0.12 ] || { T="b$(echo $V | tr -d .)_"; export RGE_BELT_V=$V; }; [ "$X" = 1.35 ] || { T="${T}x$(echo $X | tr -d .)_"; export RGE_BIN_X=$X; }
case "$Z" in -*) T="${T}zm$(echo ${Z#-} | tr -d .)" ;; *) T="${T}zp$(echo $Z | tr -d .)" ;; esac
HZ=${GT_HEIGHTS:-0}; S=""; [ "$HZ" = 1 ] && S="_z"
if [ "$HZ" = 1 ] && [ "$V" != 0.12 ]; then echo "[build_set] GT_HEIGHTS=1 only for belt 0.12" >&2; exit 2; fi
TEST=${GT_TEST:-0}; [ "$TEST" -gt 0 ] && T="${T}_test"
NRAND=${GT_RAND:-1500}; NRZ=${GT_RANDZ:-12000}; NSTR=600; EP_TM=""; EP_NN=3000; TMH=2000      # TMH = random-height tasks kept out of the time-model training (tm_fit_z.py TM_HOLD)
[ "$TEST" -gt 0 ] && { NRAND=$TEST; NRZ=$TEST; NSTR=4; EP_TM=50; EP_NN=30; TMH=$(( TEST / 2 )); }
if [ -z "$GT_TORCH_PY" ]; then for p in /PublicHDD/ryugaeun/cls_train/.venv/bin/python "$HOME/nn_train/.venv/bin/python"; do [ -x "$p" ] && { GT_TORCH_PY=$p; break; }; done; GT_TORCH_PY=${GT_TORCH_PY:-python3}; fi
FROM=${GT_FROM:-$H/planner/logs/nnws_zm011_z/w2_n2c.npz}
cd "$H/planner" || exit 1; mkdir -p logs; P=$GT_PY; L=logs/build_$T.log
DB=logs/real_A6_$T; RD=logs/tm_rand_a6_$T; RDZ=logs/tm_randz_a6_$T; TM=logs/tmodel_a6_$T$S.npz; D=logs/ord_a6_$T$S; NW=logs/nnws_$T$S
KG=logs/real_keys_a6_$T.txt; KR=logs/tm_rand_a6_${T}_keys.txt; KZ=logs/tm_randz_a6_${T}_keys.txt
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $L; }
free_cpu() { while pgrep -f "rt_run_h[w].py|isaac_worl[d]" > /dev/null; do sleep 60; done; }
start_env() { case "$1" in *.npz) echo "RN_NN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;; *) echo "RN_DB=$1" ;; esac; }
fails() { $P - "$1" "$2" <<'PYEOF'
import json, glob, sys
rows = json.load(open(sorted(glob.glob(sys.argv[1] + "/_pool_*.json"))[-1])); bad = [r[0] for r in rows if not (r[1] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and r[4] <= 1e-5)]
open(sys.argv[2], "w").write("".join(k + "\n" for k in bad)); print(len(bad))
PYEOF
}
todo() { python3 - "$1" "$2" "$3" <<'PYEOF'
import os, sys
db, kg, out = sys.argv[1:4]; have = {f[2:-4] for f in os.listdir(db)} if os.path.isdir(db) else set()
todo = [l for l in open(kg) if l.strip() and l.strip().replace(",", "_") not in have]; open(out, "w").write("".join(todo)); print(len(todo))
PYEOF
}
solve_lib() {      # $1 = key file: pass 1 from FROM, pass 2 (failures only) from solved neighbours
  local k=$1 tag=$2 n
  n=$(todo $DB $k ${k%.txt}_todo.txt); say "$T library$tag: $n of $(wc -l < $k) cells to solve (start: $FROM)"
  [ "$n" -gt 0 ] || return 0
  free_cpu; env DB_REFRESH=0 $(start_env "$FROM") nice -n 19 $P -W ignore db_pool.py ${k%.txt}_todo.txt $DB $N > logs/build_${T}_pool1.out 2>&1; say "$T library$tag pass 1: $(tail -1 logs/build_${T}_pool1.out | cut -c1-300)"
  n=$(fails $DB ${k%.txt}_fail.txt)
  if [ "$n" -gt 0 ]; then free_cpu; RN_DB=$DB nice -n 19 $P -W ignore db_pool.py ${k%.txt}_fail.txt $DB $N > logs/build_${T}_pool2.out 2>&1; say "$T library$tag pass 2 (the $n failures, from solved neighbours): $(tail -1 logs/build_${T}_pool2.out | cut -c1-300)"; fi
}
say "== build $T: belt $V m/s, bin x $X m, bin z $Z m, heights $HZ, $N processes, python $P, torch python $GT_TORCH_PY"
# 1) keys
if [ ! -s $KG ]; then
  say "$T keys: $($P mk_keys_belt.py $V $KG $KR $NRAND | tail -1)"
  [ "$TEST" -gt 0 ] && { head -n $TEST $KG > $KG.t && mv $KG.t $KG; head -n $TEST $KR > $KR.t && mv $KR.t $KR; }
fi
[ "$HZ" = 1 ] && [ ! -s $KZ ] && say "$T keys with heights: $($P mk_keys_z.py $KZ $NRZ | tail -1)"
say "$T bins $($P -W ignore -c 'import os; os.environ["LPP_VIN"]="1"; import cyc2; print(cyc2.BINS, "belt", cyc2.VB)' 2>/dev/null | tail -1)"
# 2) library
if [ -n "$GT_PART" ]; then
  pp=${GT_PART%/*}; np_=${GT_PART#*/}; KP=logs/real_keys_a6_${T}_part$(echo $pp | tr , _).txt
  awk -v p="$pp" -v n=$np_ 'BEGIN { split(p, a, ","); for (i in a) s[a[i] + 0] = 1 } ((NR % n) in s)' $KG > $KP
  solve_lib $KP " part $pp/$np_"; say "PARTDONE $T part $pp/$np_: $(ls $DB | grep -c '^c_.*\.npz$') cells in $DB. Copy $DB to the machine that finishes the set and start build_set.sh there without GT_PART"; exit 0
fi
if [ ! -f $DB.done ]; then solve_lib $KG ""; n=$(ls $DB | grep -c '^c_.*\.npz$'); say "$T library: $n cells of $(wc -l < $KG)"; touch $DB.done; fi
# 3) off-grid tasks, 4) cycle-time model
if [ ! -f $TM ]; then
  free_cpu; DB_REFRESH=0 RN_DB=$DB nice -n 19 $P -W ignore db_pool.py $KR $RD $N > logs/build_${T}_rand.out 2>&1; say "$T off-grid tasks: $(tail -1 logs/build_${T}_rand.out | cut -c1-300)"
  if [ "$HZ" = 1 ]; then
    free_cpu; DB_REFRESH=0 RN_DB=$DB nice -n 19 $P -W ignore db_pool.py $KZ $RDZ $N > logs/build_${T}_randz.out 2>&1; say "$T random-height tasks: $(tail -1 logs/build_${T}_randz.out | cut -c1-300)"
    TM_DB=$DB TM_RAND=$RD TM_RANDZ=$RDZ TM_OUT=$TM TM_OLD=logs/tmodel_a6.npz TM_HOLD=$TMH nice -n 10 $P tm_fit_z.py $EP_TM > logs/build_${T}_tm.out 2>&1
  else
    TM_DB=$DB TM_RAND=$RD TM_OUT=$TM nice -n 10 $P tm_fit.py $EP_TM > logs/build_${T}_tm.out 2>&1
  fi
  say "$T cycle-time model: $(tail -1 logs/build_${T}_tm.out | cut -c1-600)"
fi
# 5) order network
if [ ! -f $D/m1.npz ]; then
  mkdir -p $D; export ORD_TMODEL=$TM; [ "$HZ" = 1 ] && export ORD_Z=-0.02,0.05; free_cpu
  for r in "0 51000 1.4 sim" "0 52000 0.93 hw"; do set -- $r; nice -n 19 $P ordsim_cli.py data --streams $NSTR --seed $2 --y-vis $3 --expert full --procs $N --out $D/r0_$4.npz > $D/r0_$4.log 2>&1; say "$T order data r0_$4: $(tail -1 $D/r0_$4.log | cut -c1-200)"; done
  nice -n 10 $P -m gp8_control.learning.train_order_policy --data $D/r0_sim.npz $D/r0_hw.npz --target adv --adv-margin 0 --out $D/m0.npz > $D/m0.log 2>&1; say "$T order network m0: $(grep -v '^  ep' $D/m0.log | tail -1 | cut -c1-200)"
  free_cpu
  for r in "61000 1.4 sim" "62000 0.93 hw"; do set -- $r; nice -n 19 $P ordsim_cli.py data --streams $NSTR --seed $1 --y-vis $2 --expert full --procs $N --policy $D/m0.npz --out $D/r1_$3.npz > $D/r1_$3.log 2>&1; say "$T order data r1_$3: $(tail -1 $D/r1_$3.log | cut -c1-200)"; done
  nice -n 10 $P -m gp8_control.learning.train_order_policy --data $D/r0_sim.npz $D/r0_hw.npz $D/r1_sim.npz $D/r1_hw.npz --target adv --adv-margin 0 --out $D/m1.npz > $D/m1.log 2>&1; say "$T order network m1: $(grep -v '^  ep' $D/m1.log | tail -1 | cut -c1-200)"
  nice -n 10 $P ordsim_cli.py eval --streams $(( NSTR < 40 ? NSTR : 40 )) --seed 41 --y-vis 0.93 --policies fifo nn --policy $D/m1.npz --procs $N > $D/eval.out 2>&1; say "$T order network vs FIFO: $(tail -2 $D/eval.out | tr '\n' ' ' | cut -c1-400)"
  unset ORD_Z ORD_TMODEL
fi
# 6) start networks (GPU)
if [ ! -f $NW/w2_n2c.npz ]; then
  mkdir -p $NW; nice -n 10 $P -W ignore nnws_pack.py $DB $NW/grid.npz $N 2>&1 | grep PACKED | cut -c1-200 | tee -a $L; nice -n 10 $P -W ignore nnws_pack.py $RD $NW/rand.npz $N 2>&1 | grep PACKED | cut -c1-200 | tee -a $L
  [ "$HZ" = 1 ] && nice -n 10 $P -W ignore nnws_pack.py $RDZ $NW/randz.npz $N 2>&1 | grep PACKED | cut -c1-200 | tee -a $L
  NNWS_Z=$HZ NNWS_DIR=$NW nice -n 5 $GT_TORCH_PY nnws_train_dir.py n2c $EP_NN 2048 6 2>&1 | grep -E "NN2|Error|Traceback" | cut -c1-900 | tee -a $L
  [ -f $NW/w2_n2c.npz ] && rm -f $NW/grid.npz $NW/rand.npz $NW/randz.npz $NW/pred_n2c.npz
fi
ok=1; for f in $DB.done $TM $D/m1.npz $NW/w2_n2c.npz; do [ -e $f ] || { ok=0; say "MISSING $f"; }; done
[ $ok = 1 ] && say "BUILDDONE $T: library $DB, cycle-time model $TM, order network $D/m1.npz, start networks $NW/w2_n2c.npz — use with: GT_BELT=$V GT_BIN_X=$X source env.sh" || say "BUILDSTOP $T"
