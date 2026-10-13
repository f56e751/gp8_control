#!/bin/bash
# Build the 100 / 200 / 500-episode datasets as links into the 1,000-episode one.
# A smaller dataset is the first N seeds of data_seeds_1000.txt, so the sets are nested.
# Run from the parent directory of gp8_control.
set -euo pipefail
RUNS=gp8_control/examples/dppo/runs
SEEDS=gp8_control/examples/dppo/data_seeds_1000.txt
[ -f "$SEEDS" ] || { echo "run from the parent directory of gp8_control" >&2; exit 1; }
for n in 100 200 500; do
    mkdir -p "$RUNS/data_$n"
    head -n "$n" "$SEEDS" | while read -r seed; do
        f=$(printf 'ep_%06d.npz' "$seed")
        [ -e "$RUNS/data_1000/$f" ] || { echo "missing $RUNS/data_1000/$f" >&2; exit 1; }
        ln -sfn "../data_1000/$f" "$RUNS/data_$n/$f"
    done
    echo "data_$n: $(ls "$RUNS/data_$n" | wc -l) episodes"
done
