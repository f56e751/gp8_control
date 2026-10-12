#!/bin/bash
# Mean / min / max over training seeds of the per-seed evaluation means.
#     summarize.sh <dir> <config> [<config> ...]
# reads <dir>/test.<config>_s<seed>.jsonl (one JSON line per evaluation episode).
dir=$1; shift
for c in "$@"; do
    all=""
    for f in "$dir"/test."${c}"_s[0-9]*.jsonl; do
        [ -f "$f" ] || continue
        all="$all $(grep -o '"return": [-0-9.]*' "$f" | awk '{s+=$2} END{printf "%.2f", s/NR}')"
    done
    echo "$all" | awk -v c="$c" '{
        if (NF) { s=0; mn=1e9; mx=-1e9
            for (i=1; i<=NF; i++) { s+=$i; if ($i<mn) mn=$i; if ($i>mx) mx=$i }
            printf "%-24s mean %.2f  min %.2f  max %.2f  seeds %d  [%s]\n", c, s/NF, mn, mx, NF, $0
        } else printf "%-24s -\n", c }'
done
