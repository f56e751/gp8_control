"""E1 report: guard on vs off, trained policies.

  python -m gp8_control.sim_experiment.aggregate_e1 OUT_ROOT

OUT_ROOT/guard<g>_seed<s>/ holds summary.json, eval_10seed/seed_<k>.jsonl
(fixed-seed evaluation traces) and diag/ (push_diag --policy model).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _m(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def _sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def eval_stats(run: Path) -> dict | None:
    files = sorted((run / "eval_10seed").glob("seed_*.jsonl"))
    if not files:
        return None
    returns, sel, exe = [], Counter(), Counter()
    push_offered = push_chosen_when_offered = 0
    incomplete = 0
    for f in files:
        recs = [json.loads(l) for l in f.read_text().splitlines() if l]
        if not recs or not recs[-1].get("truncated_by_time"):
            incomplete += 1
        returns.append(sum(float(r["reward"]) for r in recs))
        for r in recs:
            sel["skip" if r["is_skip_slot"] else r["skill_name"]] += 1
            if r.get("executed") and r.get("executed_skill"):
                exe[r["executed_skill"]] += 1
            pairs = r.get("valid_pairs") or []
            if any(int(p[1]) == 1 and int(p[0]) < 6 for p in pairs):
                push_offered += 1
                if not r["is_skip_slot"] and r["skill_name"] == "push":
                    push_chosen_when_offered += 1
    return {"returns": returns, "selected": sel, "executed": exe, "n": len(files),
            "incomplete": incomplete, "push_offered": push_offered,
            "push_chosen_when_offered": push_chosen_when_offered}


def diag_stats(run: Path) -> dict | None:
    d = run / "diag"
    if not d.exists():
        return None
    p_push, objs = [], []
    for f in sorted(d.glob("model_seed*_steps.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            pp = r.get("policy_probs") or {}
            if r.get("push_ok_slots", 0) > 0 and "p_push" in pp:
                p_push.append(pp["p_push"])
    for f in sorted(d.glob("model_seed*_push_objects.jsonl")):
        objs += [json.loads(l) for l in f.read_text().splitlines() if l]
    return {"p_push_when_offered": p_push, "objs": objs}


def report(root: Path) -> str:
    runs = sorted(p for p in root.glob("guard*_seed*") if p.is_dir())
    by_guard = defaultdict(list)
    lines = [f"# E1 결과 — `{root}`", "",
             "## 학습 run별 (고정 평가 seed 0~9)", "",
             "| run | 학습 시간(분) | 평가 보상 평균±sd | 선택 throw/push/skip | 실행 throw/push | push 허용된 결정 중 push 선택 | 정책의 push 확률 평균 (push 허용 시) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        guard = run.name.split("_")[0]
        s = {}
        if (run / "summary.json").exists():
            s = json.loads((run / "summary.json").read_text())
        ev = eval_stats(run)
        dg = diag_stats(run)
        by_guard[guard].append((ev, dg))
        if ev is None:
            lines.append(f"| {run.name} | - | (평가 없음) | | | | |")
            continue
        sel, exe = ev["selected"], ev["executed"]
        frac = ev["push_chosen_when_offered"] / max(1, ev["push_offered"])
        pp = _m(dg["p_push_when_offered"]) if dg else float("nan")
        lw = s.get("learn_wall_s")
        lines.append(
            f"| {run.name} | {lw / 60 if lw else float('nan'):.0f} | {_m(ev['returns']):.1f} ± {_sd(ev['returns']):.1f}"
            f"{' (미완 ' + str(ev['incomplete']) + ')' if ev['incomplete'] else ''} "
            f"| {sel['throw']}/{sel['push']}/{sel['skip']} | {exe['throw']}/{exe['push']} "
            f"| {ev['push_chosen_when_offered']}/{ev['push_offered']} ({frac:.2f}) | {pp:.3f} |")
    lines += ["", "## 가드 설정별 합계", "",
              "| 설정 | run 수 | 평가 보상 평균 | push 선택 비율(전체 결정 중) | push 허용 시 push 선택 비율 | push 물체 결과 자기통/다른통/바닥 | 물리 플래그 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for guard, items in sorted(by_guard.items()):
        evs = [e for e, _ in items if e]
        rets = [x for e in evs for x in e["returns"]]
        sel = Counter()
        for e in evs:
            sel.update(e["selected"])
        tot = sum(sel.values())
        off = sum(e["push_offered"] for e in evs)
        cho = sum(e["push_chosen_when_offered"] for e in evs)
        objs = [o for _, d in items if d for o in d["objs"]]
        oc = Counter(o["outcome"] for o in objs)
        fl = Counter(f for o in objs for f in o["flags"])
        lines.append(
            f"| {guard} | {len(items)} | {_m(rets):.1f} | {sel['push'] / max(1, tot):.3f} | {cho / max(1, off):.3f} "
            f"| {oc['class_bin']}/{oc['wrong_bin']}/{oc['missed']} | {dict(fl) or '-'} |")
    lines += ["", "클립: 각 run의 `diag/clips/`."]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(report(Path(argv[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
