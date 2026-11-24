"""Aggregate push_diag runs into one markdown report.

  python -m gp8_control.sim_experiment.aggregate_diag OUT_ROOT

OUT_ROOT holds one folder per setting (e.g. guard1/, guard0/), each with
``<policy>_seed<k>_summary.json`` / ``_steps.jsonl`` / ``_push_objects.jsonl``.
Rows are (setting, policy), averaged over seeds.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def _sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def collect(root: Path):
    rows = defaultdict(list)
    for sfile in sorted(root.glob("*/*_summary.json")):
        setting = sfile.parent.name
        s = json.loads(sfile.read_text())
        tag = sfile.name[: -len("_summary.json")]
        steps = [json.loads(l) for l in (sfile.parent / f"{tag}_steps.jsonl").read_text().splitlines() if l]
        objs = [json.loads(l) for l in (sfile.parent / f"{tag}_push_objects.jsonl").read_text().splitlines() if l]
        ret = sum(float(r["reward"]) for r in steps)
        ev = [e for r in steps for e in r.get("events", [])]
        by_skill = Counter()
        for e in ev:
            sk = e.get("manipulated_skill") or ("collateral" if e.get("collateral") else "untouched")
            ok = e.get("actual_bin_name") is not None and e.get("actual_bin_name") == e.get("class_bin_name")
            by_skill[(sk, e.get("class_name"), "ok" if ok else "bad")] += 1
        rows[(setting, s["policy"])].append({
            "seed": s["seed"], "return": ret, "summary": s, "objs": objs, "by_skill": by_skill,
            "episode_s": max((r.get("t") or 0.0) for r in steps) if steps else 0.0,
        })
    return rows


def report(root: Path) -> str:
    rows = collect(root)
    out = [f"# push 진단 결과 — `{root}`", ""]
    out += ["## 1. 행동 마스크·에피소드 보상 (seed 평균 ± 표준편차)", "",
            "| 설정 | 정책 | seed 수 | 에피소드 보상 | push 허용 비율 | throw 허용 비율 | push 실행(성공/실패) | throw 실행(성공/실패) |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (setting, policy), rs in sorted(rows.items()):
        ex = Counter()
        for r in rs:
            ex.update(r["summary"]["executed"])
        out.append(
            f"| {setting} | {policy} | {len(rs)} | {_mean([r['return'] for r in rs]):.1f} ± {_sd([r['return'] for r in rs]):.1f} "
            f"| {_mean([r['summary']['push_offered_frac_of_live_slots'] for r in rs]):.2f} "
            f"| {_mean([r['summary']['throw_offered_frac_of_live_slots'] for r in rs]):.2f} "
            f"| {ex.get('push_ok', 0)}/{ex.get('push_fail', 0)} | {ex.get('throw_ok', 0)}/{ex.get('throw_fail', 0)} |")
    out += ["", "## 2. push된 물체의 결과와 물리 이상 플래그 (모든 seed 합)", "",
            "| 설정 | 정책 | push 물체 | 자기 통 | 다른 통 | 바닥 | 관통 | sink | fast | 위로 튐 | 회전 | 최대 속도 p95 (m/s) | 최대 관통 p95 (mm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (setting, policy), rs in sorted(rows.items()):
        objs = [o for r in rs for o in r["objs"]]
        if not objs:
            continue
        oc = Counter(o["outcome"] for o in objs)
        fl = Counter(f for o in objs for f in o["flags"])
        sp = sorted(o.get("max_speed", 0.0) for o in objs)
        pn = sorted(o.get("max_pen_m", 0.0) for o in objs)
        p95 = lambda xs: xs[min(len(xs) - 1, int(0.95 * len(xs)))]
        out.append(
            f"| {setting} | {policy} | {len(objs)} | {oc['class_bin']} | {oc['wrong_bin']} | {oc['missed']} "
            f"| {fl['penetration']} | {fl['sink']} | {fl['fast']} | {fl['launch_up']} | {fl['spin']} "
            f"| {p95(sp):.2f} | {1000 * p95(pn):.1f} |")
    out += ["", "## 3. 스킬·클래스별 분류 결과 (모든 seed 합, 물체 수)", "",
            "| 설정 | 정책 | 스킬 | 클래스 | 자기 통 | 실패 | 성공률 |", "|---|---|---|---|---:|---:|---:|"]
    for (setting, policy), rs in sorted(rows.items()):
        bs = Counter()
        for r in rs:
            bs.update(r["by_skill"])
        keys = sorted({(k[0], k[1]) for k in bs})
        for sk, cl in keys:
            ok, bad = bs[(sk, cl, "ok")], bs[(sk, cl, "bad")]
            if sk == "untouched":
                continue
            out.append(f"| {setting} | {policy} | {sk} | {cl} | {ok} | {bad} | {ok / max(1, ok + bad):.2f} |")
    out += ["", "## 4. 실행 실패 사유 (모든 seed 합)", ""]
    for (setting, policy), rs in sorted(rows.items()):
        ex = Counter()
        for r in rs:
            ex.update({k: v for k, v in r["summary"]["executed"].items() if k.startswith("fail_detail")})
        if ex:
            out.append(f"- **{setting} / {policy}**: " + "; ".join(f"{k[12:]} ×{v}" for k, v in ex.most_common(6)))
    out += ["", "클립: 각 설정 폴더의 `clips/` (파일 이름 = 정책_seed_번호_이유_물체)."]
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(report(Path(argv[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
