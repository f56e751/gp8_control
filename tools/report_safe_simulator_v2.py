"""Compact paired report for phase-owned route Safe PPO; no web server."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import numpy as np

from gp8_control.tools.train_safe_simulator_v2 import ROOT, run_paths, save_json, sha
from gp8_control.tools.report_safe_simulator import paired_statistics

METRICS = (
    ('task_success_percent', 'Correct bin success', '%', False),
    ('contacts', 'Unintended contacts / 20 offers', 'events', True),
    ('contacts_per_100_actions', 'Contacts / 100 skill attempts', 'events', True),
    ('impulse_per_action_ns', 'Impulse / skill attempt', 'N s', True),
    ('falls_per_100_offered', 'Collateral falls / 100 offers', 'objects', True),
    ('cost_per_offered', 'Weighted risk / offered object', 'cost', True),
    ('proximity_integral', 'Near-object exposure / episode', 's', True),
    ('physical_unresolved', 'Unresolved physical objects at terminal', 'objects', True),
    ('sim_seconds', 'Episode duration including deadlines', 's', True),
    ('throughput_per_minute', 'Correct bins / minute', 'objects', False),
)
ARMS = ('baseline', 'v2_zero', 'safe_seed0', 'safe_seed1')
MAIN_METRICS = ('task_success_percent','contacts','impulse_per_action_ns','falls_per_100_offered')


def contact_start_regions(result, opportunity_records=None):
    records = result['trace']['residuals'] if opportunity_records is None else opportunity_records
    windows = []
    for record in records:
        if not record.get('active') or 'native_boundary_offset' not in record:
            continue
        origin = record['start_sim_time']-record['native_boundary_offset']
        windows.append((origin+record['edit_start'], origin+record['edit_end'], record['phase']))
    counts = Counter()
    for event in result['trace']['contacts']:
        when = event['start_time']
        roles = [role for a,b,role in windows if a-1e-8 <= when <= b+1e-8]
        if len(roles) > 1:
            raise RuntimeError('Two route edit windows overlap a contact start')
        counts['editable' if roles else 'outside_editable'] += 1
        if roles:
            counts['editable_'+roles[0]] += 1
    return dict(editable=counts['editable'], outside_editable=counts['outside_editable'],
                by_role={key[9:]:value for key,value in counts.items() if key.startswith('editable_')})


def verdict(metrics):
    success = metrics['task_success_percent']['difference_ci95']
    contact = metrics['contacts']['difference_ci95']
    impulse = metrics['impulse_per_action_ns']['difference_ci95']
    fall = metrics['falls_per_100_offered']['difference_ci95']
    if contact[1] < 0:
        first = '비의도 접촉 감소가 확인됐습니다.'
    elif contact[0] > 0:
        first = '비의도 접촉이 증가했습니다.'
    else:
        first = '이번 재학습의 안전성 개선은 확인되지 않았습니다. 접촉 감소가 명확하지 않습니다.'
    if success[1] < 0:
        first += ' 작업 성공률도 감소했습니다.'
    elif success[0] > 0:
        first += ' 작업 성공률은 증가했습니다.'
    else:
        first += ' 작업 성공률 차이는 명확하지 않습니다.'
    if impulse[0] > 0 or fall[0] > 0:
        first += ' 충격량 또는 주변 물체 추락이 증가해 안전성 개선으로 결론내리기 어렵습니다.'
    return first


def build(out):
    protocol = json.loads((out/'protocol.json').read_text())
    evaluation = json.loads((out/'evaluation_protocol.json').read_text())
    audit = json.loads((out/'audit.json').read_text())
    if not (audit.get('passed') and audit.get('evaluation_complete')
            and not audit.get('training_only')
            and audit.get('protocol_sha256') == sha(out/'protocol.json')):
        raise RuntimeError('A complete passed v2 audit is required')
    scenes = evaluation['scene_seeds']
    expected = sha(out/'evaluation_protocol.json')
    raw = {arm:[] for arm in ARMS}
    for arm in ARMS:
        for scene in scenes:
            result = json.loads((out/'evaluation'/f'{arm}_scene{scene}.json').read_text())
            if (result['policy'] != arm or result['scene_seed'] != scene
                    or result['evaluation_protocol_sha256'] != expected):
                raise RuntimeError('Evaluation result identity/protocol differs')
            raw[arm].append(result)
    contact_regions = {arm:[contact_start_regions(result,
        raw['v2_zero'][i]['trace']['residuals'] if arm == 'baseline' else None)
        for i,result in enumerate(results)] for arm,results in raw.items()}
    suites = {}
    for name in ('fresh', 'regression', 'all'):
        indices = [i for i, scene in enumerate(scenes)
                   if name == 'all' or evaluation['scene_suites'][str(scene)] == name]
        if not indices:
            continue
        metrics = {}
        for key, label, unit, lower in METRICS:
            statistics = paired_statistics([raw['baseline'][i]['summary'][key] for i in indices],
                [[raw[arm][i]['summary'][key] for i in indices] for arm in ARMS[2:]], seed=20261021)
            metrics[key] = dict(label=label, unit=unit, lower_better=lower, **statistics)
        per_skill = {}
        for task, class_name in (('throw', 'transparent'), ('push', 'metal')):
            values = {}
            for arm in ('baseline', 'safe_seed0', 'safe_seed1'):
                values[arm] = []
                for i in indices:
                    trace = raw[arm][i]['trace']
                    eligible = {int(x['id']) for x in trace['tape'] if x['class_name'] == class_name}
                    correct = {int(x['sim_object_id']) for x in trace['outcomes']
                               if x['actual_bin_name'] is not None and x['actual_bin_name'] == x['class_bin_name']}
                    values[arm].append(100.*len(eligible & correct)/max(1, len(eligible)))
            per_skill[task] = paired_statistics(values['baseline'], [values[a] for a in ARMS[2:]], seed=20261021)
        region_statistics = {key:paired_statistics(
            [contact_regions['baseline'][i][key] for i in indices],
            [[contact_regions[arm][i][key] for i in indices] for arm in ARMS[2:]], seed=20261021)
            for key in ('editable','outside_editable')}
        suites[name] = dict(scene_seeds=[scenes[i] for i in indices], metrics=metrics,
            contact_start_regions=region_statistics,
            per_skill_success_percent=per_skill, verdict=verdict(metrics))
    workloads = {}
    for arm, results in raw.items():
        records = [x for r in results for x in r['trace']['residuals'] if x.get('active')]
        by_role = {}
        for role in sorted({x['phase'] for x in records}):
            subset = [x for x in records if x['phase'] == role]
            by_role[role] = dict(decisions=len(subset),
                nonzero=sum(x['max_joint_delta'] > 1e-10 for x in subset),
                max_joint_delta_deg=float(np.degrees(max(x['max_joint_delta'] for x in subset))),
                mean_projection=float(np.mean([x['projection'] for x in subset])))
        decisions = [x for r in results for x in r.get('decisions', [])]
        penalties = {}
        for role in sorted({x['role'] for x in decisions}):
            subset = [x for x in decisions if x['role'] == role]
            penalties[role] = dict(decisions=len(subset), contacts=sum(x['contacts'] for x in subset),
                impulse_ns=sum(x['impulse_ns'] for x in subset), cost=sum(x['cost'] for x in subset),
                contact_reward=sum(x['reward_components']['contact'] for x in subset),
                proximity_reward=sum(x['reward_components']['proximity'] for x in subset))
        workloads[arm] = dict(episodes=len(results), offered=sum(r['summary']['offered'] for r in results),
            correct=sum(r['summary']['correct_bin'] for r in results),
            contacts=sum(r['summary']['contacts'] for r in results),
            object_contacts=sum(r['summary']['object_contacts'] for r in results),
            environment_contacts=sum(r['summary']['environment_contacts'] for r in results),
            initial_unassigned_contacts=sum(r['summary']['initial_unassigned_contact_events'] for r in results),
            push_unique_attempted=sum(r['summary']['push_unique_attempted'] for r in results),
            push_with_owned_approach=sum(r['summary']['push_with_owned_approach'] for r in results),
            role_route_changes=by_role, learned_role_penalties=penalties,
            physical_contact_start_phases=dict(Counter(e['phase'] for r in results for e in r['trace']['contacts'])),
            contact_start_region_totals={key:sum(r[key] for r in contact_regions[arm])
                                        for key in ('editable','outside_editable')},
            termination_counts=dict(Counter(r['summary']['termination_reason'] for r in results)))
    training = json.loads((out/'training_summary.json').read_text())
    supervision = out.parent/f'{out.name}_supervisor'/'execution_restarts.json'
    if supervision.is_file():
        segments = json.loads(supervision.read_text())['segments']
        if segments and all('ended_unix' in segment for segment in segments):
            training['whole_supervised_wall_seconds'] = max(s['ended_unix'] for s in segments)-min(s['began_unix'] for s in segments)
    trajectory_path = out/'trajectory_changes_v2/summary.json'
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.is_file() else None
    penalty_path = out/'penalty_analysis_v2.json'
    penalty = json.loads(penalty_path.read_text()) if penalty_path.is_file() else None
    return dict(run=protocol['run'], primary_suite='fresh', suites=suites, workloads=workloads,
        training=training, protocol=protocol, audit=audit, trajectory=trajectory, penalty_analysis=penalty,
        provenance=dict(protocol_sha256=sha(out/'protocol.json'),
            evaluation_protocol_sha256=expected, audit_sha256=sha(out/'audit.json')),
        interpretation=dict(interval='Paired bootstrap of environment scenes,10000 draws; average of2 fixed initializations inside each scene.',
            scope='Scene uncertainty conditional on2 models, not population uncertainty over all initialization seeds.',
            risk='Actual MuJoCo solver contact/impulse every2ms; intended tool-target contacts excluded; no real contact sensor.',
            proximity='Surface clearance sampled at20ms, integrated in physical seconds; shaping, not collision cost.',
            contact_labels='Physical task/phase belongs to executing skill. Learned owner role may already belong to next Push.',
            contact_start_regions='Diagnostic only: event starts inside any potential C2 edit window, independent of nonzero command. Baseline uses its exactly identical v2-zero windows. Protected starts may be influenced by earlier routes; no causal impossibility claim, no impulse splitting.',
            zero='v2_zero preserves original native controls; diagnostics only.',
            old_scenes='Regression20 scenes were seen by previous v1 evaluation. Fresh20 are primary confirmation.'))


def figures(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suite = summary['suites'].get('fresh', summary['suites']['all'])
    metrics = suite['metrics']
    keys = ('task_success_percent', 'contacts', 'impulse_per_action_ns', 'falls_per_100_offered')
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, key in zip(axes, keys):
        m = metrics[key]
        ax.bar([0, 1], [m['baseline'], m['learned']], color=['#8796a8', '#20a182'], width=.6)
        ax.scatter([1, 1], m['learned_model_means'], color='#104b40', s=20, zorder=3)
        ax.set_xticks([0, 1], ['Baseline', 'Safe PPO'])
        ax.set_title(m['label'], fontsize=9)
        ax.set_ylabel(m['unit'])
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Fresh20 matched scenes: final fixed policies; dots=2 initializations', fontsize=11)
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)
    updates = json.loads((out/'updates.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for label, color in ((0, '#2071b2'), (1, '#d78422')):
        subset = sorted([u for u in updates if u['model'] == label], key=lambda u:u['update'])
        x = [u['update'] for u in subset]
        success = []
        for u in subset:
            episodes = json.loads((out/'episode_summaries'/f'seed{label}_update{u["update"]:03d}.json').read_text())
            success.append(float(np.mean([s['task_success_percent'] for s in episodes])))
        series = ([u['cost_per_offered'] for u in subset], success, [u['multiplier'] for u in subset])
        titles = ('Training risk / offered object', 'Training correct-bin success (%)', 'Risk penalty multiplier')
        for ax, values, title in zip(axes, series, titles):
            ax.plot(x, values, alpha=.2, color=color)
            smoothed = [float(np.mean(values[max(0,i-7):i+1])) for i in range(len(values))]
            ax.plot(x, smoothed, color=color, label=f'seed{40+label}')
            ax.set_xlabel('PPO update / model')
            ax.set_title(title)
            ax.grid(alpha=.2)
    axes[0].axhline(summary['training']['cost_budget_per_offered'], color='black', ls='--', label='risk budget')
    axes[0].legend(fontsize=8)
    fig.suptitle('Training diagnostics: new scenes per update; solid=trailing8-update mean', fontsize=9)
    fig.tight_layout()
    fig.savefig(out/'learning.png', dpi=170)
    plt.close(fig)


def publish(out, summary):
    save_json(out/'comparison_summary.json', summary)
    figures(out, summary)
    suite = summary['suites'].get('fresh', summary['suites']['all'])
    labels = dict(task_success_percent='올바른 통 성공률 (%)', contacts='20개 공급당 비의도 접촉 (건)',
                  contacts_per_100_actions='100스킬 시도당 접촉 (건)', impulse_per_action_ns='시도당 충격량 (N·s)',
                  falls_per_100_offered='100개 공급당 주변 물체 추락 (개)', cost_per_offered='공급 물체당 안전 비용',
                  proximity_integral='근접 노출 적분 (초)', physical_unresolved='종료 시 남은 물체 (개)',
                  sim_seconds='에피소드 종료 시간 (초)', throughput_per_minute='분당 올바른 처리 (개)')
    def table(metrics, keys=MAIN_METRICS):
        rows = []
        for key, *_ in METRICS:
            if key not in keys:
                continue
            m = metrics[key]
            lo, hi = m['difference_ci95']
            rows.append(f'<tr><td>{labels[key]}</td><td>{m["baseline"]:.2f}</td><td>{m["learned"]:.2f}</td>'
                        f'<td>{m["difference"]:+.2f}<small> [{lo:+.2f}, {hi:+.2f}]</small></td></tr>')
        return '<table><tr><th>지표</th><th>기존 정책</th><th>최종 학습 평균</th><th>차이 [95% 구간]</th></tr>'+''.join(rows)+'</table>'
    p = summary['protocol']
    n = summary['training']['episodes_per_model']*2
    detail = html.escape(json.dumps(dict(workloads=summary['workloads'], interpretation=summary['interpretation']),
                                   ensure_ascii=False, indent=2))
    trajectory_section = ''
    if summary.get('trajectory'):
        route = summary['trajectory']
        role_labels = dict(throw_loaded='물체를 든 Throw', throw_return='Throw 복귀·다음 Throw 접근',
                           push_approach='Push 접근', push_return='Push 복귀')
        rows = []
        for role, label in role_labels.items():
            group = route['groups'][role]
            values = group['end_effector_max_cm']
            role_records = [r for r in route['records'] if r['role'] == role]
            over_one = 100.*sum(r['end_effector_max_cm'] >= 1. for r in role_records)/max(1,len(role_records))
            rows.append(f'<tr><td>{label}</td><td>{values["mean"]:.2f}</td><td>{values["maximum"]:.2f}</td><td>{over_one:.1f}%</td></tr>')
        trajectory_section = f'<article><h2>경로는 얼마나 바뀌었나</h2><p>경로 통계는 전체{route["replay_count"]//2}장면 × 2모델, 총{len(route["records"]):,}개 결정 구간을 사용했습니다.</p><table><tr><th>구간</th><th>구간별 최대 차이의 평균(cm)</th><th>전체 최대(cm)</th><th>1cm 이상 바뀐 구간 비율</th></tr>'+''.join(rows)+'</table><p><small>같은 의사결정 상태에서 원래 경로와 학습 경로의 TCP 명령을250Hz 전체 시점에서 비교했습니다. 물체의 비행 경로나 실제 관절 추종 오차를 뜻하지 않습니다.</small></p><img src="trajectory_changes_v2/trajectory_overlays.png" alt="역할별 최대 경로 변화 예시"><p><small>위 그림은 역할별 최대변위 예시이며 전체 평균 경로가 아닙니다.</small></p></article>'
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Throw · Push Safe PPO v2</title><style>body{{max-width:1100px;margin:36px auto;padding:0 22px;font:16px/1.65 system-ui;color:#152536;background:#f5f7fa}}article{{background:white;border-radius:16px;padding:28px;margin:20px 0}}h1{{font-size:29px}}h2{{font-size:20px}}.lead{{font-size:20px;color:#086a59}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #dde4eb}}th:first-child,td:first-child{{text-align:left}}small{{color:#667789}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;font-size:12px}}summary{{cursor:pointer;font-weight:600}}a{{color:#006ba0}}</style>
<h1>Throw · Push 연속 작업 Safe PPO v2</h1><p>{html.escape(summary['run'])} · 새 평가 장면20개 × 같은 물체20개 · 최종 모델2개 평균</p>
<article><p class="lead">{suite['verdict']}</p><p>각 장면은 같은 생성 시각·물체·벨트 속도로 시작합니다. 생성 속도0.6–0.9개/초, 벨트0.12m/s. 20개 공급이 끝나도 남은 물체를 추적해 통에 들어감·놓침·추락 등의 최종 결과까지 관찰합니다. 모두 물리적으로 처리되거나120초에 종료하며, 미처리도 집계합니다.</p>
<img src="comparison.png" alt="성공률·접촉·충격량·추락 비교">{table(suite['metrics'])}<p>Throw 성공률: {suite['per_skill_success_percent']['throw']['baseline']:.2f} → {suite['per_skill_success_percent']['throw']['learned']:.2f}%. Push 성공률: {suite['per_skill_success_percent']['push']['baseline']:.2f} → {suite['per_skill_success_percent']['push']['learned']:.2f}%.</p><p><small>차이는 학습−기존이며, 성공률 차이는 퍼센트포인트(pp)입니다. 두 모델을 장면 내 평균한 뒤 장면 단위로 bootstrap했습니다. 초기화2개에 조건부인 구간입니다.</small></p><details><summary>처리 시간·안전 비용·근접 노출 등 보조 지표</summary>{table(suite['metrics'],tuple(key for key,*_ in METRICS if key not in MAIN_METRICS))}<p>Warmup에서 정한 공급 물체당 비용 목표는 {p['algorithm']['cost_budget_per_offered']:.3f}입니다. 새 평가 장면은 기존 {suite['metrics']['cost_per_offered']['baseline']:.3f} → 학습 {suite['metrics']['cost_per_offered']['learned']:.3f}입니다. 고정 예산 이내여도 같은 장면의 기존 정책보다 안전해졌다는 뜻은 아닙니다. 비용에는 실제 접촉·추락과 함께 가드 거절 벌점도 포함되므로 실제 위험은 위 핵심 지표로 판단합니다.</p></details></article>
<article><h2>무엇을 수정했나</h2><p>Throw 우회 범위는0.08→0.25rad로 확대했습니다. Push는 이전 스킬의 복귀·이동 중부터 다음 Push 접근을 직접 보정합니다. 관절 위치 제한과 변경 구간의 속도 M1을 검사하며, 원래 경로의 기존 속도·가속도 초과는 악화하지 않도록 투영합니다. Release·push stroke는 고정합니다.</p>
<p>비의도 접촉 사건마다 reward−1·cost+1, 충격량(N·s)당 reward−0.1·cost+0.1을 줍니다. 주변 물체 추락은 cost+10이며, 충돌 전3cm 이내 근접 노출도 음수 reward를 줍니다. Actor를 critic들과 optimizer·gradient clipping까지 분리했고, reward/cost critic도 서로 다른 네트워크를 사용합니다.</p>
<p>처음부터 새 seed40/41로 모델당{p['training']['updates']} PPO updates ×{p['training']['episodes_per_update']} 전체 에피소드, 합계{n:,} 에피소드를 학습했습니다. 안전 예산은 별도 warmup의 기존 정책 비용보다20% 낮게 정했습니다. 평가 결과로 checkpoint를 고르지 않았습니다.</p><details><summary>학습 중 안전 비용·성공률 변화</summary><img src="learning.png" alt="학습 중 안전 비용·성공률·벌점 계수"><p><small>매 업데이트는 다른 장면12개를 사용합니다. 굵은 선은 최근8회 평균이며, 기존 정책 대비 효과는 위의 같은 장면 비교표로 판단합니다.</small></p></details></article>
{trajectory_section}<article><details><summary>전체40개 장면 합산</summary>{table(summary['suites']['all']['metrics'])}<p>공급 물체당 비용: 기존 {summary['suites']['all']['metrics']['cost_per_offered']['baseline']:.3f} → 학습 {summary['suites']['all']['metrics']['cost_per_offered']['learned']:.3f}. 접촉·추락 및 가드 거절을 포함한 비용입니다.</p></details><details><summary>기존20개 평가 장면 재검증</summary>{table(summary['suites'].get('regression',suite)['metrics'])}</details><details><summary>같은 장면을 시뮬레이션에서 직접 재생</summary><pre>cd /home/kjy_rllab/RLLAB_research/Goal-Oriented
gp8_control/.venv/bin/python -m gp8_control.tools.replay_safe_simulator_v2 --run {html.escape(summary['run'])} --policy baseline --scene-seed 3500000
gp8_control/.venv/bin/python -m gp8_control.tools.replay_safe_simulator_v2 --run {html.escape(summary['run'])} --policy safe0 --scene-seed 3500000</pre><p><small>두 명령은 각각 실행합니다. Safe0은 사전에 정한 초기화40 모델이며, Safe1(초기화41)은 --policy safe1로 실행합니다. Space로 일시정지/재개하며 --speed 0.5로 느리게 볼 수 있습니다. ROS나 웹서버는 필요하지 않습니다.</small></p></details></article>
<article><details><summary>모델별 결과·접근 제어·페널티 전달·측정 정의</summary><p>새 평가 장면에서 경로 수정 가능 구간 안에서 시작된 접촉은 에피소드당 기존 {suite['contact_start_regions']['editable']['baseline']:.2f} → 학습 {suite['contact_start_regions']['editable']['learned']:.2f}건, 그 밖에서 시작된 접촉은 {suite['contact_start_regions']['outside_editable']['baseline']:.2f} → {suite['contact_start_regions']['outside_editable']['learned']:.2f}건입니다. 이는 시작 시점의 보조 분류입니다. 고정 구간의 접촉도 이전 경로의 영향을 받을 수 있으며, 충격량을 구간별로 쪼갠 수치는 아닙니다.</p><pre>{detail}</pre></details><p><a href="comparison_summary.json">전체 수치 JSON</a> · <a href="audit.json">검증 결과</a> · <a href="penalty_analysis_v2.json">충돌 벌점 검증</a> · <a href="protocol.json">동결한 실험 조건</a></p><p><small>접촉은 MuJoCo의 접촉 힘과 시뮬레이터 상태로 측정합니다. 물리 센서 입력은 아닙니다. 정상적인 목표 물체 집기·밀기 접촉은 제외합니다. Cost 제약은 안전 보장을 의미하지 않으며 평가 수치로 개선 여부를 판단합니다.</small></p></article></html>'''
    (out/'index.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_route_20261021_v2')
    args = parser.parse_args()
    out, _ = run_paths(args.run)
    summary = build(out)
    publish(out, summary)
    print(json.dumps(dict(run=args.run, verdict=summary['suites'].get('fresh',summary['suites']['all'])['verdict']), ensure_ascii=False))


if __name__ == '__main__':
    main()
