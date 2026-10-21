"""Compact paired baseline/Safe PPO report for the native feature simulator."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path

import numpy as np

from gp8_control.tools.train_safe_simulator import ROOT, save_json, sha

METRICS = (
    ('task_success_percent', '올바른 쓰레기통 성공률', '%', False),
    ('contacts_per_100_actions', '비의도 접촉 / 100스킬 시도', '건', True),
    ('falls_per_100_offered', '주변 물체 추락 / 100물체', '개', True),
    ('impulse_per_action_ns', '접촉 충격량 / 스킬 시도', 'N·s', True),
    ('contacts_per_20_offered', '20개 공급 작업당 비의도 접촉', '건', True),
    ('actions', '20개 공급 작업당 스킬 실행 시도', '회', False),
    ('throughput_per_minute', '올바르게 처리한 물체 / 분', '개', False),
    ('sim_seconds', '에피소드 종료까지 시간', '초', True),
    ('target_reach_percent', '10개 성공 도달률', '%', False),
    ('target_capped_seconds', '10개 성공 제한시간 점수', '초', True),
    ('physical_unresolved', '종료 시 물리적으로 남은 물체', '개', True),
    ('active_count_mean', '평균 동시 물체 수', '개', False),
    ('active_count_max', '장면별 최대 동시 물체 수', '개', False),
    ('admission_max_delay', '장면별 최대 생성 지연', '초', True),
)


def values_from_result(result, config):
    summary = result['summary']
    values = {key:float(summary[key]) for key, *_ in METRICS if key in summary}
    reached = summary['target_success_time'] is not None
    values['target_reach_percent'] = 100. if reached else 0.
    values['contacts_per_20_offered'] = 20.*summary['contacts']/summary['offered']
    values['target_capped_seconds'] = float(summary['target_success_time'] if reached else config['time_limit'])
    return values


def paired_statistics(baseline, learned, *, repetitions=10000, seed=20261020):
    baseline, learned = np.asarray(baseline, float), np.asarray(learned, float)
    if baseline.ndim != 1 or learned.shape != (2, len(baseline)) or not len(baseline):
        raise ValueError('Expected complete scenes with exactly two learned models')
    if not np.isfinite(baseline).all() or not np.isfinite(learned).all():
        raise ValueError('Metrics must be finite')
    learned_scene_mean = learned.mean(axis=0)
    difference = learned_scene_mean-baseline
    indices = np.random.default_rng(seed).integers(0, len(baseline), size=(repetitions, len(baseline)))
    samples = difference[indices].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    first, second = float(baseline.mean()), float(learned_scene_mean.mean())
    return dict(baseline=first, learned=second, difference=float(difference.mean()),
        difference_ci95=[float(lo), float(hi)],
        relative_change_percent=(100.*(second-first)/first if abs(first) > 1e-12 else None),
        learned_model_means=learned.mean(axis=1).tolist(),
        baseline_by_scene=baseline.tolist(), learned_model_by_scene=learned.tolist(),
        learned_mean_by_scene=learned_scene_mean.tolist())


def build_summary(out, *, require_audit=True):
    protocol_path = out/'protocol.json'
    evaluation_path = out/'evaluation_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    audit_path = out/'audit.json'
    if require_audit:
        audit = json.loads(audit_path.read_text())
        if not (audit.get('passed') and audit.get('evaluation_complete')
                and audit.get('protocol_sha256') == sha(protocol_path)):
            raise RuntimeError('A complete passed audit is required before publishing results')
    else:
        audit = dict(passed=False, development_fixture=True)
    scenes = evaluation['scene_seeds']
    config = evaluation['episode_config']
    arms = ('baseline', 'safe_seed0', 'safe_seed1')
    raw = {name:[] for name in arms}
    for name in arms:
        for scene in scenes:
            result = json.loads((out/'evaluation'/f'{name}_scene{scene}.json').read_text())
            if result['scene_seed'] != scene or result['policy'] != name:
                raise ValueError('Result ordering/identity differs from the fixed evaluation protocol')
            raw[name].append(result)
    metrics = {}
    by_arm = {name:[values_from_result(result, config) for result in results] for name, results in raw.items()}
    for key, label, unit, lower_better in METRICS:
        values = paired_statistics([v[key] for v in by_arm['baseline']],
            [[v[key] for v in by_arm[name]] for name in arms[1:]])
        metrics[key] = dict(label=label, unit=unit, lower_better=lower_better, **values)
    terminations = {name:dict(Counter(r['summary']['termination_reason'] for r in results))
                    for name, results in raw.items()}
    workload = {name:dict(offered=sum(r['summary']['offered'] for r in results),
        correct=sum(r['summary']['correct_bin'] for r in results),
        physical_actions=sum(r['summary']['actions'] for r in results),
        contacts=sum(r['summary']['contacts'] for r in results),
        collateral_falls=sum(r['summary']['collateral_falls'] for r in results),
        unadmitted=sum(r['summary']['not_admitted'] for r in results),
        unresolved_outcomes=sum(r['summary']['unresolved'] for r in results),
        physical_unresolved=sum(r['summary']['physical_unresolved'] for r in results),
        reached_target=sum(r['summary']['target_success_time'] is not None for r in results),
        active_count_mean=float(np.mean([r['summary']['active_count_mean'] for r in results])),
        active_count_max=max(r['summary']['active_count_max'] for r in results),
        nonzero_decisions=sum(r['summary']['nonzero_decisions'] for r in results),
        policy_decisions=sum(r['summary']['policy_decisions'] for r in results),
        max_joint_delta_rad=max(r['summary']['max_joint_delta_rad'] for r in results))
        for name, results in raw.items()}
    for name, results in raw.items():
        active_residuals = [record for result in results for record in result['trace']['residuals']
                            if record.get('active', False)]
        workload[name]['projected_decisions'] = sum(
            float(record.get('projection', 1.)) < 1. - 1e-9 for record in active_residuals)
        workload[name]['rejected_decisions'] = sum(bool(record.get('rejected', False))
                                                     for record in active_residuals)
        workload[name]['mean_projection'] = float(np.mean(
            [record.get('projection', 1.) for record in active_residuals])) if active_residuals else 1.
        workload[name]['push_transfer_only_decisions'] = sum(
            record.get('task') == 'push' and record.get('active', False)
            and record.get('prepare_end') is not None
            and not record.get('prepare_window_active', False)
            and record.get('transfer_window_active', False)
            for result in results for record in result['trace']['residuals'])
        workload[name]['contact_phase_counts'] = dict(Counter(
            event['phase'] for result in results for event in result['trace']['contacts']))
    per_skill = {}
    for task, class_name in (('throw', 'transparent'), ('push', 'metal')):
        scene_rates = {}
        for name, results in raw.items():
            values = []
            for result in results:
                eligible = {int(item['id']) for item in result['trace']['tape'] if item['class_name'] == class_name}
                correct = {int(item['sim_object_id']) for item in result['trace']['outcomes']
                           if item['actual_bin_name'] is not None and item['actual_bin_name'] == item['class_bin_name']}
                values.append(100.*len(eligible & correct)/max(len(eligible), 1))
            scene_rates[name] = values
        per_skill[task] = paired_statistics(scene_rates['baseline'], [scene_rates[name] for name in arms[1:]])
    survival = {name:[dict(scene_seed=scene, reached=r['summary']['target_success_time'] is not None,
        observed_seconds=(r['summary']['target_success_time'] if r['summary']['target_success_time'] is not None
                          else r['summary']['sim_seconds']),
        censor_or_failure_reason=(None if r['summary']['target_success_time'] is not None
                                 else r['summary']['termination_reason']),
        capped_score_seconds=(r['summary']['target_success_time'] if r['summary']['target_success_time'] is not None
                              else config['time_limit'])) for scene, r in zip(scenes, results)]
        for name, results in raw.items()}
    training = json.loads((out/'training_summary.json').read_text())
    return dict(run=protocol['run'], experiment_date='2026-10-20', audit_passed=audit['passed'], scene_seeds=scenes,
        scene_count=len(scenes), learned_models=2, episode_config=config,
        offered_per_arm_per_model=len(scenes)*config['offer_count'], metrics=metrics,
        per_skill_success_percent=per_skill, termination_counts=terminations,
        raw_workload_totals=workload, target_time_observations=survival,
        training=dict(initialization_seeds=protocol['model_initialization_seeds'],
            updates_per_model=training['updates_per_model'], episodes_per_model=training['episodes_per_model'],
            episodes_per_update=protocol['training']['episodes_per_update'],
            warmup_episodes=protocol['training']['warmup_episodes'],
            decisions_per_model=[r['cumulative_decisions'] for r in training['seed_results']],
            final_batch_mean_cost=[float(r['mean_cost']) for r in training['seed_results']],
            final_batch_multiplier=[float(r['multiplier']) for r in training['seed_results']],
            cost_limit_per_decision=float(protocol['algorithm']['cost_limit_per_decision']),
            final_batch_meets_cost_limit=[float(r['mean_cost']) <= float(protocol['algorithm']['cost_limit_per_decision'])
                                         for r in training['seed_results']]),
        definitions=dict(primary_success='correct native class-bin outcomes / all20 offered objects in each scene',
            aggregation='arithmetic mean of scene rates; two fixed learned models averaged within scene',
            intervals='paired percentile bootstrap,10000 resamples of environment-seed clusters;95% confidence interval for learned-minus-baseline',
            scope='intervals describe sampled scene uncertainty conditional on the two trained models; no estimate of all initialization variability',
            contact='MuJoCo solver contact, force>0.1N, same-pair gaps>0.03s create a new event; normal tool-target grip/stroke excluded; handled-object state is a causal attribution heuristic',
            falls='previously contacted non-target objects crossing0.08m below belt height inside work zone; bin footprints are not excluded',
            arrivals='same immutable external20-object tape, common physical-clearance FIFO; actual admission may differ with policy',
            terminal='all offered objects physically resolved, or observed120s budget; all unfinished offered IDs counted',
            ten_success='reach rate uses every scene; time score uses observed reach time or120s if not reached; not a conditional mean or Kaplan-Meier estimate',
            task_completion='episode elapsed time includes120s deadlines with unfinished work; physical resolution includes misses/falls; it is not time to20 successful bin placements'),
        provenance=dict(training_protocol_sha256=sha(protocol_path), evaluation_protocol_sha256=sha(evaluation_path),
                        audit_sha256=sha(audit_path) if audit_path.exists() else None,
                        checkpoint_sha256=evaluation['checkpoint_sha256']))


def verdict(summary):
    m = summary['metrics']
    contacts, success = m['contacts_per_100_actions'], m['task_success_percent']
    c_lo, c_hi = contacts['difference_ci95']
    s_lo, s_hi = success['difference_ci95']
    if c_hi < 0:
        first = '비의도 접촉 감소가 확인됐습니다.'
    elif c_lo > 0:
        first = '이번 학습에서는 비의도 접촉이 증가했습니다.'
    else:
        first = '이번 학습의 접촉 개선은 통계적으로 명확하지 않습니다.'
    if s_hi < 0:
        second = '작업 성공률도 감소해 기존 정책보다 낫다고 판단하기 어렵습니다.'
    elif s_lo > 0:
        second = '작업 성공률은 증가했습니다.'
    else:
        second = '작업 성공률 차이는 명확하지 않습니다.'
    if m['falls_per_100_offered']['difference_ci95'][0] > 0:
        second += ' 주변 물체 추락 증가도 관측됐습니다.'
    return first, second


def number(value, digits=2):
    return f'{value:,.{digits}f}'


def delta_text(metric):
    difference = metric['difference']
    lo, hi = metric['difference_ci95']
    unit = 'pp' if metric['unit'] == '%' else metric['unit']
    return f'{difference:+.2f} {unit} · 95% CI [{lo:+.2f}, {hi:+.2f}]'


def write_csv(out, summary):
    with (out/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['metric', 'label', 'unit', 'baseline', 'learned',
            'difference', 'ci95_low', 'ci95_high', 'relative_change_percent'])
        writer.writeheader()
        for key, metric in summary['metrics'].items():
            writer.writerow(dict(metric=key, label=metric['label'], unit=metric['unit'],
                baseline=metric['baseline'], learned=metric['learned'], difference=metric['difference'],
                ci95_low=metric['difference_ci95'][0], ci95_high=metric['difference_ci95'][1],
                relative_change_percent=metric['relative_change_percent']))


def write_figure(out, summary):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/feature-safe-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.facecolor':'white', 'figure.facecolor':'white'})
    keys = ('task_success_percent', 'contacts_per_100_actions', 'falls_per_100_offered', 'impulse_per_action_ns')
    titles = ('Correct-bin success (%)', 'Unintended contacts /100 skill attempts',
              'Collateral falls /100 offered', 'Contact impulse /skill attempt (N s)')
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), constrained_layout=True)
    for ax, key, title in zip(axes.ravel(), keys, titles):
        m = summary['metrics'][key]
        baseline, learned = np.asarray(m['baseline_by_scene']), np.asarray(m['learned_mean_by_scene'])
        for a, b in zip(baseline, learned):
            ax.plot([0, 1], [a, b], color='#cbd5e1', lw=.65, alpha=.7, zorder=1)
        ax.scatter(np.zeros(len(baseline)), baseline, color='#475569', s=19, alpha=.65, zorder=2)
        ax.scatter(np.ones(len(learned)), learned, color='#0f766e', s=19, alpha=.65, zorder=2)
        ax.scatter([0, 1], [m['baseline'], m['learned']], marker='D', s=100,
                   color=['#0f172a', '#0f766e'], edgecolor='white', linewidth=1.1, zorder=3)
        ax.set_xticks([0, 1], ['Baseline', 'Safe PPO (2-model mean)'])
        ax.set_xlim(-.25, 1.25)
        ax.set_title(title, loc='left', weight='bold', pad=27)
        ax.grid(axis='y', alpha=.15)
        if min(float(baseline.min()), float(learned.min())) >= 0:
            ax.set_ylim(bottom=0)
        lo, hi = m['difference_ci95']
        ax.text(0., 1.02, f"Difference {m['difference']:+.2f}; 95% CI [{lo:+.2f}, {hi:+.2f}]",
                transform=ax.transAxes, va='bottom', fontsize=8.5)
    fig.suptitle(f"Native feature/simulator · {summary['scene_count']} paired workloads · 20 objects each",
                 fontsize=14, weight='bold')
    fig.savefig(out/'comparison.png', dpi=170)
    plt.close(fig)


def write_html(out, summary):
    escape = html.escape
    first, second = verdict(summary)
    metrics = summary['metrics']
    config = summary['episode_config']
    cards = []
    for key in ('task_success_percent', 'contacts_per_100_actions', 'falls_per_100_offered'):
        m = metrics[key]
        better = m['difference'] < 0 if m['lower_better'] else m['difference'] > 0
        cards.append(f'''<article class="card"><div class="label">{escape(m['label'])}</div>
        <div class="values">{number(m['baseline'])}<span>→</span>{number(m['learned'])}<small>{m['unit']}</small></div>
        <div class="delta {'better' if better else 'worse'}">{escape(delta_text(m))}</div></article>''')
    rows = []
    main_keys = ('task_success_percent', 'contacts_per_100_actions', 'falls_per_100_offered',
                 'impulse_per_action_ns', 'contacts_per_20_offered', 'actions', 'throughput_per_minute')
    for key in main_keys:
        m = metrics[key]
        rows.append(f"<tr><th>{escape(m['label'])}</th><td>{number(m['baseline'])} {m['unit']}</td><td>{number(m['learned'])} {m['unit']}</td><td>{escape(delta_text(m))}</td></tr>")
    scene_count = summary['scene_count']
    training = summary['training']
    term = summary['termination_counts']
    counts_rows = []
    for label, reason in (('자연 종료', 'workload_resolved'), ('120초 시간 제한 종료', 'time_limit')):
        base = term['baseline'].get(reason, 0)
        learned = np.mean([term[name].get(reason, 0) for name in ('safe_seed0', 'safe_seed1')])
        counts_rows.append(f'<tr><th>{label}</th><td>{base} / {scene_count}</td><td>{learned:g} / {scene_count}</td></tr>')
    workload = summary['raw_workload_totals']
    reach_base = workload['baseline']['reached_target']
    reach_arms = [workload[name]['reached_target'] for name in ('safe_seed0', 'safe_seed1')]
    skill_rows = ''.join(f"<tr><th>{name.title()} 대상 성공률</th><td>{number(m['baseline'])}%</td><td>{number(m['learned'])}%</td></tr>"
                         for name,m in summary['per_skill_success_percent'].items())
    episode_rows = ''.join(f"<tr><th>{metrics[key]['label']}</th><td>{number(metrics[key]['baseline'])} {metrics[key]['unit']}</td><td>{number(metrics[key]['learned'])} {metrics[key]['unit']}</td></tr>"
        for key in ('sim_seconds', 'target_reach_percent', 'target_capped_seconds', 'physical_unresolved'))
    condition_rows = ''.join(f"<tr><th>{metrics[key]['label']}</th><td>{number(metrics[key]['baseline'])} {metrics[key]['unit']}</td><td>{number(metrics[key]['learned'])} {metrics[key]['unit']}</td></tr>"
        for key in ('active_count_mean', 'active_count_max', 'admission_max_delay'))
    individual_rows = ''.join(f"<tr><th>{metrics[key]['label']}</th><td>{number(metrics[key]['learned_model_means'][0])} {metrics[key]['unit']}</td><td>{number(metrics[key]['learned_model_means'][1])} {metrics[key]['unit']}</td></tr>"
        for key in ('task_success_percent', 'contacts_per_100_actions', 'falls_per_100_offered', 'impulse_per_action_ns'))
    learned_raw = [workload[name] for name in ('safe_seed0', 'safe_seed1')]
    nonzero_counts = [item['nonzero_decisions'] for item in learned_raw]
    max_residual_degrees = [float(np.rad2deg(item['max_joint_delta_rad'])) for item in learned_raw]
    push_transfer_counts = [item['push_transfer_only_decisions'] for item in learned_raw]
    projected_counts = [item['projected_decisions'] for item in learned_raw]
    rejected_counts = [item['rejected_decisions'] for item in learned_raw]
    mean_projections = [item['mean_projection'] for item in learned_raw]
    final_costs = training['final_batch_mean_cost']
    final_multipliers = training['final_batch_multiplier']
    cost_status = ['목표 이하' if met else '목표 초과' for met in training['final_batch_meets_cost_limit']]
    replay = summary['run'] + '_reproduce'
    command_base = 'gp8_control/.venv/bin/python -m gp8_control.tools.'
    reproduction = '\n'.join((
        f"cd {ROOT.parent}",
        f"{command_base}supervise_safe_simulator --fresh --run {replay} --workers 8 --updates {training['updates_per_model']} --episodes {training['episodes_per_update']} --warmup-episodes {training['warmup_episodes']} --updates-per-pool 6 --episode-config gp8_control/runs/{summary['run']}/episode_config.json",
        f"{command_base}train_safe_simulator eval --run {replay} --scene-seeds {summary['scene_seeds'][0]}:{summary['scene_seeds'][-1]+1} --workers 8",
        f"{command_base}audit_safe_simulator --run {replay} --identity-proof gp8_control/runs/{summary['run']}/implementation_verification.json",
        f"{command_base}report_safe_simulator --run {replay}"))
    diagnostic_section = ''
    diagnostic_path = out/'diagnostics.json'
    if diagnostic_path.is_file():
        diagnostics = json.loads(diagnostic_path.read_text())
        if diagnostics.get('provenance', {}).get('summary_sha256') == sha(out/'summary.json'):
            base_tasks = diagnostics['policies']['baseline']['event_start_task']
            learned_tasks = diagnostics['learned_two_model_average']['event_start_task']
            transfer_percent = diagnostics['learned_two_model_average']['pooled_push_windows']['transfer_only_percent']
            diagnostic_section = f'''<details class="panel"><summary>위험이 개선되지 않은 이유에 대한 기록</summary>
            <p>20개 평가 장면의 접촉 이벤트 충격량 합은 Push에서 {base_tasks['push']['impulse_ns']:.1f}→{learned_tasks['push']['impulse_ns']:.1f}N·s,
            Throw에서 {base_tasks['throw']['impulse_ns']:.1f}→{learned_tasks['throw']['impulse_ns']:.1f}N·s였습니다. 새 정책 값은 두 모델의 평균입니다.
            Push 쪽 감소가 Throw 쪽 증가로 상쇄됐습니다.</p>
            <p class="note">이는 이벤트가 시작한 스킬 기준의 기술 통계입니다. 이벤트가 다음 단계까지 이어지면 그 충격도 최초 라벨에 합산됩니다. 스킬별 인과 효과를 분리한 실험은 아닙니다.</p>
            <p>새 정책 Push 결정의 {transfer_percent:.1f}%는 stroke 이후 이동만 보정할 수 있었습니다. 준비 경로에 대한 제어 권한과 이전 스킬에서 이어지는 동작의 영향이 개선을 제한했을 가능성이 있습니다.</p>
            <p>모델31의 두 장면에서는 Push 동작 완료 기록이 있어도 금속 물체 1개씩 최종 결과가 확정되지 않아 120초에 종료했습니다. 최종 미해결 물체의 좌표는 저장되지 않아 물리적인 걸림 원인을 확정할 수 없습니다.</p>
            <a href="diagnostics.json">단계별 원본 집계·추론 구분 보기</a></details>'''
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>기존 정책과 Safe PPO 비교</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f5f7fa;color:#172033;font-family:system-ui,-apple-system,"Noto Sans CJK KR",sans-serif;line-height:1.55}}
    main{{max-width:1180px;margin:auto;padding:38px 28px 55px}}.eyebrow{{color:#64748b;font-size:13px;letter-spacing:.04em}}h1{{font-size:30px;letter-spacing:-.04em;margin:8px 0 22px}}h2{{font-size:19px;margin:0 0 9px}}p{{margin:8px 0}}.verdict{{padding:24px 26px;border-radius:15px;background:#172e45;color:white}}.verdict p{{color:#dce7f1}}
    .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:20px 0}}.card{{background:white;border:1px solid #e2e8f0;border-radius:13px;padding:20px}}.label{{font-size:14px;color:#526173}}.values{{font-size:31px;font-weight:700;letter-spacing:-.04em;margin:12px 0 8px}}.values span{{font-size:20px;color:#94a3b8;margin:0 12px}}.values small{{font-size:14px;font-weight:400;margin-left:6px}}.delta{{font-size:12px}}.better{{color:#0f766e}}.worse{{color:#9f453f}}
    .panel{{background:white;border:1px solid #e2e8f0;border-radius:13px;padding:23px;margin-top:19px}}.note{{font-size:13px;color:#5e6c7d}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:12px 10px;text-align:left;border-bottom:1px solid #e8edf2}}thead th{{color:#526173;font-weight:600;background:#f8fafc}}tbody th{{font-weight:500}}td{{font-variant-numeric:tabular-nums}}figure{{margin:0}}img{{display:block;width:100%;height:auto}}details{{margin-top:18px}}summary{{cursor:pointer;font-weight:600}}.links{{display:flex;gap:18px;margin-top:20px;font-size:13px}}a{{color:#0f766e}}code{{font-size:12px;background:#f1f5f9;padding:3px 5px;border-radius:3px}}@media(max-width:850px){{.cards{{grid-template-columns:1fr}}main{{padding:24px 16px}}.table-wrap{{overflow:auto}}h1{{font-size:25px}}}}
    </style><main><div class="eyebrow">2026.10.20 · FEATURE/SIMULATOR · 동일한 20개 물체 · {scene_count}개 시드</div>
    <h1>기존 정책과 새 Safe PPO 비교</h1><section class="verdict"><h2>{escape(first)}</h2><p>{escape(second)}</p></section>
    <div class="cards">{''.join(cards)}</div><p class="note">왼쪽은 기존 정책, 오른쪽은 새 정책 두 모델의 평균입니다. 같은 시드 안에서 두 모델을 평균한 뒤 장면 간 평균을 구했습니다. 접촉 횟수는 여러 번의 접촉이 가능하므로 100을 넘을 수 있습니다.</p>
    <section class="panel"><h2>같은 작업량에서의 결과</h2><div class="table-wrap"><table><thead><tr><th>측정 지표</th><th>기존 정책</th><th>새 Safe PPO</th><th>차이 · 새 정책 − 기존 정책</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <p class="note">성공률은 각 장면의 올바른 쓰레기통 처리 수 / 공급 예정 20개입니다. 표는 장면별 비율의 평균이며, 접촉·추락은 정상적인 집기와 Push의 의도한 접촉을 제외합니다. 95% 신뢰구간은 환경 시드를 묶음으로 10,000회 재추출한 paired bootstrap입니다. 모델 초기화 전반의 변동성까지 추정한 구간은 아닙니다.</p></section>
    <section class="panel"><figure><img src="comparison.png" alt="동일 시드에서 성공률, 비의도 접촉, 주변 물체 추락, 접촉 충격량 비교"></figure><p class="note">선 하나는 동일한 물체 배치를 뜻합니다. 오른쪽 점은 해당 시드에서 두 학습 모델을 평균한 값입니다.</p></section>
    <section class="panel"><h2>에피소드는 언제 끝났나</h2><p>20개 물체의 최종 물리 결과가 모두 확정되면 종료합니다. 120초를 넘기면 남은 물체에도 실패·미처리 벌점을 적용하고 종료합니다. 성공한 물체 수로 조기 종료하지 않았습니다.</p><table><thead><tr><th>종료 조건·보조 지표</th><th>기존 정책</th><th>새 Safe PPO · 모델 평균</th></tr></thead><tbody>{''.join(counts_rows)}{episode_rows}{skill_rows}</tbody></table>
    <p class="note">‘결과 확정’에는 놓침·추락도 포함됩니다. 에피소드 종료 시간에는 미처리 물체가 남은 120초 종료도 포함되며, 20개 성공까지 걸린 시간을 의미하지 않습니다. 10개 성공 도달: 기존 {reach_base}/{scene_count}, 새 모델 각각 {reach_arms[0]}/{scene_count}·{reach_arms[1]}/{scene_count}. 미도달 장면을 빼고 시간을 평균하지 않았습니다. 제한시간 점수는 도달한 시간, 미도달이면 120초를 사용한 전 장면 평균이며 실제 소요시간의 비검열 평균이나 생존분석 추정치는 아닙니다.</p></section>
    {diagnostic_section}
    <details class="panel"><summary>모델별 결과·혼잡도·학습 조건</summary><p>새 초기화 seed {training['initialization_seeds']}에서 모델별 {training['updates_per_model']}회 PPO update, {training['episodes_per_model']}개 완전한 연속 작업 에피소드를 학습했습니다. 실제 경로 결정 수는 모델별 {training['decisions_per_model']}회입니다. 기존 가중치는 불러오지 않았습니다.</p>
    <table><thead><tr><th>개별 모델 결과</th><th>초기화 seed30</th><th>초기화 seed31</th></tr></thead><tbody>{individual_rows}</tbody></table>
    <p>최종 학습 배치의 결정당 평균 safety cost는 모델30 <strong>{final_costs[0]:.3f}</strong> ({cost_status[0]}), 모델31 <strong>{final_costs[1]:.3f}</strong> ({cost_status[1]})입니다. 목표는 <strong>{training['cost_limit_per_decision']:.2f}</strong>이며 충돌 확률이 아닌 가중 비용입니다. 최종 λ는 각각 {final_multipliers[0]:.3f}·{final_multipliers[1]:.3f}입니다.</p>
    <p class="note">마지막 학습 배치 하나의 값으로 수렴이나 안전 제약 달성을 판정할 수 없습니다. 이 값은 최종 평가 장면의 cost 측정치가 아니며, 접촉 통계 개선만으로 cost 목표 충족을 주장하지 않습니다.</p>
    <p class="note">0이 아닌 실제 경로 보정 결정: 모델별 {nonzero_counts}회. 실행된 최대 관절 보정은 각각 {max_residual_degrees[0]:.3f}°·{max_residual_degrees[1]:.3f}°입니다. 준비 구간이 짧아 Push 종료 후 이동만 보정한 결정은 각각 {push_transfer_counts}회입니다.</p>
    <p class="note">속도·관절 제한으로 제안 보정을 축소한 결정은 모델별 {projected_counts}회, 완전히 거절한 결정은 {rejected_counts}회입니다. 제안 보정에 적용한 배율 평균은 {mean_projections[0]:.3f}·{mean_projections[1]:.3f}입니다(1은 전체 적용).</p>
    <table><thead><tr><th>실제로 형성된 혼잡도</th><th>기존 정책</th><th>새 Safe PPO</th></tr></thead><tbody>{condition_rows}</tbody></table>
    <p>생성 요청률은 초당 {config['rate_range'][0]}–{config['rate_range'][1]}개, 벨트 속도는 {config['belt_speed']}m/s입니다. 각 정책은 같은 생성 시간·크기·위치·재질의 20개 물체를 받습니다. 입구가 막히면 공통 FIFO 규칙으로 실제 생성이 지연될 수 있습니다.</p><p>MuJoCo의 물리 접촉과 접촉력을 직접 측정합니다. 0.1N 초과 접촉에서 같은 쌍의 간격이 0.03초를 넘으면 새 사건으로 집계합니다. 정책 입력은 물리 상태이며 접촉 센서나 실제 영상 인식 검증이 아닙니다.</p>
    <p>이번 검증은 20개 물체의 유한 작업이며 무한 유입 전체를 검증한 결과는 아닙니다. 투명·금속 물체를 10개씩 번갈아 공급합니다. 두 비교군 모두 브랜치의 공유 시뮬레이터에 physical_belt=True, belt_contact_params="old"를 명시해 실행했습니다. 정확한 시뮬레이터 상태를 입력하고 흡착 확률을 1로 고정했습니다. 실제 센서 오차·흡착 실패 조건으로의 일반화는 별도 검증이 필요합니다.</p>
    <p>기존 경로의 관절1–3에 6개 계수로 작은 보정을 주는 정책입니다. Throw의 집기·release 주변과 Push 실제 stroke, 최종 인계 상태는 보호합니다. 준비 구간이 없는 Push는 종료 후 이동만 수정할 수 있습니다. 최대 제안 보정은 0.08rad이며 속도 조건을 만족하도록 축소됩니다. 이는 새로운 전체 동작 계획을 학습한 정책은 아닙니다.</p>
    <p>물체 간 접촉의 원인 구분에는 조작 중·조작 이력 상태를 이용한 휴리스틱이 포함되며, 센서로 인과를 확정한 것은 아닙니다. Safety cost 목표0.10은 경로 결정당 가중 비용 평균이며 충돌 확률10%나 무충돌 보장을 뜻하지 않습니다.</p>
    <p>주변 물체 추락은 사전에 비의도 접촉을 받은 비대상 물체가 작업 영역에서 벨트보다 8cm 아래로 내려간 경우입니다. 쓰레기통 바닥 영역을 별도로 제외하지 않아, 우연히 쓰레기통에 들어간 주변 물체도 이 지표에 포함될 수 있습니다.</p>
    <p>검증: 초기 가중치·빈 Adam 재구성, warmup 정규화 재계산, 모든 on-policy batch의 이전 가중치 log probability·critic, 고정 마지막 checkpoint, 시드별 동일 공급 물체, 평가 actor 출력과 물리 결과 집계를 확인했습니다.</p>
    <p>학습 도중 worker 메모리 누적으로 프로세스를 재시작했습니다. 같은 체크포인트의 모델·optimizer·난수 상태를 복원해 고정된 에피소드 순서와 학습 예산을 이어갔습니다. 구간별 실행 시간은 전체 학습 시간으로 표시하지 않았습니다.</p>
    <p>재현 명령은 별도 이름으로 새 학습을 시작합니다. 8개 worker를 사용하고 6개 PPO update가 완전히 저장될 때마다 외부 supervisor가 프로세스를 재시작합니다. 같은 동결 소스·의존성 환경에서 사용합니다.</p><pre style="overflow:auto;font-size:11px;background:#f8fafc;padding:14px;border-radius:7px">{escape(reproduction)}</pre></details>
    <div class="links"><a href="summary.json">전체 통계 JSON</a><a href="summary.csv">표 CSV</a><a href="audit.json">검증 기록</a><a href="protocol.json">실험 정의</a></div>
    </main></html>'''
    (out/'index.html').write_text(body)


def report(run='feature_safe_20261020_v1', *, allow_unaudited=False):
    out = ROOT/'runs'/run
    summary = build_summary(out, require_audit=not allow_unaudited)
    save_json(out/'episode_config.json', summary['episode_config'])
    save_json(out/'summary.json', summary)
    write_csv(out, summary)
    write_figure(out, summary)
    write_html(out, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_20261020_v1')
    parser.add_argument('--allow-unaudited', action='store_true', help='Development fixtures only; never use for final results')
    args = parser.parse_args()
    summary = report(args.run, allow_unaudited=args.allow_unaudited)
    print(json.dumps(dict(run=args.run, audit_passed=summary['audit_passed'], verdict=verdict(summary),
                         outputs=['index.html', 'comparison.png', 'summary.json', 'summary.csv']), ensure_ascii=False))


if __name__ == '__main__':
    main()
