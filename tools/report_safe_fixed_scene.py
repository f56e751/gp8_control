"""A short, local report for fixed-reset native single-skill evaluations."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ('target_success_percent', '목표 쓰레기통 성공률', '%'),
    ('contacts_per100', '비의도 접촉 / 100회', '건'),
    ('impulse_ns_per_attempt', '시도당 접촉 충격량', 'N·s'),
    ('falls_per100', '주변 물체 추락 / 100회', '개'),
    ('collision_free_success_percent', '접촉 없이 성공', '%'),
)


def difference(group, metric):
    row = group.get('paired_differences', {}).get(metric, {})
    mean = row.get('mean', row.get('difference', row.get('difference_mean')))
    interval = row.get('ci95', row.get('difference_ci95'))
    return mean, interval


def num(value):
    return '—' if value is None else f'{float(value):.2f}'


def group_name(group):
    count = group['neighbor_count']
    return f"{group['target_skill'].capitalize()} · " + (
        '이웃 2·5개 합산' if count == 'all' else f'이웃 {count}개')


def build(out):
    summary = json.loads((out/'comparison_summary.json').read_text())
    audit = json.loads((out/'audit.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    protocol_sha = hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()
    if not (audit.get('passed') and audit.get('evaluation_complete')
            and audit.get('protocol_sha256') == protocol_sha
            and summary.get('complete') and summary.get('protocol_sha256') == protocol_sha):
        raise RuntimeError('A completed, passed fixed-scene audit is required')
    review = None
    if (out/'independent_review.json').is_file():
        review = json.loads((out/'independent_review.json').read_text())
        if not review.get('passed') or review.get('protocol_sha256') != protocol_sha:
            raise RuntimeError('The independent review belongs to a different protocol')
    groups = summary['groups']
    if isinstance(groups, dict):
        groups = list(groups.values())
    main = [g for g in groups if g['neighbor_count'] == 'all']
    if len(main) != 2:
        main = groups

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gp8_safe_fixed_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font = FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), facecolor='white')
    x = np.arange(len(main))
    for ax, (metric, label, unit) in zip(axes, METRICS[:3]):
        a = [g['baseline'][metric] for g in main]
        b = [g['learned'][metric] for g in main]
        bars = (ax.bar(x-.19, a, .36, color='#65758b', label='기존 정책'),
                ax.bar(x+.19, b, .36, color='#16847d', label='최신 학습 정책 · 2모델 평균'))
        for collection in bars:
            ax.bar_label(collection, fmt='%.2f', padding=4, fontsize=9)
        ax.set_xticks(x, [g['target_skill'].capitalize() for g in main])
        ax.set_title(label, fontsize=12, pad=12)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(max(a+b)*1.22, 1.))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('현재 시뮬레이터 · 고정 초기 배치에서 한 번의 작업 비교', fontsize=15)
    fig.text(.5, .02, '벨트 0.12 m/s · 추가 물체 생성 없음 · 같은 초기 상태와 관찰 시간 · 재학습 없이 최종 모델 평가',
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)

    rows = []
    detailed_rows = []
    for group in groups:
        for metric, label, unit in METRICS:
            delta, interval = difference(group, metric)
            ci = '—' if interval is None else f'[{num(interval[0])}, {num(interval[1])}]'
            row = (f'<tr><td>{html.escape(group_name(group))}</td><td>{label}</td>'
                   f'<td>{num(group["baseline"][metric])}</td><td>{num(group["learned"][metric])}</td>'
                   f'<td>{num(delta)} {ci}</td></tr>')
            (rows if group in main else detailed_rows).append(row)

    statements = []
    for group in main:
        delta, ci = difference(group, 'contacts_per100')
        skill_label = group['target_skill'].capitalize()
        if group['baseline']['contacts_per100'] == 0 and group['learned']['contacts_per100'] == 0:
            result = '두 정책 모두 접촉이 없어 추가적인 접촉 감소 효과를 평가하기 어렵습니다'
        elif ci is not None and ci[1] < 0:
            result = '접촉 감소가 확인됐습니다'
        elif ci is not None and ci[0] > 0:
            result = '접촉이 증가했습니다'
        else:
            result = '접촉 감소가 명확하게 확인되지 않았습니다'
        success_delta, success_ci = difference(group, 'target_success_percent')
        if success_ci is not None and success_ci[1] < 0:
            result += '. 성공률도 감소했습니다'
        increased = []
        for metric, risk_label in (('impulse_ns_per_attempt', '충격량'), ('falls_per100', '주변 물체 추락')):
            _, interval = difference(group, metric)
            if interval is not None and interval[0] > 0:
                increased.append(risk_label)
        if increased:
            result += '. ' + '·'.join(increased) + '도 증가했습니다'
        statements.append(f'{skill_label}: {result}.')
    verdict = ' '.join(statements)
    source_run = protocol.get('model_run', protocol.get('source_run', 'feature_safe_route_20261021_v2'))
    table_header = '<thead><tr><th>조건</th><th>지표</th><th>기존</th><th>학습</th><th>학습−기존 [95% 구간]</th></tr></thead>'
    compact_rows = []
    for group in main:
        cells = ''.join(f'<td>{num(group["baseline"][key])} → {num(group["learned"][key])}</td>'
                        for key, _, _ in METRICS[:4])
        compact_rows.append(f'<tr><td>{group["target_skill"].capitalize()} ({group["n_scenes"]}배치)</td>{cells}</tr>')
    compact_table = ('<table><thead><tr><th>작업</th><th>성공률 (%)</th><th>접촉 / 100회</th>'
                     '<th>충격량 (N·s/회)</th><th>주변 추락 / 100회</th></tr></thead><tbody>'
                     + ''.join(compact_rows) + '</tbody></table>')
    per_model_rows = []
    for group in main:
        for arm in ('safe_seed0', 'safe_seed1'):
            values = group['per_policy'][arm]
            cells = ''.join(f'<td>{num(values[key])}</td>' for key, _, _ in METRICS[:4])
            per_model_rows.append(f'<tr><td>{group["target_skill"].capitalize()} · {arm}</td>{cells}</tr>')
    per_model_table = ('<table><thead><tr><th>모델</th><th>성공률 (%)</th><th>접촉 / 100회</th>'
                       '<th>충격량 (N·s/회)</th><th>주변 추락 / 100회</th></tr></thead><tbody>'
                       + ''.join(per_model_rows) + '</tbody></table>')
    push = next((g for g in main if g['target_skill'] == 'push'), None)
    preparation = ''
    if push is not None:
        delta, interval = difference(push, 'push_prepare_contacts_per100')
        preparation = (f'<p>Push 접근·준비 단계에 시작한 접촉: 100회당 '
                       f'{num(push["baseline"]["push_prepare_contacts_per100"])} → '
                       f'{num(push["learned"]["push_prepare_contacts_per100"])}건. '
                       f'차이 {num(delta)}, 95% 구간 [{num(interval[0])}, {num(interval[1])}]. '
                       '실제 미는 동작·복귀까지 포함한 전체 접촉은 위 표와 구분합니다.</p>')
    control_note = ''
    control_details = ''
    contact_note = ''
    if review is not None:
        control = [r for r in review['control_diagnostics'] if r['policy'].startswith('safe_seed')]
        approach = [r for r in control if r['role'] == 'push_approach']
        decisions = sum(r['decisions'] for r in approach)
        rejected = sum(r['rejected'] for r in approach)
        applied = sum(r['nonzero'] for r in approach)
        if decisions and rejected == decisions and applied == 0:
            control_note = (f'<p class="answer"><b>Push 접근 보정은 {rejected}/{decisions}회 모두 차단됐습니다.</b> '
                            '신경망이 보정값을 냈지만, 가속도 제한 검사에서 모든 후보가 거절돼 '
                            '기존 접근 경로가 실행됐습니다. '
                            'Throw 경로와 Push 복귀 경로의 보정은 실제 적용됐습니다.</p>')
        role_names = dict(throw_loaded='Throw 잡은 뒤 이동', throw_return='Throw 복귀',
                          push_approach='Push 접근', push_return='Push 복귀')
        control_rows = ''.join(f'<tr><td>{role_names[r["role"]]} · {r["policy"]}</td>'
                               f'<td>{r["decisions"]}</td><td>{r["nonzero"]}</td><td>{r["rejected"]}</td></tr>'
                               for r in control)
        control_details = ('<table><thead><tr><th>구간 · 모델</th><th>결정 수</th><th>실제 보정 적용</th>'
                           '<th>보정 완전 거절</th></tr></thead><tbody>'+control_rows+'</tbody></table>')
        if push is not None:
            kinds = review['push_contact_semantics']
            def other_count(arm):
                counts = kinds[arm]['raw_simulator_kind_counts']
                return counts.get('object_other', 0)+counts.get('robot_other', 0)
            before = 100.*other_count('baseline')/push['n_scenes']
            after = 50.*(other_count('safe_seed0')+other_count('safe_seed1'))/push['n_scenes']
            clarification = review.get('posthoc_contact_type_clarification', {}).get('metrics', {})
            other = clarification.get('true_other_trash_contacts_per100')
            contact_ci = ('' if other is None else
                          f' 차이의 95% 구간 [{num(other["ci95"][0])}, {num(other["ci95"][1])}]도 '
                          '0을 포함해 감소가 명확하지 않습니다. ')
            target = clarification.get('robot_target_contacts_per100')
            target_note = ('' if target is None else
                           f' 작업 대상 자신과의 비의도 접촉은 {num(target["baseline"])} → '
                           f'{num(target["learned"])}건/100회로 늘었습니다.')
            falls = review['push_fall_diagnostics']['baseline']['count']
            contact_note = (f'<p>사용자님이 피하려는 <b>다른 쓰레기와의 접촉</b>만 후속 분해하면, '
                            f'Push는 100회당 {num(before)} → {num(after)}건으로 거의 같았습니다. '
                            +contact_ci+
                            '위 전체 접촉에는 작업 대상 물체와의 비의도 접촉도 포함됩니다. '
                            +target_note+
                            f' Push 중 이웃 물체 추락은 모든 정책에서 {push["n_scenes"]}회 중 {falls}건이었습니다. '
                            '접촉 유형의 후속 분해는 사전 지정한 전체 접촉 지표와 구분합니다.</p>')
    old = ''
    evidence_path = out/'old_evidence.json'
    if evidence_path.is_file():
        old = '<p><a href="old_evidence.json">이전 실험의 조건과 수치 근거</a></p>'
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>고정 초기 배치 비교</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#192b40;background:#f4f7fb;margin:0}main{max-width:1100px;margin:32px auto;padding:28px;background:white;border-radius:14px}h1{font-size:27px;margin:0 0 12px}h2{font-size:20px;margin-top:32px}.answer{background:#eaf3f4;border-left:5px solid #16847d;padding:16px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dde5ed;padding:10px}th{background:#f0f4f8}details{margin-top:24px}summary{cursor:pointer;font-weight:650}code,pre{background:#f0f4f8}pre{padding:14px;overflow:auto}a{color:#086f84}.muted{color:#536578;font-size:14px}</style><main>
<h1>고정 초기 배치에서도 안전성 개선이 유지되는가?</h1>
<p class="answer">VERDICT</p>
CONTROLNOTE
<p>현재 <code>feature/simulator</code>에서 로봇을 동일한 초기 자세로 시작하고, 목표 물체 1개와 이웃 2개 또는 5개를 한 번 배치했습니다. 각 배치에 대해 기존 정책, 경로 보정 0인 정책, 최종 학습 모델 2개를 비교했습니다. 벨트는 계속 움직이고 새 물체는 생성되지 않습니다. 카메라 감지 후 목표 물체가 공통 작업 시작 위치에 도착할 때까지 기다리고, 기존 선택기로 한 개의 작업을 고릅니다. 집기·던지기 또는 접근·밀기부터 15초 동안 물리를 관찰하며, 먼저 쓰레기통에 도착하더라도 관찰을 계속합니다.</p>
<p><b>이번에는 재학습하지 않았습니다.</b> SOURCE 모델의 고정 초기 배치 적용 결과입니다. 배치는 여러 개이며, 각 배치의 시작 상태를 네 정책에서 동일하게 리셋했습니다.</p>
<p class="muted">총 SCENECOUNT개 배치, ROLLOUTCOUNT회 실행. 장면 단위로 비교하며, 같은 장면의 두 학습 모델 실행을 독립 장면으로 세지 않았습니다.</p>
COMPACT
<p class="muted">기존 정책 → 최신 학습 정책 2모델 평균. 성공률은 높을수록, 나머지 지표는 낮을수록 좋습니다.</p>
CONTACTNOTE
<img src="comparison.png" alt="기존 정책과 최신 학습 정책의 성공률, 접촉, 충격량 비교">
<p class="muted">학습 수치는 두 학습 모델의 평균입니다. 95% 구간은 같은 장면의 차이를 묶어 계산한 bootstrap 구간이며, 두 초기화 모델에 조건부인 결과입니다. 접촉 수가 작아져도 성공률·충격량·주변 추락을 함께 봐야 합니다.</p>
<details><summary>통계 구간·이웃 개수별 결과</summary><table>HEADER<tbody>ROWSDETAILS</tbody></table></details>
<details><summary>실제 경로 보정·Push 접근 단계·모델별 결과</summary>CONTROLDETAILSPREPARATIONPERMODEL</details>
<details><summary>예전 결과와 이번 조건의 차이</summary>
<p>이전 throw 실험에서는 접촉이 100회당 33.98건에서 3.71건으로 약 89% 줄었고, 성공률은 100%였습니다. 그 실험은 이미 성공적으로 잡힌 물체에서 시작하고 복귀·대기 최종 자세까지 학습했습니다. 이번에는 현재 시뮬레이터의 감지·집기부터 포함하며, 최신 정책은 경로의 시작·끝을 유지합니다. 시뮬레이터의 벨트·접촉 설정도 바뀌었습니다. 현재 기본 Throw의 마지막 명령 TCP 높이는 약 10cm이고, 이전 기본 대기 높이는 약 5cm였습니다. 높이 차이는 코드·저장 명령으로 확인했지만, 접촉 감소의 원인으로 독립 입증한 것은 아닙니다. 따라서 이번 결과만으로 차이의 원인을 ‘연속 작업 여부’ 하나로 단정할 수는 없습니다.</p>
<p>과거 push의 성공은 6cm 이상 밀기였지만 이번 기본 성공은 올바른 쓰레기통 도착입니다. 서로 다른 성공률을 직접 비교하지 않습니다.</p>OLD</details>
<details><summary>이 평가로 확인할 수 있는 범위</summary><p>고정 초기 배치에서 최신 연속 작업 학습 모델을 적용했을 때의 성능을 확인했습니다. 고정 배치에 맞춰 새로 학습했을 때의 성능은 평가하지 않았습니다. 목표 물체는 가장 하류에 있고 이웃은 양옆·상류에 배치하며, 박스의 yaw는 0입니다. 목표 x 범위는 Throw 0.36–0.44m, Push 0.47–0.49m이며, 모든 물체는 벨트 위에 완전히 놓입니다. 결과는 이 배치 범위에 대한 것입니다.</p><p>입력 차원은 같은 107개이지만 추가 공급 비율은 항상 0이며, 남은 물체 비율의 분모는 학습 때의 20개에서 이번의 3개 또는 6개로 바뀝니다. 후속 작업으로 이어지는 체인도 없습니다. 이런 입력·행동 분포 차이가 있으므로, 성능 차이만으로 특정 원인 하나를 입증할 수는 없습니다.</p></details>
<details><summary>측정과 재현 자료</summary><p>MuJoCo 물리 단계의 실제 접촉력 0.1N 초과를 집계합니다. 같은 형상 쌍에서 30ms 넘는 공백 뒤의 접촉은 새 사건입니다. 정상적인 흡착·밀기 접촉은 제외하고, 다른 물체 및 환경과의 비의도 접촉을 집계했습니다. 단계별 수치는 접촉이 시작된 단계 기준이며, 단계 경계를 가로지르는 사건 전체 충격량을 정확히 분할한 값은 아닙니다.</p>
<p><a href="protocol.json">실험 조건</a> · <a href="comparison_summary.json">모든 통계와 모델별 수치</a> · <a href="audit.json">초기 상태·0보정 일치·원본 보존 검증</a> · <a href="independent_review.json">독립 재검산·접촉 종류·보정 거절 진단</a> · <a href="verification/root_checks.json">단위 테스트·재생 확인</a></p></details>
<details><summary>시뮬레이터에서 같은 장면 직접 보기</summary><p>프로젝트 상위 폴더에서 아래 명령을 실행합니다. <code>--policy baseline</code>과 <code>--policy safe0</code>을 바꾸면 같은 배치를 비교할 수 있습니다. 창에서 Space로 일시 정지·재개합니다. 처음에는 공통 벨트 이동·감지 구간이 있고, 이후 한 번의 작업을 실행합니다.</p>
<pre>cd /home/kjy_rllab/RLLAB_research/Goal-Oriented
gp8_control/.venv/bin/python -m gp8_control.tools.replay_safe_fixed_scene --policy safe0 --skill throw --neighbors 5 --scene-seed 5200000 --paused
gp8_control/.venv/bin/python -m gp8_control.tools.replay_safe_fixed_scene --policy safe0 --skill push --neighbors 5 --scene-seed 5400000 --paused</pre></details>
<p class="muted">로컬 파일 보고서입니다. HTTP 서버를 실행하지 않았습니다.</p></main></html>'''
    replacements = {'VERDICT': html.escape(verdict), 'SOURCE': html.escape(str(source_run)),
                    'HEADER': table_header, 'ROWS': ''.join(rows), 'DETAILS': ''.join(detailed_rows), 'OLD': old,
                    'SCENECOUNT': str(summary['scene_count']), 'ROLLOUTCOUNT': str(summary['result_count']),
                    'COMPACT': compact_table, 'PREPARATION': preparation, 'PERMODEL': per_model_table,
                    'CONTROLNOTE': control_note, 'CONTROLDETAILS': control_details, 'CONTACTNOTE': contact_note}
    for key in sorted(replacements, key=len, reverse=True):
        document = document.replace(key, replacements[key])
    (out/'index.html').write_text(document)
    return dict(report=str(out/'index.html'), chart=str(out/'comparison.png'), verdict=verdict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_fixed_20261021_v1')
    args = parser.parse_args()
    print(json.dumps(build(ROOT/'runs'/args.run), ensure_ascii=False))


if __name__ == '__main__':
    main()
