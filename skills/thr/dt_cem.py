"""
GP8 평면 던지기(dt_gp8_env)의 액션 시퀀스를 CEM으로 최적화 — DT 학습 데이터의
'prior' 궤적 생성기 + 이 rig의 도달 능력 상한 측정기.

왜: 논문 절차의 무작위 OU 데이터는 THR rig의 bin 거리(1.20~1.70 m)를 거의
만들지 못한다 (무작위 800회에서 ≥1.2 m가 0.1%, ≥1.7 m는 0%). 오프라인 DT는
데이터에 없는 행동을 만들 수 없으므로, 무작위만으로는 원거리를 배울 수 없다.
논문도 이 문제를 알고 있고 §5.3/Fig 5.4에서 prior가 있는 변형이 없는 것보다
낫다고 보고한다. 여기서 CEM으로 목표거리별 성공 궤적을 만들어 데이터셋의
prior 부분으로 쓴다 (무작위 부분은 논문 기본값 그대로 유지).

CEM 대상: 액션 시퀀스 a[0..T−1] ∈ [−1,1]^3 (모터) + 릴리즈 스텝 k_rel.
목적: |x_land − d_goal| 최소화 (릴리즈 성공 실패는 큰 벌점).
결과 궤적은 env를 그대로 굴려 얻으므로 GP8 위치·속도 한계를 자동 만족한다.

실행:
    ../Thr_Phy/.venv/bin/python dt_cem.py --reach        # 도달 상한 스캔
    ../Thr_Phy/.venv/bin/python dt_cem.py --goal 1.5     # 한 목표 최적화
"""

import argparse

import numpy as np

from .dt_gp8_env import GP8Config, GP8ThrowArm

T_MAX = 10          # 최대 스텝 (10 Hz × 1 s) [paper §5.1]


def rollout(arm, motor_seq, k_rel, gripper_closed_val=1.0, open_val=-1.0):
    """액션 시퀀스를 env에 굴린다 → (x_land, temp_mem, released).
    temp_mem은 Thr_DT의 HER 포맷 (state4, action4, reward, done, success)."""
    arm.reset()
    temp_mem, done, k = [], False, 0
    x_land, released = None, False
    while not done and k < len(motor_seq):
        state = arm.get_state()
        a = np.zeros(4)
        a[:3] = np.clip(motor_seq[k], -1.0, 1.0)
        a[3] = open_val if k >= k_rel else gripper_closed_val
        reward, done, term, obj_pos, success = arm.step(a)
        # 무작위 수집과 같은 규약: 라벨은 **실현된 액션** (env가 속도·가속도·
        # 위치 한계 교집합으로 투영한 뒤의 값). 2026-08-05.
        temp_mem.append((state, arm.action_exec.copy(), reward, done, success))
        if success:
            x_land, released = float(obj_pos[0]), True
        k += 1
    return x_land, temp_mem, released


# 카티시안 작업영역(TCP x > 0.20, z > 0.02) 위반에 대한 벌점 계수.
# 왜 필요한가: 수집기는 위반 궤적을 버리는데(사용자 지시), CEM이 제약을 모른 채
# 풀면 그 해의 섭동이 통째로 버려진다 — 실측 prior 롤아웃 496개가 이렇게
# 폐기되어 원거리(≥1.2 m) 커버리지가 74%→62%로 떨어졌다. 벌점을 목적함수에
# 넣어 **애초에 작업영역 안에서 데모 궤적을 뽑는다** (2026-08-05).
# 단위 맞춤: 비용은 착탄 오차[m]이므로, 1 cm 위반이 10 cm 오차와 같은 무게.
W_WORKSPACE = 10.0


def workspace_violation(mem):
    """궤적 상태들의 작업영역 위반 깊이 [m] (0이면 전 구간 만족)."""
    from .dt_gp8_env import TCP_X_MIN, TCP_Z_MIN, tcp_xz
    v = 0.0
    for m in mem:
        x, z = tcp_xz(np.asarray(m[0], float)[:3])
        v = max(v, TCP_X_MIN - x, TCP_Z_MIN - z)
    return max(v, 0.0)


def _land_dist(p, v, z_obj, g=9.81):
    """무항력 포물선의 z=z_obj 도달 수평거리 (평면 x-z). 도달 불가면 None."""
    z0 = p[1] - z_obj
    if z0 <= 0.0:
        return None
    disc = v[1] ** 2 + 2.0 * g * z0
    if disc < 0:
        return None
    t = (v[1] + np.sqrt(disc)) / g
    return p[0] + v[0] * t


def traj_from_seq(arm, seq, k_rel, mem=None, followthrough=True):
    """액션 시퀀스 롤아웃 결과 → 10 Hz 스텝 관절 궤적 + release 시각.

    규약: k_rel은 '그리퍼를 여는 액션의 0-based 스텝 인덱스'이므로 release가
    실제로 일어나는 시각은 그 스텝을 **실행한 뒤**, 즉 t_rel = (k_rel+1)·Δt다.

    followthrough: release 이후 구간을 덧붙인다. 이게 없으면 궤적이 release에서
    끝나 지터의 '+' 방향(늦은 release)이 궤적 범위 밖으로 클립되어 무효가 된다
    — NLP은 감속 구간, Physics-only는 팔로우스루+감속을 갖고 있으므로 DT만
    한쪽 지터를 면제받는 셈이어서 공정하지 않다. 규칙은 dt_gp8_env가 갖고 있고
    (env의 release 계산과 동일해야 하므로) 여기서는 그대로 호출한다.
    리턴: (Qp (n,3), t_nodes (n,), t_rel)."""
    from .dt_gp8_env import followthrough_nodes
    if mem is None:
        _, mem, _ = rollout(arm, seq, k_rel)
    Qp = [m[0][:3] for m in mem] + [arm.q.copy()]
    dt = arm.cfg.dt
    t_rel = (k_rel + 1) * dt
    if followthrough:
        Qp += followthrough_nodes(Qp[-1], arm.qdot_seg, dt)
    Qp = np.asarray(Qp, float)
    return Qp, np.arange(len(Qp)) * dt, t_rel


def jitter_cost(arm, seq, k_rel, d_goal, jitter=0.05, n_probe=5, T=T_MAX):
    """release 시각을 **연속시간**으로 ±jitter[s] 흔들었을 때의 착탄 오차
    (평균 + 최악). 시뮬의 지터 장치(sim_env.RELEASE_JITTER)와 정확히 같은 조건:
    10 Hz 스텝 궤적을 시뮬 그리드로 보간한 뒤 t_rel±jitter의 여러 시점에서
    GP8 FK/Jacobian으로 release 상태를 계산해 탄도를 본다.

    스텝 단위(±1스텝=±0.1 s) 강건성보다 사용자 스펙(±0.05 s)에 맞다 —
    ±1스텝을 요구하면 도달 범위가 1.2 m대로 좁아지는 것을 확인했다."""
    from .dt_gp8_env import PLANAR_IDX, SIM_DT, q6
    from throwing import fk_pos, jacobian

    x_land, mem, released = rollout(arm, seq, k_rel)
    if not released or x_land is None:
        return 1e3, None
    Qp, t_nodes, t_nom = traj_from_seq(arm, seq, k_rel, mem=mem)

    # 시뮬이 실제로 재생하는 궤적으로 평가한다: 스텝 노드를 240 Hz로 편 뒤,
    # env가 보간 모드면 컨트롤러 보간(MotoROS Hermite)까지 거친다.
    if getattr(arm, "interp", None):
        ts_e = np.arange(0.0, t_nodes[-1] + SIM_DT / 2, SIM_DT)
        Q_e = np.column_stack([np.interp(ts_e, t_nodes, Qp[:, j])
                               for j in range(3)])
        Qd_e = np.gradient(Q_e, ts_e, axis=0)
        from sim.gp8_interp import controller_track
        ts_e, Q_e, Qd_e = controller_track(
            ts_e, Q_e, Qd_e, traj_hz=arm.interp.get("traj_hz"),
            period=arm.interp.get("period"), out_dt=SIM_DT)
    else:
        # env(release_state)·planner(gp8_dt_traj_fn)와 **같은** 공식 보간 곡선.
        # 종전에는 여기만 선형보간이라 CEM이 다른 물리를 최적화했다 — 목표
        # 1.35 m 해가 실제로는 1.72 m에 떨어졌다 (2026-08-05 발견).
        from .dt_gp8_env import controller_curve
        ts_e, Q_e, Qd_e = controller_curve(Qp, float(t_nodes[1] - t_nodes[0]))

    errs = []
    for dtj in np.linspace(-jitter, jitter, n_probe):
        t = float(np.clip(t_nom + dtj, ts_e[0], ts_e[-1]))
        q3 = np.array([np.interp(t, ts_e, Q_e[:, j]) for j in range(3)])
        qd3 = np.array([np.interp(t, ts_e, Qd_e[:, j]) for j in range(3)])
        p3 = fk_pos(q6(q3))
        Jv, _ = jacobian(q6(q3))
        qd6 = np.zeros(6)
        qd6[PLANAR_IDX] = qd3
        v3 = Jv @ qd6
        # 착지 기준면은 env(_ballistic_landing)와 **같아야** 한다: 물체 반대각
        # 위에 조준면 z_land(=−0.08, pit 바닥이 아니라 bin 조준점)를 더한 높이다.
        # 종전에는 z_land를 빼먹어 CEM만 z=0 기준으로 풀었고, 그만큼 더 낙하하는
        # 동안 수평으로 더 나가 원거리에서 13~41 cm 계통 오차가 났다 (2026-08-05).
        z_ref = getattr(arm.cfg, "z_land", 0.0)
        d = _land_dist(np.array([p3[0], p3[2]]), np.array([v3[0], v3[2]]),
                       z_ref, arm.cfg.gravity)
        errs.append(1e3 if d is None else abs(d - d_goal))
    pen = W_WORKSPACE * workspace_violation(mem)
    return float(np.mean(errs) + 0.5 * np.max(errs) + pen), x_land


def seq_cost(arm, seq, k_rel, d_goal, T=T_MAX, robust=True):
    """액션 시퀀스의 비용 = 착탄 오차. robust=True면 릴리즈 스텝 ±1(=±0.1 s,
    10 Hz)에 대한 평균 + 최악 오차를 함께 벌점한다.

    왜 robust가 기본인가: 오차만 최소화하면 CEM은 knife-edge 해를 찾는다
    (실측: 목표 2.00 m를 0.0 cm로 맞춘 해가 액션을 σ=0.12로 흔들면 착지가
    −0.91~1.08 m로 붕괴). 그런 궤적을 prior로 넣으면 ① 섭동 데이터가 목표
    주변에 모이지 않아 학습이 안 되고 ② 배운 정책이 release 지터에 무력하다.
    NLP 플래너가 W_ACC로 release 윈도우 '전 구간'의 착탄 정확도를 강제하는 것과
    같은 취지 — 세 모델이 각자의 방식으로 타이밍 강건성을 추구하게 둔다."""
    ks = [k_rel - 1, k_rel, k_rel + 1] if robust else [k_rel]
    errs, pen = [], 0.0
    for k in ks:
        if k < 1 or k >= T:
            continue
        x_land, mem, released = rollout(arm, seq, k)
        errs.append(1e3 if (not released or x_land is None)
                    else abs(x_land - d_goal))
        pen = max(pen, W_WORKSPACE * workspace_violation(mem))
    return float(np.mean(errs) + 0.5 * np.max(errs) + pen)


def cem_throw(d_goal, cfg=None, n_iter=14, pop=160, elite=16, T=T_MAX,
              rng=None, sigma0=0.7, verbose=False, robust="jitter",
              jitter=0.05, arm=None):
    """목표거리 d_goal[m]에 대한 액션 시퀀스 CEM 최적화.
    robust: "jitter"(기본, 연속시간 ±jitter 강건 — 시뮬 조건과 동일) |
            True(릴리즈 스텝 ±1 = ±0.1 s 강건, 더 보수적) | False(공칭만)
    arm: 기존 env 재사용 (보간 설정을 물려받기 위해 — 넘기지 않으면 새로 만든다)
    리턴: dict(cost, x_land, motor_seq, k_rel, temp_mem)."""
    rng = np.random.default_rng(0) if rng is None else rng
    cfg = GP8Config() if cfg is None else cfg
    arm = GP8ThrowArm(cfg, rng=rng) if arm is None else arm

    def cost_of(seq, k_rel):
        if robust == "jitter":
            return jitter_cost(arm, seq, k_rel, d_goal, jitter=jitter, T=T)[0]
        return seq_cost(arm, seq, k_rel, d_goal, T, bool(robust))

    best = dict(cost=np.inf, x_land=None, motor_seq=None, k_rel=None,
                temp_mem=None)
    by_k = {}
    # 릴리즈 스텝별로 별도 분포를 유지 (k_rel은 이산 — 스텝마다 CEM을 돌리고
    # 최적 k_rel을 고른다. T가 10이라 전수 탐색이 저렴하다.)
    for k_rel in range(3, T):
        mu = np.zeros((T, 3))
        sd = np.full((T, 3), sigma0)
        k_best = dict(cost=np.inf, x_land=None, motor_seq=None, k_rel=k_rel,
                      temp_mem=None)
        for it in range(n_iter):
            cand = np.clip(mu[None] + sd[None] * rng.standard_normal((pop, T, 3)),
                           -1.0, 1.0)
            costs = np.array([cost_of(seq, k_rel) for seq in cand])
            idx = np.argsort(costs)[:elite]
            mu, sd = cand[idx].mean(0), cand[idx].std(0) + 1e-3
            if costs[idx[0]] < k_best["cost"]:
                seq = cand[idx[0]]
                x_land, mem, released = rollout(arm, seq, k_rel)
                k_best = dict(cost=float(costs[idx[0]]), x_land=x_land,
                              motor_seq=seq.copy(), k_rel=k_rel, temp_mem=mem)
        if k_best["motor_seq"] is not None:
            # 같은 목표를 **다른 릴리즈 타이밍**으로 맞춘 해 — 스윙 모양이
            # 실제로 다르다. prior의 행동 다양성은 여기서 나온다.
            by_k[k_rel] = k_best
            if k_best["cost"] < best["cost"]:
                best = k_best
        if verbose:
            print(f"    k_rel={k_rel}: cost {k_best['cost']:.4f} "
                  f"(x_land={k_best['x_land']}), 전체 best {best['cost']:.4f}")
    best = dict(best)
    best["by_k"] = by_k
    return best


def reach_scan(goals=None, rng=None, verbose=True):
    """목표거리별 CEM 최적 오차 — 이 rig의 도달 능력 프로파일."""
    goals = np.arange(0.8, 2.21, 0.2) if goals is None else np.asarray(goals)
    rng = np.random.default_rng(1) if rng is None else rng
    out = []
    for d in goals:
        b = cem_throw(float(d), rng=rng)
        out.append((float(d), b["x_land"], b["cost"], b["k_rel"]))
        if verbose:
            print(f"  목표 {d:.2f}m → 최적 착지 {b['x_land']:.3f}m "
                  f"(오차 {100 * b['cost']:.1f}cm, release step {b['k_rel']})",
                  flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", action="store_true", help="도달 상한 스캔")
    ap.add_argument("--goal", type=float, default=None)
    args = ap.parse_args()
    if args.reach:
        rows = reach_scan()
        ok = [d for d, x, c, k in rows if c < 0.05]
        print(f"\n오차 5cm 이내 달성 목표: {ok}")
        print(f"→ 이 rig(3축 평면, 10Hz, 1s)의 실용 도달 범위 상한 ≈ "
              f"{max(ok) if ok else float('nan')} m")
    else:
        d = args.goal if args.goal is not None else 1.5
        b = cem_throw(d, verbose=True)
        print(f"목표 {d}m → 착지 {b['x_land']:.3f}m, release step {b['k_rel']}, "
              f"오차 {100 * b['cost']:.1f}cm")
