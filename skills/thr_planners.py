"""THR 던지기 계획 모델 3종 (nlp / dt / phy) 의 **로봇용 traj_fn 어댑터**.

THR 시뮬(`THR/sim/sim_env.py`)이 정의한 traj_fn 인터페이스를 그대로 쓴다:

    plan = traj_fn(target, p_start, v_start, ctx) -> dict(
        ts (N,), Q (N,6) planner 프레임, Qd (N,6),
        t_rel [s] 또는 release_window (t0, t1),
        q_of/qd_of (선택, 스플라인 정밀도 release 상태),
        q_start (6,), info dict)

같은 인터페이스라 `thr_throw_skill.py` 하나가 셋 다 로봇에 태울 수 있다.
계획 코드 자체는 `skills/thr/` 에 벤더링된 THR 원본을 그대로 호출한다.

──────────────────────────────────────────────────────────────────────────────
로봇 이식에서 의도적으로 달라지는 것 (전부 아래 3가지뿐)
──────────────────────────────────────────────────────────────────────────────
1. **NLP multistart 가 순차**다. THR `nlp_planner._solve_round` 는 fork Pool 로
   후보를 병렬 solve 하지만, ROS2 노드에서 fork 는 DDS 스레드와 충돌해 hang
   위험이 있다 (기존 `robust_throw_skill.py` 의 INIT_VARIANTS_ROS 주석과 같은
   판단). 해의 정의는 동일하고 벽시계만 느려진다.
2. **pybullet dry-run 충돌 검사가 없다.** THR 은 시뮬 안에서 궤적을 재생해
   기둥/바닥/물체 클리어런스를 검사하지만(`ctx["path_check"]`), 로봇에는 그
   시뮬이 없다. 대신 `thr_throw_skill` 이 dispatch 직전에 로봇 FK 기반 Cartesian
   안전 엔벨로프 게이트를 건다 — 그쪽이 실기 기준으로는 더 직접적이다.
3. **PC 기반 항력 추정 없음.** THR 도 `PLAN_WITH_DRAG = False` 라 기본 경로에서
   쓰지 않는다 (무항력 포물선으로 계획). 동일.

그 외 공식화 파라미터(RELEASE_TIME, W1/W2/W_ACC, QDD_LIM, 관절 한계, GP8_DIMS,
CART_CONSTRAINTS, DT 가중치/한계, phy 의 c_h/c_d/램프 거리)는 벤더링된 THR
모듈의 값을 그대로 쓴다 — 즉 **2026-08-03 업데이트가 그대로 반영된다**.
"""

from __future__ import annotations

import math
import os
import threading

import numpy as np

from gp8_control.skills.thr import dt_cem, throw_nlp
from gp8_control.skills.thr.throwing import (GP8_DIMS, GP8_Q_MAX, GP8_Q_MIN,
                                             GP8_QD_MAX, ik_position, jacobian)

# THR 시뮬 궤적 샘플 주기 (sim_env.DT). 계획 궤적을 이 격자로 뽑는다 — 로봇
# 스트리머가 4 ms 로 다시 리샘플하므로 조밀할수록 손실이 없다.
DT: float = 1.0 / 240.0

# 던지기 시작 TCP 높이 [m] (THR sim_env.P_START_Z). 물체를 집은 뒤 이 높이로
# 들어올린 자세에서 스윙이 시작된다는 THR 규약을 로봇에서도 유지한다.
P_START_Z: float = float(os.environ.get("GP8_THR_P_START_Z", "0.20"))

# NLP 다시작(multistart) 예산 — 순차 실행이라 THR(fork 병렬)보다 보수적으로 잡는다.
NLP_MAX_RANDOM: int = int(os.environ.get("GP8_THR_NLP_MAX_RANDOM", "32"))

# THR `nlp_planner.INIT_VARIANTS` 와 동일한 초기해 집합 (순서까지 동일).
INIT_VARIANTS = (
    dict(dq_swing=[0.0, 0.6, 0.9, 0.0, 0.7, 0.0], T0=0.5, chi0=0.4),
    dict(dq_swing=[0.0, 1.1, 0.9, 0.0, 0.7, 0.0], T0=0.5, chi0=0.4),
    dict(dq_swing=[0.0, 0.6, 0.9, 0.0, 0.3, 0.0], T0=0.5, chi0=0.4),
    None,
    dict(dq_swing=[-0.13, 1.1, 0.9, 0.0, 0.5, 0.0], T0=1.0, chi0=0.6),
    dict(dq_swing=[0.0, 1.1, 0.4, 0.0, 0.3, 0.0], T0=1.1, chi0=0.6),
)


def _log(logger, msg):
    (logger.info if logger is not None else print)(msg)


# ===========================================================================
# warm start DB (NLP 전용) — THR nlp_planner 와 동일한 v2 축적 포맷
# ===========================================================================
# offline 에 (grasp 12 × target 12) 최적해를 저장해 두고 런타임엔 최근접 entry 로
# full warm start polish → cold multistart 수십 초 → 1~2 초.
#
# 파일 포맷은 THR `nlp_planner._db_read` 와 같은 v2 축적형:
#     {"version": 2, "configs": [{"params": {...}, "entries": [...]}, ...]}
# **공식화 파라미터(params)가 다른 해는 서로 다른 config 로 공존**하므로, W1 이나
# 관절 한계를 바꿔 가며 여러 벌을 같은 파일에 쌓아 둘 수 있고 런타임은 지금
# 설정과 일치하는 config 만 골라 쓴다 (불일치면 조용히 cold 로 간다).
WARM_DB_PATH: str = os.environ.get(
    "GP8_THR_WARM_DB", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "thr", "warm_db_thr.pkl"))

_WARM_DB: "list | None" = None      # 로드 캐시 (None=미로드, []=없음/불일치)


def warm_db_params() -> dict:
    """entry 유효성 키 — THR `nlp_planner._warm_db_params` 와 동일 구성.
    이 조합이 같아야 같은 공식화의 해다 (basin/제약 구성 호환 → lam_g 재사용 가능).
    """
    return dict(
        rt=throw_nlp.RELEASE_TIME,
        w_acc=throw_nlp.W_ACC,
        n_ctrl=throw_nlp.N_CTRL,
        n_win=throw_nlp.N_WIN,
        q_lo=throw_nlp.Q_LO.tolist(),
        q_hi=throw_nlp.Q_HI.tolist(),
        qd_max=GP8_QD_MAX.tolist(),
        col=((throw_nlp.COL_R, throw_nlp.COL_H)
             if throw_nlp.CART_CONSTRAINTS else "off"),
        pos_mode=getattr(throw_nlp, "POS_LIMIT_MODE", "colloc"),
        w1=throw_nlp.W1,
        w2=throw_nlp.W2,
        w_sens=throw_nlp.W_SENS,
        qdd_lim=throw_nlp.QDD_LIM.tolist(),
        t_bounds=tuple(throw_nlp.T_BOUNDS),
        degree=throw_nlp.DEGREE,
        dims=tuple(sorted(GP8_DIMS.items())),
        flight=getattr(throw_nlp, "FLIGHT_MODEL", "parabola"),
    )


def db_read(path: str = None) -> dict:
    """warm DB 파일 → v2 dict. 없거나 손상되면 빈 DB."""
    import pickle
    path = path or WARM_DB_PATH
    if not os.path.exists(path):
        return dict(version=2, configs=[])
    try:
        with open(path, "rb") as f:
            db = pickle.load(f)
    except Exception:                                       # noqa: BLE001
        return dict(version=2, configs=[])
    if "configs" in db:
        return db
    cfgs = []
    if db.get("entries"):                                   # 구 단일 config 포맷
        cfgs.append(dict(params=db.get("params", {}), entries=db["entries"]))
    return dict(version=2, configs=cfgs)


def db_select(db: dict, params: dict):
    """params 키가 전부 일치하는 config (없으면 None). subset 비교라 하위 호환."""
    for cfg in db["configs"]:
        if all(cfg["params"].get(k) == v for k, v in params.items()):
            return cfg
    return None


def db_merge_save(new_entries, path: str = None) -> list:
    """현재 공식화 config 에 entry '축적' (같은 target/p_start 는 최신으로 교체).
    다른 공식화 config 는 그대로 보존한다 — 파라미터를 되돌리면 예전 entry 재사용."""
    import pickle
    path = path or WARM_DB_PATH
    db = db_read(path)
    params = warm_db_params()
    cfg = db_select(db, params)
    if cfg is None:
        cfg = dict(params=params, entries=[])
        db["configs"].append(cfg)

    def _key(e):
        return (tuple(np.round(np.asarray(e["target"], float), 3)),
                tuple(np.round(np.asarray(e["p_start"], float), 3)))

    merged = {_key(e): e for e in cfg["entries"]}
    for e in new_entries:
        merged[_key(e)] = e
    cfg["entries"] = list(merged.values())
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(db, f)
    os.replace(tmp, path)                # 원자적 교체 — 중단돼도 파일이 깨지지 않는다
    return cfg["entries"]


def load_warm_db(logger=None) -> list:
    """지금 공식화와 일치하는 entry 리스트 (없으면 []). 프로세스당 1회 로드."""
    global _WARM_DB
    if _WARM_DB is None:
        db = db_read()
        cfg = db_select(db, warm_db_params())
        _WARM_DB = cfg["entries"] if cfg else []
        if _WARM_DB:
            _log(logger, f"  warm DB: {os.path.basename(WARM_DB_PATH)} "
                         f"entry {len(_WARM_DB)}개 (현재 공식화 일치)")
        elif db["configs"]:
            _log(logger, f"  warm DB: 현재 공식화와 일치하는 config 없음 — cold 진행 "
                         f"(다른 공식화 config {len(db['configs'])}개는 보존됨)")
    return _WARM_DB


def reset_warm_db_cache() -> None:
    """공식화 상수를 바꾼 뒤 다시 조회하려면 호출 (빌더가 쓴다)."""
    global _WARM_DB
    _WARM_DB = None


# ===========================================================================
# 1) NLP — CasADi/IPOPT B-spline (THR nlp_planner.nlp_traj_fn 의 로봇판)
# ===========================================================================

def nlp_traj_fn(target, p_start, v_start=None, ctx=None, logger=None):
    """THR `nlp_planner.plan_nlp_throw` + `nlp_traj_fn` 의 순차 이식.

    release 는 고정 시각이 아니라 **윈도우** [t*−rt/2, t*+rt/2] 로 돌려준다.
    플래너가 W_ACC 로 윈도우 전 구간의 착탄 정확도를 강제하므로 그 안 어디서
    놓아도 맞는다 — 밸브 지터에 대한 내성이 여기서 나온다 (rt = 0.1 s,
    2026-07-31 사용자 스펙으로 0.05 → 0.1 상향).
    """
    p_target = np.asarray(target, float)
    p_start = np.asarray(p_start, float)
    if v_start is not None and not np.allclose(np.asarray(v_start, float), 0.0):
        raise ValueError("NLP 플래너는 정지 출발(v_start=0)만 지원")

    # 시작자세를 먼저 확정 — 한계 안 해가 없으면 어떤 초기해로도 못 푼다
    # (P₀는 상수라 hull 제약이 못 잡고 dense 검증이 전 후보를 기각).
    q_start = throw_nlp.ik_start_pose(p_start)      # ValueError = 이 지점 불가

    fails: list[str] = []
    best = None

    def _try(init=None, tag="", warm=None):
        nonlocal best
        try:
            res = throw_nlp.solve_throw_nlp(p_start, p_target, init=init,
                                            warm_data=warm, q_start=q_start)
        except (RuntimeError, AssertionError, ValueError) as e:
            fails.append(f"{tag}: NLP 실패 — {type(e).__name__}")
            return False
        # 0.06°(1e-3 rad) 이하 잔존 위반은 물리적으로 무의미 → 허용 (THR과 동일)
        if res.get("pos_viol_dense", 0.0) > 1e-3:
            fails.append(f"{tag}: 위치한계 잔존 위반 {res['pos_viol_dense']:.1e}")
            return False
        if best is None or res["J"] < best[0]["J"]:
            best = (res, tag)
        return True

    # ---- ① warm DB polish 먼저 (THR plan_nlp_throw 와 같은 선택 규칙) ----
    #      target 최근접 entry 들 중 시작점이 가까운 순으로 상위 2개에 full warm
    #      start (primal+dual 재시동) → cold 수십 초 대신 1~2 초. entry 2개를 보는
    #      이유는 같은 target 의 다른 basin 을 커버하기 위해서다.
    db = load_warm_db(logger)
    if db:
        d_tgt = min(float(np.linalg.norm(np.asarray(e["target"])[:2] - p_target[:2]))
                    for e in db)
        near = [e for e in db
                if float(np.linalg.norm(np.asarray(e["target"])[:2]
                                        - p_target[:2])) < d_tgt + 1e-6]
        # 나쁜 국소해 entry 배제 (같은 target 최소 J 의 2배 초과)
        j_min = min(e.get("J", np.inf) for e in near)
        if np.isfinite(j_min):
            near = [e for e in near if e.get("J", np.inf) <= 2.0 * j_min]
        near.sort(key=lambda e: float(np.linalg.norm(
            np.asarray(e["p_start"])[:2] - p_start[:2])))
        for k, ent in enumerate(near[:2]):
            if _try(warm=ent, tag=f"warm-db#{k}"):
                break
        if best is not None:
            _log(logger, f"  NLP: warm DB polish 적중 (target 거리 {d_tgt * 1e3:.0f}mm, "
                         f"p_start 거리 {np.linalg.norm(np.asarray(near[0]['p_start'])[:2] - p_start[:2]) * 1e3:.0f}mm)")

    # ---- ② cold multistart (warm 이 없거나 전부 불발일 때) ----
    if best is None:
        for k, init in enumerate(INIT_VARIANTS):
            _try(init, f"init{k}")
    # THR은 라운드마다 J 최소해를 dry-run 검사하고 통과하면 즉시 채택한다. 로봇에는
    # dry-run 이 없으므로 '고정 변형 전부 → 최소 J 채택' 으로 단순화하고, 전멸했을
    # 때만 무작위 섭동 라운드를 돌린다 (THR과 같은 base/분포).
    if best is None and NLP_MAX_RANDOM > 0:
        rng = np.random.default_rng(
            int(abs(p_target[0] * 1e4 + p_target[1] * 1e3 + p_start[0] * 1e2)) % 2**31)
        base = np.array([0.0, 0.6, 0.9, 0.0, 0.7, 0.0])
        for k in range(NLP_MAX_RANDOM):
            dq = base + rng.uniform(-0.5, 0.5, 6) * [1, 1, 1, 0, 1, 0]
            _try(dict(dq_swing=dq.tolist(), T0=float(rng.uniform(0.4, 1.3)),
                      chi0=float(rng.uniform(0.3, 0.7))), f"rand{k}")
            if best is not None:
                break
    if best is None:
        raise ValueError("NLP 계획 실패 (multistart 전멸):\n  " + "\n  ".join(fails[-8:]))

    res, tag = best
    q_of, qd_of, _ = throw_nlp._spline_eval(res["P"], res["t_f"])
    ts = np.arange(0.0, res["t_f"] + DT / 2, DT)
    Q = np.array([q_of(t) for t in ts])
    Qd = np.array([qd_of(t) for t in ts])
    rt = res["release_time"]
    _log(logger, f"  NLP: {tag} 채택 (J={res['J']:.3f}, t_f={res['t_f']:.3f}s, "
                 f"t*={res['t_star']:.3f}s, 윈도우 ±{rt / 2 * 1e3:.0f}ms, "
                 f"실패후보 {len(fails)}개)")
    return dict(ts=ts, Q=Q, Qd=Qd,
                release_window=(res["t_star"] - rt / 2, res["t_star"] + rt / 2),
                q_of=q_of, qd_of=qd_of, q_start=np.asarray(res["q_start"], float),
                info=dict(model="nlp", t_f=float(res["t_f"]),
                          t_star=float(res["t_star"]), J=float(res["J"]),
                          release_time=float(rt), tag=tag))


# ===========================================================================
# 2) DT — GP8 rig 학습 Decision Transformer (THR dt_planner.gp8_dt_traj_fn)
# ===========================================================================

# 벤더링된 DT 체크포인트 3종과, 실기 bin 목표(1.10/1.35/1.60 m)에서 실측한 결과:
#
#   gp8_dt_best.pth        THR weights_gp8/     2026-07-30  ← THR dt_planner 의 기본값
#       d_g 1.10 만 릴리즈하고 착지 0.21 m, 1.35/1.60 은 10스텝 만료(미-release).
#       현재 rig 에서는 **쓸 수 없다** (기하가 tool 0.240 으로 바뀌기 전 학습분).
#   gp8_dt_best_v9.pth     THR weights_gp8_v9/  2026-08-03 17:04  ← 기본값
#       1.10/1.35/1.60 전부 통과, 예측 착지 오차 6.2 / 7.9 / 1.2 cm.
#   gp8_dt_ft2_real-11.pth THR weights_gp8_ft2/ 2026-08-03 16:17 (실기 11회 파인튜닝)
#       너무 세게 던진다 (1.10 목표에 1.67 m 착지 — 56 cm 초과로 거부) + 릴리즈
#       타이밍 민감도 59~104 mm/10 ms. 실기 파인튜닝 데이터를 다시 볼 것.
#
# jitter 영상(2026-08-03 17:27)이 어느 체크포인트로 렌더링됐는지는 결과 json 에
# 기록이 없지만, v9 생성(17:04) 직후이고 위 실측상 v9 만 정상 동작하므로 v9 로
# 본다. `--weights` / GP8_THR_DT_WEIGHTS 로 바꿔 `--thr-scan` 비교 가능.
_THR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thr")
DT_WEIGHTS: str = os.environ.get(
    "GP8_THR_DT_WEIGHTS", os.path.join(_THR_DIR, "weights", "gp8_dt_best_v9.pth"))
DT_TARGET_RETURN: float = float(os.environ.get("GP8_THR_DT_RETURN", "1.0"))

_DT_LOCK = threading.Lock()
_DT_CACHE: dict = {}


def load_dt(weights: str = DT_WEIGHTS, logger=None):
    """(model, arm, cfg, torch) — 프로세스당 가중치별 1회 로드."""
    with _DT_LOCK:
        hit = _DT_CACHE.get(weights)
        if hit is not None:
            return hit
        import torch

        # NLP 쪽 threadpoolctl(1) 과 같은 이유: torch/BLAS 가 코어 수만큼 스레드를
        # 띄우면 ros2_control 의 4 ms RT UDP 루프가 굶어 RUN_STALL → comm-loss 로
        # 간다 (2026-07-21 실기 재현). 210k 파라미터라 1스레드로도 rollout 은 ms 단위.
        torch.set_num_threads(1)

        from gp8_control.skills.thr.dt_gp8_env import GP8Config, GP8ThrowArm
        from gp8_control.skills.thr.dt_model import ModelConfig, build_model

        model = build_model(ModelConfig(), dropout=0.0)     # 추론 — dropout 끔
        ckpt = model.load(weights)
        model = model.to(torch.device("cpu")).eval()
        cfg = GP8Config()
        arm = GP8ThrowArm(cfg, rng=np.random.default_rng(0))
        # τ(그리퍼 임계) = 학습셋 그리퍼 액션 평균 [논문 §5.1] — 체크포인트에 저장됨
        if ckpt.get("gripper_thresh") is not None:
            arm.gripper_thresh = ckpt["gripper_thresh"]
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _log(logger, f"  DT 로드: {os.path.basename(weights)} "
                     f"(params {n_par}, τ={arm.gripper_thresh:.4f})")
        _DT_CACHE[weights] = (model, arm, cfg, torch)
        return _DT_CACHE[weights]


def dt_rollout(d_goal, weights: str = DT_WEIGHTS, target_return=DT_TARGET_RETURN,
               logger=None):
    """THR `dt_planner.gp8_dt_rollout` 그대로 (Thr_DT evaluate_dt 프로토콜).
    리턴: (mem, k_rel, x_land_env)."""
    model, arm, cfg, torch = load_dt(weights, logger)
    arm.reset()
    arm.update_target(np.array([float(d_goal), 0.0, 0.0]))

    states = torch.zeros((0, model.state_dim), dtype=torch.float32)
    actions = torch.zeros((0, model.act_dim), dtype=torch.float32)
    rewards = torch.zeros(0, dtype=torch.float32)
    rtg = torch.tensor(target_return, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)

    mem, k_rel, x_land, done, k = [], None, None, False, 0
    while not done:
        s4 = arm.get_state()
        state = np.append(s4, arm.target[0])
        states = torch.cat(
            [states, torch.from_numpy(state).reshape(1, model.state_dim).float()])
        actions = torch.cat([actions, torch.zeros((1, model.act_dim))])
        rewards = torch.cat([rewards, torch.zeros(1)])
        with torch.no_grad():
            a = model.get_action(states, actions, rewards, rtg, timesteps)
        actions[-1] = a
        a = a.detach().cpu().numpy()
        gr_open = a[-1] < arm.gripper_thresh          # release 판정 [§5.1]
        reward, done, _, obj_pos, success = arm.step(a)
        mem.append((s4, a.copy(), reward, done, success))
        if gr_open and k_rel is None:
            k_rel, x_land = k, float(obj_pos[0])
        k += 1
        rewards[-1] = reward
        rtg = torch.cat([rtg, rtg[0, -1].reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), dtype=torch.long) * k], dim=1)
    return mem, k_rel, x_land


def dt_traj_fn(target, p_start=None, v_start=None, ctx=None, logger=None,
               weights: str = DT_WEIGHTS, calib=None):
    """THR `dt_planner.gp8_dt_traj_fn` 의 이식 (수정 없음).

    조준 yaw = atan2(t_y, t_x) [§5.1], d_g = 로봇 원점 기준 수평거리.
    ⚠ `p_start` 는 **쓰지 않는다** — DT 스윙은 항상 학습 시 home 자세
    (`GP8Config.home_pose`)에서 출발한다. 즉 물체를 어디서 집었는지가 던지기에
    영향을 주지 않는다 (NLP/phy 와 다른 점).
    """
    from gp8_control.skills.thr.dt_gp8_env import DG_MAX, DG_MIN

    p = np.asarray(target, float)
    d_target = float(np.hypot(p[0], p[1]))
    d_goal = d_target if calib is None else calib[0] * d_target + calib[1]
    d_clip = min(max(d_goal, DG_MIN), DG_MAX)
    if abs(d_clip - d_goal) > 1e-9:
        _log(logger, f"  (d_g {d_goal:.3f}m → 학습 범위 [{DG_MIN}, {DG_MAX}]로 클립)")

    mem, k_rel, x_land = dt_rollout(d_clip, weights=weights, logger=logger)
    if k_rel is None:
        raise ValueError(f"DT 미-release (d_g={d_clip:.2f}m, {len(mem)}스텝 만료)")

    return dt_plan_from_mem(mem, k_rel, x_land, p, d_clip, weights, logger)


def dt_plan_from_mem(mem, k_rel, x_land, target, d_clip, weights, logger=None,
                     alpha=None):
    """rollout 결과(mem) → traj_fn plan dict. `dt_traj_fn` 과 실기 데이터 수집이
    공유한다 (수집은 게인 적용 rollout 을 쓰므로 이 부분만 따로 뗐다)."""
    p = np.asarray(target, float)
    model, arm, cfg, torch = load_dt(weights, logger)
    Qp, t_nodes, t_rel = dt_cem.traj_from_seq(arm, None, k_rel, mem=mem)

    yaw = math.atan2(p[1], p[0])
    ts = np.arange(0.0, t_nodes[-1] + DT / 2, DT)
    Q = np.zeros((len(ts), 6))
    Q[:, 0] = yaw                                   # S: 조준 [§5.1]
    for j, col in enumerate([1, 2, 4]):             # L, U, B (planner 프레임 직접)
        Q[:, col] = np.interp(ts, t_nodes, Qp[:, j])
    Qd = np.gradient(Q, ts, axis=0)

    viol = np.maximum(GP8_Q_MIN - Q, Q - GP8_Q_MAX).max()
    if viol > 1e-6:      # env 가 이미 클램프하므로 정상적으로는 발생하지 않음
        _log(logger, f"  (DT 궤적 위치한계 {np.rad2deg(viol):.2f}° 초과 → 클립)")
        Q = np.clip(Q, GP8_Q_MIN, GP8_Q_MAX)
        Qd = np.gradient(Q, ts, axis=0)

    _log(logger, f"  DT: d_g={d_clip:.3f}m"
                 + (f", α={alpha:.2f}" if alpha is not None else "")
                 + f", {len(mem)}스텝, release k={k_rel} → t_rel={t_rel:.3f}s, "
                   f"아크 {ts[-1]:.3f}s (env 예측 착지 {x_land:.3f}m)")
    return dict(ts=ts, Q=Q, Qd=Qd, t_rel=float(t_rel), q_start=Q[0],
                info=dict(model="dt", d_g=d_clip, x_land_env=x_land,
                          k_rel=int(k_rel), n_steps=len(mem), alpha=alpha,
                          weights=os.path.basename(weights)))


# --- 실기 데이터 수집용: 액션 게인 α 를 곱한 rollout [논문 §4.4] ----------------
def dt_rollout_with_gain(d_goal, alpha, weights: str = DT_WEIGHTS,
                         target_return=DT_TARGET_RETURN, logger=None):
    """THR `dt_finetune_gp8._rollout_with_gain` 의 이식 (수정 없음).

    `dt_rollout` 과 같되 **모터 액션에만** 게인 α 를 곱하고 [-1,1] 로 클립한다
    (그리퍼 액션 a_gr 은 그대로 — 릴리즈 판정이 바뀌면 안 된다). 논문 §4.4 의
    α ~ U(1, α_max) 로 실기 영역을 탐색해 reality gap 을 메우는 절차다.

    저장하는 mem 의 액션은 **게인이 적용된, 실제로 실행되는 액션**이어야 한다 —
    파인튜닝이 배워야 할 것이 '실제로 실행된 것'이기 때문.
    리턴: (mem, k_rel, x_land_env).
    """
    model, arm, cfg, torch = load_dt(weights, logger)
    arm.reset()
    arm.update_target(np.array([float(d_goal), 0.0, 0.0]))

    states = torch.zeros((0, model.state_dim), dtype=torch.float32)
    actions = torch.zeros((0, model.act_dim), dtype=torch.float32)
    rewards = torch.zeros(0, dtype=torch.float32)
    rtg = torch.tensor(target_return, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)

    mem, k_rel, x_land, done, k = [], None, None, False, 0
    while not done:
        s4 = arm.get_state()
        state = np.append(s4, arm.target[0])
        states = torch.cat([states, torch.from_numpy(state).reshape(
            1, model.state_dim).float()])
        actions = torch.cat([actions, torch.zeros((1, model.act_dim))])
        rewards = torch.cat([rewards, torch.zeros(1)])
        with torch.no_grad():
            a = model.get_action(states, actions, rewards, rtg, timesteps)
        a = a.detach().cpu().numpy().copy()
        a[:-1] = np.clip(a[:-1] * float(alpha), -1.0, 1.0)      # 게인 [§4.4]
        actions[-1] = torch.from_numpy(a).float()
        gr_open = a[-1] < arm.gripper_thresh
        reward, done, _, obj_pos, success = arm.step(a)
        mem.append((s4, a.copy(), reward, done, success))
        if gr_open and k_rel is None:
            k_rel, x_land = k, float(obj_pos[0])
        k += 1
        rewards[-1] = reward
        rtg = torch.cat([rtg, rtg[0, -1].reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), dtype=torch.long) * k], dim=1)
    return mem, k_rel, x_land


# ===========================================================================
# 3) PHY — TossingBot Physics-only 탄도 컨트롤러 (Thr_Phy run_thr_sim)
# ===========================================================================

# Thr_Phy `run_thr_sim.py` 의 GP8 rig 상수 (선정 규칙은 그쪽 모듈 docstring).
PHY_ACCEL_DIST: float = 0.28    # 릴리즈 전 직선 가속 구간 [m]
PHY_FOLLOW_DIST: float = 0.10   # 릴리즈 후 팔로우스루 [m]
PHY_DECEL_DIST: float = 0.14    # 정지 구간 [m]

_PHY_CTRL = None


def _phy_ctrl():
    global _PHY_CTRL
    if _PHY_CTRL is None:
        from gp8_control.skills.thr.tossingbot.config import BallisticConfig
        from gp8_control.skills.thr.tossingbot.physics import BallisticController
        _PHY_CTRL = BallisticController(BallisticConfig(
            gravity=9.81,               # THR sim과 동일 (논문 9.8과 0.1% 차)
            release_height_ch=0.30,     # c_h [§III-C 선정 규칙]
            release_dist_cd=0.60,       # c_d [§III-C 선정 규칙]
        ))
    return _PHY_CTRL


def _phy_ramp(r, d3, speed):
    """Thr_Phy `run_thr_sim._ramp` 그대로: 정지 → 등가속 → (r에서 릴리즈) →
    등속 → 감속 정지. 가속 스텝 수를 정수로 고정해 릴리즈가 정확히 격자점에
    떨어지게 한다 (보간 오차·반올림이 개입하지 않는다)."""
    n_acc = max(2, int(round((2 * PHY_ACCEL_DIST / speed) / DT)))
    t_acc = n_acc * DT
    a_acc = speed / t_acc
    d_acc = 0.5 * speed * t_acc
    n_fol = max(1, int(round(PHY_FOLLOW_DIST / (speed * DT))))
    n_dec = max(2, int(round((2 * PHY_DECEL_DIST / speed) / DT)))
    a_dec = speed / (n_dec * DT)

    ts, ps, vs = [], [], []
    for k in range(n_acc + 1):                       # 가속: s = ½a t²
        t = k * DT
        ts.append(t)
        ps.append(r + d3 * (0.5 * a_acc * t * t - d_acc))
        vs.append(a_acc * t)
    for j in range(1, n_fol + 1):                    # 팔로우스루: 등속
        ts.append(t_acc + j * DT)
        ps.append(r + d3 * (speed * j * DT))
        vs.append(speed)
    s0 = speed * n_fol * DT
    for j in range(1, n_dec + 1):                    # 감속 정지
        tau = j * DT
        ts.append(t_acc + n_fol * DT + tau)
        ps.append(r + d3 * (s0 + speed * tau - 0.5 * a_dec * tau * tau))
        vs.append(speed - a_dec * tau)
    return (np.asarray(ts), np.asarray(ps),
            d3[None, :] * np.asarray(vs)[:, None], t_acc)


def phy_traj_fn(target, p_start=None, v_start=None, ctx=None, logger=None):
    """Thr_Phy `run_thr_sim.tossingbot_traj_fn` 의 이식 (수정 없음).

    ⚠ `p_start` 를 쓰지 않는다 — 릴리즈점 r 은 컨트롤러가 목표로부터 해석적으로
    정한다(반경 c_d=0.60 m, 높이 c_h=0.30 m 원 위, 목표와 xy 공선 [Eq.1]).
    관절궤적은 그 직선 램프를 position-only IK 로 따라간 것이다.
    """
    p = np.asarray(target, dtype=np.float64)
    r, v_hat, _ = _phy_ctrl().release(p)
    speed = float(np.linalg.norm(v_hat))
    d3 = v_hat / speed
    ts, ps, vels, t_rel = _phy_ramp(r, d3, speed)

    # position-only IK 로 관절 궤적화 (연속 seed 로 브랜치 고정)
    q = np.array([math.atan2(p[1], p[0]), 0.6, -0.5, 0.0, 1.0, 0.0])
    Q, Qd = [], []
    for k, (pt, vt) in enumerate(zip(ps, vels)):
        q, ok = ik_position(np.asarray(pt), q)
        if not ok:
            raise ValueError(f"IK 실패 @ sample {k}, tcp={np.round(pt, 3)}")
        Q.append(q.copy())
        # 관절속도는 수치미분이 아니라 **해석적** 최소노름 해 q̇ = J_v⁺ v_des —
        # J_v q̇ = v_des 를 정확히 만족하므로 재생 궤적에서 릴리즈 속도를 다시
        # 뽑아도 컨트롤러가 푼 v̂ 와 기계정밀도로 일치한다.
        Jv, _ = jacobian(q)
        Qd.append(np.linalg.pinv(Jv) @ vt)
    Q, Qd = np.asarray(Q), np.asarray(Qd)

    _log(logger, f"  PHY: |v_rel|={speed:.2f} m/s, 릴리즈점 "
                 f"({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f}) m, "
                 f"t_rel={t_rel:.3f}s, 아크 {ts[-1]:.3f}s")
    return dict(ts=ts, Q=Q, Qd=Qd, t_rel=float(t_rel), q_start=Q[0],
                info=dict(model="phy", v_rel=speed, r_release=r.tolist(),
                          qd_ratio=float(np.max(np.abs(Qd)
                                                / np.abs(GP8_QD_MAX)))))


# ===========================================================================
MODELS = ("nlp", "dt", "phy")


def get_traj_fn(model: str, weights: str | None = None, logger=None):
    """모델 이름 → traj_fn(target, p_start) 클로저 (THR bench_jitter.get_traj_fn 대응)."""
    if model == "nlp":
        return lambda target, p_start: nlp_traj_fn(target, p_start, logger=logger)
    if model == "dt":
        w = weights or DT_WEIGHTS
        return lambda target, p_start: dt_traj_fn(target, p_start, logger=logger,
                                                  weights=w)
    if model == "phy":
        return lambda target, p_start: phy_traj_fn(target, p_start, logger=logger)
    raise ValueError(f"알 수 없는 모델: {model} (가능: {', '.join(MODELS)})")


def geometry_summary() -> str:
    """벤더링된 THR 기하/공식화 파라미터 요약 (로그·스크립트 헤더용)."""
    return (f"tool={GP8_DIMS['tool']:.3f} (link6→TCP "
            f"{GP8_DIMS['d6'] + GP8_DIMS['tool']:.3f}m), "
            f"rt={throw_nlp.RELEASE_TIME:.3f}s, W1={throw_nlp.W1}, "
            f"W_ACC={throw_nlp.W_ACC:.0e}, qdd={throw_nlp.QDD_LIM[0] / throw_nlp.GP8_QD_MAX[0]:.0f}×qd, "
            f"cart_constraints={throw_nlp.CART_CONSTRAINTS}, "
            f"GRIP_OFF={throw_nlp.GRIP_OFF:.3f}")
