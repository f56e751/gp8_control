"""던지기 NLP 플래너 어댑터 (gp8_control.throwing.nlp) — **관절 경계조건 → multistart + warm DB → 궤적**.

    plan = nlp_plan(q_start, p_target, v0=…, p_f=…, vf=…)  → dict(ts, Q, Qd, Qdd, release_window, q_of/qd_of/qdd_of, …)

`throw_nlp.solve_throw_nlp` 를 (1) warm DB 최근접 entry polish → (2) 고정 초기해 6종 → (3) 무작위 섭동 순으로
호출해 첫 수렴 해(J 최소)를 고른다. 공개 API 는 `gp8_control.throwing.throw_planner`(로봇 관절 규약 입·출력).
`nlp_traj_fn(target, p_start, ctx)` 는 구 THR traj_fn 인터페이스 호환 래퍼(빌더·스캔 스크립트용).

로봇 이식에서 THR 원본과 의도적으로 다른 점
  1. multistart 가 **순차** — ROS2 노드에서 fork Pool 은 DDS 스레드와 충돌한다 (해의 정의는 동일, 벽시계만 느림).
  2. pybullet dry-run 충돌 검사 없음 — 대신 `nlp_throw_skill.check_arc` 가 dispatch 직전 FK 기반 게이트를 건다.
  3. 항력 추정 없음 (무항력 포물선, THR 기본과 동일).
공식화 상수(RELEASE_TIME, W1/W2/W_ACC, QDD_LIM, 관절 한계, GP8_DIMS, CART_CONSTRAINTS)는 throw_nlp.py 값 그대로.

warm DB (`warm_db_thr.pkl`, v2 축적 포맷): 공식화 파라미터(`warm_db_params`)가 같은 config 의 entry 만 쓴다.
종점 자세를 고정(p_f)하면 제약 수가 달라 lam_g 는 무시되고 primal 초기값으로만 쓰인다.
"""

from __future__ import annotations

import os
import time

import numpy as np

from gp8_control.throwing.nlp import throw_nlp
from gp8_control.throwing.nlp.throwing import GP8_DIMS, fk_pos

# planner 프레임 ↔ 로봇(URDF) 프레임 부호: q_robot = q_planner · PLANNER_SIGN (U·R·B·T 반대).
# 2026-09-18 검증: planner fk_pos(q) 와 robot/gp8.GP8.forward_kinematics(q·SIGN) 의 TCP 차이 6.7e-16 m
# (한계 안 200 자세) — 기하(link6→TCP 0.320 m)·부호 모두 일치.
PLANNER_SIGN = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
_PLANNER_SIGN = PLANNER_SIGN

# 계획 궤적 샘플 주기 = 로봇 스트림 격자 (robot_base.STREAM_DT, 500 Hz / 2 ms). 샘플이 격자에 정확히 놓이면 스트리머는
# 값을 그대로 내보낸다 (보간 없음 — 2026-09-21 사용자: "보간을 없애"). 1/240 이던 것을 격자에 맞춤.
DT: float = 0.002

# 던지기 시작점 = 잡은 물체 위치 + [0, 0, START_LIFT] (2026-09-18 사용자 규약). 시작 관절속도는 그 자세에서
# 벨트를 따라가는 J⁻¹·v_belt (throw_planner.object_boundary).
START_LIFT: float = float(os.environ.get("GP8_THR_START_LIFT", "0.05"))

# 종점 관절자세 p_f 의 **기본 규약** (다음 물체가 없을 때): throw_nlp.solve_throw_nlp(p_f=…).
#   GP8_THR_P_F=""(기본)      종점 자유
#   GP8_THR_P_F="start"      p_f = q_start (던진 뒤 출발 자세로 복귀)
#   GP8_THR_P_F="d1,…,d6"    관절 각도 [deg] (planner 부호 규약 = throw_nlp.Q_LO/Q_HI 기준)
# 다음 물체가 있으면 스킬이 그 잡는 자세를 p_f 로 넘기므로 이 값은 쓰이지 않는다.
NLP_P_F_MODE = os.environ.get("GP8_THR_P_F", "").strip()


def resolve_p_f(spec, q_start):
    """p_f 지정 → 6-vector[rad] 또는 None(자유)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        t = spec.strip().lower()
        if t in ("", "free", "none"):
            return None
        if t == "start":
            return np.asarray(q_start, float).copy()
        return np.deg2rad(np.array([float(x) for x in t.split(",")], float)).reshape(6)
    return np.asarray(spec, float).reshape(6)


# 픽 자세 회전 (도구 아래 방향, tests/static_pick_throw 와 동일) — 빌더가 시작 자세 IK 에 쓴다.
PICK_R_GRASP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def pick_joint(robot, p_tcp, q_init=None):
    """TCP 위치 p_tcp 에서 픽 자세(PICK_R_GRASP) 의 로봇 관절 (robot 규약). IK 실패면 None."""
    T = np.eye(4)
    T[:3, :3] = PICK_R_GRASP
    T[:3, 3] = np.asarray(p_tcp, float).reshape(3)
    q = (robot.inverse_kinematics(T) if q_init is None
         else robot.inverse_kinematics(T, q_init=np.asarray(q_init, float)[:6]))
    return None if q is None else np.asarray(q, float)[:6]


def q_start_in_limits(q) -> bool:
    """throw_nlp.solve_throw_nlp 의 q_start 검사와 같은 기준 (Q_LO/Q_HI, 1e-9 허용)."""
    q = np.asarray(q, float)
    return bool(np.all(q >= throw_nlp.Q_LO - 1e-9) and np.all(q <= throw_nlp.Q_HI + 1e-9))


# warm DB polish 해의 J 상한 — 이보다 크면(착지오차 항 W_ACC·err² 가 지배 = 나쁜 basin) 다음 후보 entry 도 풀어 본다
WARM_J_OK: float = float(os.environ.get("GP8_THR_WARM_J_OK", "20.0"))
# NLP 다시작(multistart) 예산 — 순차 실행이라 THR(fork 병렬)보다 보수적으로 잡는다.
NLP_MAX_RANDOM: int = int(os.environ.get("GP8_THR_NLP_MAX_RANDOM", "32"))
NLP_MAX_INIT: int = int(os.environ.get("GP8_THR_NLP_MAX_INIT", "6"))      # cold 고정 초기해(INIT_VARIANTS) 중 시도할 개수 (DB 빌더가 줄임)

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
        os.path.dirname(os.path.abspath(__file__)), "warm_db_thr.pkl"))

_WARM_DB: "list | None" = None      # 로드 캐시 (None=미로드, []=없음/불일치)

# 시간 예산 (2026-09-22 시뮬 random7: 대기점 종착 던지기의 DB 후보 3개가 차례로 3 s 씩 걸려 8.9 s → 팔이 6.9 s 정지).
# 계획 프로세스가 잡마다 set_deadline(던지기 시작 시각 − 여유) 를 걸면 nlp_plan 은 첫 후보 뒤로는 마감이 지났을 때 다음 후보를
# 풀지 않고, 해가 없으면 TimeoutError 로 빨리 실패한다 → 호출자가 준비점 종착으로 다시 푼다 (촘촘한 DB, 0.4~0.9 s).
_DEADLINE: "float | None" = None


def set_deadline(t_wall: "float | None") -> None:
    """후보 사이 마감 + (GP8_THR_ACCEPT_PARTIAL 일 때) polish 반복을 그 시각에 멈추는 마감 (throw_nlp.SOLVE_DEADLINE)."""
    global _DEADLINE
    _DEADLINE = None if t_wall is None else float(t_wall)
    throw_nlp.set_solve_deadline(_DEADLINE)


def _over_deadline() -> bool:
    return _DEADLINE is not None and time.time() > _DEADLINE


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
        qd_max=throw_nlp.QD_LIM.tolist(),   # 계획 속도 한계 (거버너 모드면 별도 config)
        # 카타시안 제약 유무·기준이 해의 basin/제약 개수를 바꾸므로 유효성 키다.
        # 2026-08-05 부터 기준이 원기둥(COL_R/COL_H) → TCP 엔벨로프로 바뀌었다 —
        # 키가 달라져 구 config 의 entry 와 자동으로 갈린다 (덮어쓰지 않는다).
        col=(("env", throw_nlp.CART_MIN_X, throw_nlp.CART_MIN_Z)
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
        p_f_mode=("fixed" if NLP_P_F_MODE not in ("", "free", "none") else "free"),   # 2026-09-18
        filt=getattr(throw_nlp, "FILTER_TAG", "off"),   # 2026-10-28 명령-스플라인 공식화 (컨트롤러 필터 모델)
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
    defaults = {"p_f_mode": "free"}          # 2026-09-18 이전 config 는 전부 종점 자유
    for cfg in db["configs"]:
        if all(cfg["params"].get(k, defaults.get(k)) == v for k, v in params.items()):
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
        pf = e.get("p_f")
        belt = e.get("belt")                     # 체인 entry: 벨트 속도가 다르면 경계 속도(vf)가 달라 별개 entry 로 공존 (2026-09-21)
        # end_mode 도 키에 넣는다 (2026-09-21): rest(정지 종점) entry 와 chain entry 는 같은 다음 점의 같은 픽 자세(p_f)를
        # 가지므로 이게 없으면 서로 덮어쓴다. 옛 entry 는 end_mode 가 없어 "linear".
        return (tuple(np.round(np.asarray(e["target"], float), 3)),
                tuple(np.round(np.asarray(e["p_start"], float), 3)),
                (None if pf is None else tuple(np.round(np.asarray(pf, float), 3))),
                (None if belt is None else round(float(belt), 3)),
                e.get("end_mode", "linear"))

    merged = {_key(e): e for e in cfg["entries"]}
    for e in new_entries:
        merged[_key(e)] = e
    cfg["entries"] = list(merged.values())
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(db, f)
    os.replace(tmp, path)                # 원자적 교체 — 중단돼도 파일이 깨지지 않는다
    return cfg["entries"]


# 공식화가 달라도 primal 초기값으로 쓸 수 있는 조건: 스플라인 구조가 같아야 P(6×N_CTRL)·u_pos 가 호환된다.
_STRUCT_KEYS = ("n_ctrl", "degree", "n_win", "rt", "dims", "flight")


class ColumnDB:
    """warm DB entry 들의 열(column) 표현 (2026-09-23). entry 마다 dict 29개 키를 파이썬 객체로 들면 entry 당 ≈13 KB 라 20만 entry 면
    프로세스당 2.5 GB 였다 (로봇 PC 15 GB, 계획 프로세스 3개). 열 배열(float32)로 두면 entry 당 ≈2 KB. `db[i]` 는 그 entry 의 dict 를
    그때 만들어 준다 (polish 가 쓰는 P·t_f·t_star·lam_g·u_pos 와 조회 키). len()/반복도 된다."""
    _VEC = (("target", 3), ("p_start", 3), ("p_next", 3), ("q_start", 6), ("v0", 6), ("p_f", 6), ("vf", 6), ("p_f_rate", 6), ("q_end", 6))
    _SCA = ("t_f", "t_star", "J", "belt", "z_travel", "lift_speed", "descend_speed", "obj_z", "t_ref")

    def __init__(self, cols: dict):
        self.c = cols
        self.n = int(cols["n"])

    @classmethod
    def from_entries(cls, entries: list) -> "ColumnDB":
        n = len(entries)
        cols: dict = {"n": n}
        for k, d in cls._VEC:
            a = np.full((n, d), np.nan, np.float64)
            for i, e in enumerate(entries):
                v = e.get(k)
                if v is not None:
                    a[i] = np.asarray(v, float).reshape(d)
            cols[k] = a.astype(np.float32) if k not in ("q_start", "v0", "vf", "p_f", "p_f_rate", "q_end") else a   # 관절값은 float64 유지
        for k in cls._SCA:
            cols[k] = np.array([np.nan if e.get(k) is None else float(e[k]) for e in entries], np.float64)
        cols["end_mode"] = np.array([str(e.get("end_mode", "linear")) for e in entries])
        P0 = next((np.asarray(e["P"]) for e in entries if e.get("P") is not None), None)
        if P0 is not None:
            P = np.zeros((n,) + P0.shape, np.float32); has_p = np.zeros(n, bool)
            for i, e in enumerate(entries):
                if e.get("P") is not None and np.asarray(e["P"]).shape == P0.shape:
                    P[i] = np.asarray(e["P"], np.float32); has_p[i] = True
            cols["P"], cols["has_P"] = P, has_p
        else:
            cols["P"], cols["has_P"] = np.zeros((n, 0), np.float32), np.zeros(n, bool)
        U0 = next((np.asarray(e["u_pos"]) for e in entries if e.get("u_pos") is not None), None)
        if U0 is not None:
            U = np.full((n,) + U0.shape, np.nan, np.float64)
            for i, e in enumerate(entries):
                if e.get("u_pos") is not None and np.asarray(e["u_pos"]).shape == U0.shape:
                    U[i] = np.asarray(e["u_pos"], float)
            cols["u_pos"] = U
        else:
            cols["u_pos"] = np.full((n, 0), np.nan)
        L = max((len(np.asarray(e["lam_g"]).ravel()) for e in entries if e.get("lam_g") is not None), default=0)
        lam = np.zeros((n, L), np.float32); lam_len = np.zeros(n, np.int32)
        for i, e in enumerate(entries):
            v = e.get("lam_g")
            if v is not None:
                v = np.asarray(v, np.float32).ravel(); lam[i, :len(v)] = v; lam_len[i] = len(v)
        cols["lam_g"], cols["lam_len"] = lam, lam_len
        return cls(cols)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def __getitem__(self, i) -> dict:
        i = int(i); c = self.c
        if i < 0 or i >= self.n:
            raise IndexError(i)
        e: dict = {}
        for k, _d in self._VEC:
            v = c[k][i]
            e[k] = None if np.isnan(v[0]) else np.asarray(v, float)
        for k in self._SCA:
            v = float(c[k][i]); e[k] = None if np.isnan(v) else v
        e["end_mode"] = str(c["end_mode"][i])
        e["P"] = np.asarray(c["P"][i]) if bool(c["has_P"][i]) else None
        u = c["u_pos"][i]; e["u_pos"] = None if (u.size == 0 or np.isnan(u[0])) else np.asarray(u, float)
        ln = int(c["lam_len"][i])
        if ln > 0:
            e["lam_g"] = np.asarray(c["lam_g"][i, :ln])
        return e


def _mem_available_mb() -> float:
    """/proc/meminfo 의 MemAvailable (MB). 못 읽으면 inf (검사 생략)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return float("inf")


# 전체 DB pickle 을 파이썬 객체로 펼치는 순간 RSS 가 파일 크기의 ≈3.7배 필요하다 (938 MB 파일 → 3.5 GB, 8 s; 2026-09-22 실측).
# 로봇 PC(15 GB, swap 없음)에서 hw 앱과 시뮬 앱이 겹쳐 계획 프로세스 4개가 동시에 이걸 하자 메모리가 바닥나 시스템이 멈췄다
# (같은 날 재부팅 2회; 커널 로그도 못 남김). 대책:
#   ① 필터(GP8_THR_DB_TARGETS/BELT)된 entry 를 캐시 파일(warm_db_thr.cache_<key>.pkl)로 두고 그것만 로드 → 0.5 GB, ~1 s
#   ② 캐시가 없을 때의 전체 로드는 파일 lock 으로 한 프로세스씩 (다른 워커는 기다렸다 캐시를 읽는다)
#   ③ 가용 메모리가 전체 로드에 못 미치면 warm DB 없이(cold) 간다 — 느려질지언정 PC 는 살린다
_FULL_LOAD_RSS_RATIO = 3.8          # 전체 로드 순간 RSS / 파일 크기
_FULL_LOAD_MARGIN_MB = 1500.0       # 그 위에 남겨 둘 여유 (앱·드라이버·데스크톱)


def _norm_targets(keep: str) -> str:
    """타겟 필터 문자열을 순서·표기와 무관하게 (bin 을 JSON 에 어떤 순서로 적어도 같은 캐시)."""
    try:
        tg = sorted(tuple(round(float(v), 4) for v in t.split(",")) for t in keep.split(";") if t.strip())
    except ValueError:
        return keep
    return ";".join(",".join(f"{v:g}" for v in t) for t in tg)


def _db_cache_path(keep: str, keep_belt: str, params: dict, raw: bool = False) -> str:
    """필터·DB 파일(크기+mtime)·공식화 파라미터로 키를 만든 캐시 경로 — 셋 중 하나라도 바뀌면 다른 파일이라 자동 무효화.
    타겟 필터는 정렬해 키에 넣는다 (2026-10-02, bin 3개 — 순서만 다른 같은 bin 세트가 캐시를 다시 만들지 않게).
    raw=True 는 그 전 규칙(적힌 순서 그대로)의 경로 — 이미 있는 캐시를 계속 쓰기 위해 load_warm_db 가 먼저 찾아본다."""
    import hashlib
    st = os.stat(WARM_DB_PATH)
    k = keep if raw else _norm_targets(keep)
    key = hashlib.md5(repr(("col1", k, keep_belt, st.st_size, int(st.st_mtime),
                            sorted((a, b) for a, b in params.items() if a != "filt"))).encode()).hexdigest()[:10]   # filt 는 키에서 뺀다 (아래 primal-only 처리)
    return os.path.splitext(WARM_DB_PATH)[0] + f".cache_{key}.pkl"


def _write_db_cache(path: str, db: "ColumnDB", n0: int, params: dict, logger=None) -> None:
    import glob, pickle
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(dict(columns=db.c, n0=n0, params=params), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError as ex:
        _log(logger, f"  warm DB: 캐시 저장 실패 ({ex}) — 이번엔 그냥 진행")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return
    for old in glob.glob(os.path.splitext(WARM_DB_PATH)[0] + ".cache_*.pkl"):      # 옛 키(다른 필터·옛 DB)의 캐시 정리
        if os.path.abspath(old) != os.path.abspath(path):
            try:
                os.remove(old)
            except OSError:
                pass
    _log(logger, f"  warm DB: 캐시 저장 {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.0f} MB) — 다음 로드부터 전체 DB 를 안 펼친다")


def load_warm_db(logger=None) -> list:
    """지금 공식화와 일치하는 entry 리스트. 프로세스당 1회 로드.

    GP8_THR_DB_TARGETS / GP8_THR_DB_BELT 필터가 있으면 필터된 entry 의 **캐시 파일**을 먼저 찾아 그것만 로드한다 (전체 DB 를
    펼치는 3.5 GB 순간 메모리를 피함 — 위 주석). 캐시가 없으면 (가용 메모리가 되는 한 프로세스가) 전체를 읽어 만들어 둔다.

    일치하는 config 가 없으면(속도·가속 한계 등을 바꾼 직후) 스플라인 구조(_STRUCT_KEYS)가 같은 config 중 가장 큰 것을
    **primal 초기값 전용**으로 쓴다 (lam_g 제거 → dual 은 안 씀; 활성 제약이 달라 정확하지 않으므로). cold(25 s) 대신
    2~3 s. 제대로 하려면 새 공식화로 DB 를 재수집한다 (build_chain_db.py)."""
    global _WARM_DB
    if _WARM_DB is not None:
        return _WARM_DB
    if (os.environ.get("GP8_SEP", "0") not in ("0", "", "false")
            and os.environ.get("GP8_SEP_STOP_END", "1") not in ("0", "", "false")):
        # 11-04 사용자 "모두 분리 NLP로 풀기로 했으니까 설정이 DB랑 안 맞다는 둥의 경고창은 뜨지 않게 해": 분리 방식 (정지 종점) 은 모든 던지기를
        # 분리 NLP 로 풀어 이 DB 의 해를 쓰지 않는다 (분리 초기값은 GP8_SEP_DB). 읽지 않으니 공식화 불일치 경고·전체 로드 (5 GB) 도 없다.
        _WARM_DB = []
        return _WARM_DB
    import pickle
    params = warm_db_params()
    keep = os.environ.get("GP8_THR_DB_TARGETS", "").strip()
    keep_belt = os.environ.get("GP8_THR_DB_BELT", "").strip()          # 빌더 워커·런타임: 같은 벨트 속도의 entry 만 (옛 linear 제외)
    cache = _db_cache_path(keep, keep_belt, params) if ((keep or keep_belt) and os.path.exists(WARM_DB_PATH)) else None
    if cache is not None and not os.path.exists(cache):
        old = _db_cache_path(keep, keep_belt, params, raw=True)          # 옛 규칙 이름의 캐시가 있으면 그걸 쓴다
        if os.path.exists(old):
            cache = old
    lock = None
    if cache is not None:
        import fcntl
        lock = open(cache + ".lock", "a+")
        fcntl.flock(lock, fcntl.LOCK_EX)              # 캐시 생성은 한 프로세스만 — 나머지는 여기서 기다렸다 캐시를 읽는다
    try:
        if cache is not None and os.path.exists(cache):
            t0 = time.time()
            try:
                with open(cache, "rb") as f:
                    c = pickle.load(f)
                _WARM_DB = ColumnDB(c["columns"]) if "columns" in c else ColumnDB.from_entries(list(c["entries"]))
                _log(logger, f"  warm DB: 캐시 {os.path.basename(cache)} entry {len(_WARM_DB)}/{c.get('n0', '?')}개 "
                             f"({time.time() - t0:.1f}s; 타겟 {keep or '전부'}, 벨트 {keep_belt or '전부'})")
                if params.get("filt", "off") != "off" and c.get("params", {}).get("filt", "off") != params["filt"]:
                    _WARM_DB.c["lam_len"] = np.zeros_like(np.asarray(_WARM_DB.c["lam_len"]))   # 다른 공식화의 쌍대값은 안 쓴다 → primal 초기값만
                    _log(logger, f"  warm DB: 컨트롤러 필터 공식화({params['filt']}) 와 캐시 공식화가 달라 entry 를 primal 초기값으로만 사용 (lam_g 제거)")
                return _WARM_DB
            except Exception as ex:                                     # noqa: BLE001
                _log(logger, f"  warm DB: 캐시 읽기 실패 ({ex}) → 원본에서 다시 만든다")
                _WARM_DB = None
        # ---- 전체 로드: 메모리부터 확인 ----
        size_mb = os.path.getsize(WARM_DB_PATH) / 1e6 if os.path.exists(WARM_DB_PATH) else 0.0
        need_mb = size_mb * _FULL_LOAD_RSS_RATIO + _FULL_LOAD_MARGIN_MB
        avail_mb = _mem_available_mb()
        if size_mb > 0 and avail_mb < need_mb:
            msg = (f"warm DB 전체 로드({size_mb:.0f} MB 파일 → 순간 ≈{need_mb:.0f} MB 필요)에 가용 메모리 {avail_mb:.0f} MB 가 모자람 — "
                   f"PC 멈춤 방지를 위해 warm DB 없이(cold, 던지기당 수십 초) 진행. 다른 앱·시뮬을 끄고 다시 시작하면 캐시가 만들어진다")
            (logger.warning if logger is not None and hasattr(logger, "warning") else print)("WARN " + msg)
            _WARM_DB = []
            return _WARM_DB
        t0 = time.time()
        db = db_read()
        cfg = db_select(db, params)
        _WARM_DB = cfg["entries"] if cfg else []
        n0 = len(_WARM_DB)
        if _WARM_DB and (keep or keep_belt):
            if keep:
                tg = [np.array([float(v) for v in t.split(",")]) for t in keep.split(";") if t.strip()]
                _WARM_DB = [e for e in _WARM_DB if any(np.linalg.norm(np.asarray(e["target"], float) - t) < 1e-3 for t in tg)]
            if keep_belt:
                b = float(keep_belt)
                _WARM_DB = [e for e in _WARM_DB if e.get("belt") is not None and abs(float(e["belt"]) - b) < 1e-6
                            and e.get("end_mode", "linear") != "linear"]
            _log(logger, f"  warm DB: 필터 (타겟 {keep or '전부'}, 벨트 {keep_belt or '전부'}) → {len(_WARM_DB)}/{n0} entry "
                         f"(전체 로드 {time.time() - t0:.1f}s)")
            # 필터로 버린 entry 의 메모리를 OS 에 돌려준다 — 안 하면 워커 RSS 가 전체 DB 크기(3 GB+)로 남아 60 GB 서버에서
            # 16 워커면 swap 이 찬다 (2026-09-22 실측). glibc 는 free 된 작은 블록을 붙들고 있어 malloc_trim 이 필요.
            del db, cfg
            import gc, ctypes
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass
            cfg = None; db = {"configs": []}
        if _WARM_DB:
            t1 = time.time()
            _WARM_DB = ColumnDB.from_entries(_WARM_DB)                     # dict 리스트 → 열 배열 (entry 당 13 KB → 2 KB)
            import gc, ctypes
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass
            _log(logger, f"  warm DB: 열 배열 변환 {len(_WARM_DB)} entry ({time.time() - t1:.1f}s)")
            if cache is not None:
                _write_db_cache(cache, _WARM_DB, n0, params, logger)
        if _WARM_DB:
            _log(logger, f"  warm DB: {os.path.basename(WARM_DB_PATH)} "
                         f"entry {len(_WARM_DB)}개 (현재 공식화 일치)")
        elif db["configs"]:
            same = [c for c in db["configs"]
                    if all(c["params"].get(k) == params.get(k) for k in _STRUCT_KEYS)]
            if same:
                best = max(same, key=lambda c: len(c["entries"]))
                diff = [k for k in params if best["params"].get(k) != params.get(k)]
                _WARM_DB = ColumnDB.from_entries([{k: v for k, v in e.items() if k != "lam_g"} for e in best["entries"]])
                _log(logger, f"  warm DB: 현재 공식화와 일치하는 config 없음 → 구조가 같은 config 의 entry "
                             f"{len(_WARM_DB)}개를 primal 초기값으로만 사용 (차이: {', '.join(diff)}) — "
                             f"새 공식화로 DB 재수집 권장 (build_chain_db.py)")
            else:
                _log(logger, f"  warm DB: 현재 공식화와 일치하는 config 없음 — cold 진행 "
                             f"(다른 공식화 config {len(db['configs'])}개는 보존됨)")
        return _WARM_DB
    finally:
        if lock is not None:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

_WARM_DB_ARRAYS = None


def _warm_db_arrays(db: list) -> dict:
    """nlp_plan 의 최근접 조회용 열 배열 (DB 리스트 객체당 1회 생성). target_xy, p_start_xy, p_f(없으면 0)+has_pf, v0, vf, end_mode."""
    global _WARM_DB_ARRAYS
    if (_WARM_DB_ARRAYS is None or _WARM_DB_ARRAYS["id"] != id(db) or _WARM_DB_ARRAYS["n"] != len(db)) and isinstance(db, ColumnDB):
        c = db.c; n = len(db)
        pf = np.nan_to_num(np.asarray(c["p_f"], float)); has = ~np.isnan(np.asarray(c["p_f"], float)[:, 0])
        _WARM_DB_ARRAYS = dict(
            id=id(db), n=n,
            target_xy=np.asarray(c["target"], float)[:, :2], target_z=np.asarray(c["target"], float)[:, 2],
            p_start_xy=np.asarray(c["p_start"], float)[:, :2],
            p_f=pf, has_pf=has, v0=np.nan_to_num(np.asarray(c["v0"], float)), vf=np.nan_to_num(np.asarray(c["vf"], float)),
            p_next_xy=np.asarray(c["p_next"], float)[:, :2], end_mode=np.asarray(c["end_mode"]))
        return _WARM_DB_ARRAYS
    if _WARM_DB_ARRAYS is None or _WARM_DB_ARRAYS["id"] != id(db) or _WARM_DB_ARRAYS["n"] != len(db):
        n = len(db)
        pf = np.zeros((n, 6)); has = np.zeros(n, bool); v0a = np.zeros((n, 6)); vfa = np.zeros((n, 6))
        pnext = np.full((n, 2), np.nan)                       # 종점(다음 물체 hover) TCP xy — 체인 entry 만
        for i, e in enumerate(db):
            if e.get("p_f") is not None:
                pf[i] = np.asarray(e["p_f"], float).reshape(6); has[i] = True
                if e.get("p_next") is not None:
                    pnext[i] = np.asarray(e["p_next"], float)[:2]
            if e.get("v0") is not None:
                v0a[i] = np.asarray(e["v0"], float).reshape(6)
            if e.get("vf") is not None:
                vfa[i] = np.asarray(e["vf"], float).reshape(6)
        _WARM_DB_ARRAYS = dict(
            id=id(db), n=n,
            target_xy=np.array([np.asarray(e["target"], float)[:2] for e in db]).reshape(n, 2),
            target_z=np.array([float(np.asarray(e["target"], float)[2]) for e in db]),
            p_start_xy=np.array([np.asarray(e["p_start"], float)[:2] for e in db]).reshape(n, 2),
            p_f=pf, has_pf=has, v0=v0a, vf=vfa, p_next_xy=pnext,
            end_mode=np.array([e.get("end_mode", "linear") for e in db]))
    return _WARM_DB_ARRAYS


def reset_warm_db_cache() -> None:
    """공식화 상수를 바꾼 뒤 다시 조회하려면 호출 (빌더가 쓴다)."""
    global _WARM_DB, _WARM_DB_ARRAYS
    _WARM_DB = None
    _WARM_DB_ARRAYS = None


# ===========================================================================
# NLP 계획 (CasADi/IPOPT B-spline — THR nlp_planner.nlp_traj_fn 의 로봇판)
# ===========================================================================

def nlp_traj_fn(target, p_start, v_start=None, ctx=None, logger=None):
    """THR `nlp_planner.plan_nlp_throw` + `nlp_traj_fn` 의 순차 이식 (traj_fn 인터페이스 호환 래퍼).

    시작 자세는 ctx["q_start"](planner 규약, NLP 한계 안이면 그대로) 또는 p_start 의 한계 안 IK.
    관절 경계 속도·종점 자세는 ctx["v0"] / ctx["vf"] / ctx["p_f"] 로 넘긴다 — 본체는 `nlp_plan`.
    (v_start 는 구 THR 인터페이스의 TCP 속도 자리라 0 만 허용한다.)

    release 는 고정 시각이 아니라 **윈도우** [t*−rt/2, t*+rt/2] 로 돌려준다.
    플래너가 W_ACC 로 윈도우 전 구간의 착탄 정확도를 강제하므로 그 안 어디서
    놓아도 맞는다 — 밸브 지터에 대한 내성이 여기서 나온다.
    """
    p_target = np.asarray(target, float)
    p_start = np.asarray(p_start, float)
    if v_start is not None and not np.allclose(np.asarray(v_start, float), 0.0):
        raise ValueError("nlp_traj_fn 의 v_start 는 TCP 속도 자리(0 만 허용) — 관절 시작속도는 "
                         "ctx['v0'] 또는 nlp_plan(q_start, v0=...) 로")
    ctx = ctx if isinstance(ctx, dict) else {}

    # 시작자세를 먼저 확정 — 한계 안 해가 없으면 어떤 초기해로도 못 푼다
    # (P₀는 상수라 hull 제약이 못 잡고 dense 검증이 전 후보를 기각).
    # ctx["q_start"] (planner 규약) 가 있고 NLP 위치 한계 안이면 그 자세에서 그대로 출발
    # (lift 0 — 픽 자세). 아니면 p_start 의 한계 안 IK.
    q_hint = ctx.get("q_start")
    if q_hint is not None and q_start_in_limits(q_hint):
        q_start = np.asarray(q_hint, float)
        _log(logger, "  NLP: 픽 자세를 시작 자세로 사용 (lift 0 — 제자리 이동 없음)")
    else:
        if q_hint is not None:
            _log(logger, "  NLP: 픽 자세가 NLP 위치 한계 밖 → 같은 점의 한계 안 IK 자세로 "
                         "출발 (흡착점에서 제자리 재정렬)")
        q_start = throw_nlp.ik_start_pose(p_start)      # ValueError = 이 지점 불가

    p_f = resolve_p_f(ctx.get("p_f", NLP_P_F_MODE), q_start)          # 2026-09-18 종점 자세
    return nlp_plan(q_start, p_target, v0=ctx.get("v0"), p_f=p_f, vf=ctx.get("vf"),
                    p_start=p_start, logger=logger)


def nlp_plan(q_start, p_target, *, v0=None, p_f=None, vf=None, p_f_rate=None, t_ref=0.0,
             p_start=None, end_cart=None, warm_init=None, logger=None, hist_s=None):
    """관절 경계조건으로 NLP 던지기 계획 한 번 (planner 프레임, rad · rad/s · m).

    q_start / v0 : 시작 관절 자세·속도 (v0 None = 정지 출발)
    p_f / vf     : 종점 관절 자세·속도 (p_f None = 종점 자세 자유, vf None = 정지 종료).
                   체인 던지기에서는 p_f = 다음 물체를 잡는 자세, vf = 그 자세에서 벨트를 따라가는 관절속도.
    p_f_rate / t_ref : 움직이는 종점 — q(t_f) = p_f + p_f_rate·(t_f − t_ref) (p_f 는 궤적 시계 t_ref 의 자세).
                   보통 p_f_rate = vf. 이렇게 주면 "t_f ↔ 종점 위치" 순환을 solver 가 한 번에 푼다.
    p_target     : 착지 조준점 (x, y, z)
    p_start      : 시작 TCP (warm DB 최근접 entry 조회·난수 시드용; None 이면 FK(q_start))
    end_cart     : 정확한 움직이는 종점 dict(p, v, t_ref, j6) — throw_nlp.solve_throw_nlp 참고. 주면 p_f/vf/p_f_rate 는
                   초기값·DB 조회용이고 종점은 FK 등식으로 정확히 걸린다 (2026-09-21).
    warm_init    : 직전 해(리턴 dict 의 raw: P, t_f, t_star, lam_g, u_pos)로 먼저 polish — 같은 문제를 t_ref 만 바꿔
                   다시 풀 때(종점 재선형화) DB 최근접보다 훨씬 가깝다. 실패하면 DB → cold 순으로 진행.

    리턴 dict: ts (N,), Q (N,6), Qd (N,6), Qdd (N,6), release_window (t0, t1), q_of/qd_of/qdd_of (스플라인
    평가기), q_start, v0, vf, p_f, info(model, t_f, t_star, J, release_time, tag).
    실패(multistart 전멸)면 ValueError.
    """
    q_start = np.asarray(q_start, float).reshape(6)
    if not q_start_in_limits(q_start):
        j = int(np.argmax(np.maximum(throw_nlp.Q_LO - q_start, q_start - throw_nlp.Q_HI)))
        raise ValueError(f"q_start 가 NLP 위치 한계 밖 (관절 {'SLURBT'[j]} "
                         f"{np.rad2deg(q_start[j]):+.1f}°, 한계 [{np.rad2deg(throw_nlp.Q_LO[j]):.0f}, "
                         f"{np.rad2deg(throw_nlp.Q_HI[j]):.0f}]°)")
    p_target = np.asarray(p_target, float).reshape(3)
    p_start = fk_pos(q_start) if p_start is None else np.asarray(p_start, float).reshape(3)
    v0 = np.zeros(6) if v0 is None else np.asarray(v0, float).reshape(6)
    vf = np.zeros(6) if vf is None else np.asarray(vf, float).reshape(6)
    p_f = None if p_f is None else np.asarray(p_f, float).reshape(6)
    for name, v in (("v0", v0), ("vf", vf)):
        if np.any(np.abs(v) > throw_nlp.QD_LIM + 1e-9):
            j = int(np.argmax(np.abs(v) - throw_nlp.QD_LIM))
            raise ValueError(f"{name} 관절 {'SLURBT'[j]} {np.rad2deg(v[j]):+.0f}°/s 가 계획 속도 한계 "
                             f"{np.rad2deg(throw_nlp.QD_LIM[j]):.0f}°/s 를 넘는다")
    if end_cart is not None:
        _log(logger, f"  NLP: 정확한 움직이는 종점 FK(q(t_f)) = p(t_ref={float(end_cart.get('t_ref', t_ref)):.2f}s) + v·(t_f−t_ref), "
                     f"p {np.round(end_cart['p'], 3).tolist()} m, v {np.round(end_cart['v'], 3).tolist()} m/s, "
                     f"툴 수직 + J6 {np.rad2deg(float(end_cart['j6'])):.0f}°")
    elif p_f is not None and p_f_rate is not None and np.any(p_f_rate):
        _log(logger, f"  NLP: 움직이는 종점 q(t_f)=p_f(t_ref={t_ref:.2f}s)+rate·(t_f−t_ref), "
                     f"p_f {np.round(np.rad2deg(p_f), 1).tolist()}°, |rate|max {np.rad2deg(np.max(np.abs(p_f_rate))):.1f}°/s")
    elif p_f is not None:
        _log(logger, f"  NLP: 종점 자세 고정 q(t_f)=p_f {np.round(np.rad2deg(p_f), 1).tolist()}°")
    if np.any(v0) or np.any(vf):
        _log(logger, f"  NLP: 관절 경계속도 |v0|max {np.rad2deg(np.max(np.abs(v0))):.1f}°/s, "
                     f"|vf|max {np.rad2deg(np.max(np.abs(vf))):.1f}°/s")
    fails: list[str] = []
    best = None

    def _try(init=None, tag="", warm=None):
        nonlocal best
        _t0 = time.time(); throw_nlp.TIMING.clear()
        try:
            res = throw_nlp.solve_throw_nlp(p_start, p_target, v0=v0, vf=vf, init=init,
                                            warm_data=warm, q_start=q_start, p_f=p_f,
                                            p_f_rate=p_f_rate, t_ref=t_ref, end_cart=end_cart, hist_s=hist_s)
            _t1 = time.time()
            if logger is not None:
                rel = (lambda t: "" if _DEADLINE is None else f"{t - _DEADLINE:+.3f}")
                _log(logger, f"  NLP 시간: {tag} 전체 {_t1 - _t0:.2f}s, 끝 = 마감 {rel(_t1)}s | " + "; ".join(
                    f"ω{x['w']:g}: 풀이 {x['solve']:.2f}s {x['it']}반복 {x['st'][:20]} 끝 마감 {rel(x['end'])} 투영 {x['part']:.2f}s"
                    for x in throw_nlp.TIMING))
        except (RuntimeError, AssertionError, ValueError) as e:
            fails.append(f"{tag}: NLP 실패 — {type(e).__name__}: {(str(e).splitlines() or ['?'])[0][:120]}")
            return False
        # 0.06°(1e-3 rad) 이하 잔존 위반은 물리적으로 무의미 → 허용 (THR과 동일)
        if res.get("pos_viol_dense", 0.0) > 1e-3:
            fails.append(f"{tag}: 위치한계 잔존 위반 {res['pos_viol_dense']:.1e}")
            return False
        if best is None or res["J"] < best[0]["J"]:
            best = (res, tag)
        return True

    # ---- ① warm DB polish 먼저 ----
    #      target 최근접 entry 들 중 시작점이 가까운 순으로 상위 2개에 full warm
    #      start (primal+dual 재시동) → cold 수십 초 대신 1~2 초. entry 2개를 보는
    #      이유는 같은 target 의 다른 basin 을 커버하기 위해서다.
    #      ⚠ (2026-08-06) THR plan_nlp_throw 에 있던 'J ≤ 2×j_min 나쁜 국소해 배제'
    #      필터는 **삭제했다**. J 는 p_start 가 다르면 2~6배씩 정당하게 벌어지는
    #      값이라 같은 target 의 서로 다른 시작점 entry 끼리 J 비교는 무의미하고,
    #      바닥에 붙여 스윙하는 basin 일수록 J 가 낮아 이 필터가 안전한
    #      정확-일치 entry 를 버리고 grazing entry 를 고르는 사고가 실측됐다
    #      ((0.5,0.1)→(1.6,0.225): 12개 중 11개 폐기 → 200 mm 밖 entry polish
    #      → z_min=+0.0199 로 dispatch 게이트 위반 보고).
    have_prev = warm_init is not None and warm_init.get("P") is not None
    _t_db0 = time.time()
    db = load_warm_db(logger)
    near = []; db_close = False
    if db:
        # 최근접 후보 선택 — 벡터화 (2026-09-21: entry 70k 에서 파이썬 루프·sort 가 solve(0.7 s)보다 긴 1.6 s 였다).
        # 정렬 키는 이전과 동일: (공식화 불일치, round(종점 자세 거리, 3), round(경계속도 거리, 2), 시작 TCP xy 거리).
        A = _warm_db_arrays(db)
        d_t = np.linalg.norm(A["target_xy"] - p_target[:2], axis=1)
        d_tgt = float(d_t.min())
        idx = np.flatnonzero(d_t < d_tgt + 1e-6)                          # 같은(최근접) 타겟의 entry 들
        ps_dist = np.linalg.norm(A["p_start_xy"][idx] - p_start[:2], axis=1)
        if p_f is None:
            pf_dist = np.where(A["has_pf"][idx], 10.0, 0.0)
        else:
            # 종점 거리는 TCP 공간(m)으로: entry 의 p_next(다음 물체 xy) vs 지금 종점 TCP xy. 관절 거리(rad)를 1차 키로 쓰면
            # 종점만 가깝고 시작점이 20~40 cm 먼 entry 가 뽑혀 polish 가 6 s 걸렸다 (2026-09-21 실측) — 시작·종점 거리를 합쳐 본다.
            p_end_xy = (np.asarray(end_cart["p"], float)[:2] if end_cart is not None else fk_pos(p_f)[:2])
            pn = A["p_next_xy"][idx]
            pf_tcp = np.where(np.isnan(pn[:, 0]), np.nan, np.linalg.norm(np.nan_to_num(pn) - p_end_xy, axis=1))
            pf_joint = np.where(A["has_pf"][idx], np.linalg.norm(A["p_f"][idx] - p_f, axis=1), 10.0)
            pf_dist = np.where(np.isnan(pf_tcp), np.minimum(10.0, pf_joint), pf_tcp)   # p_next 없는 옛 entry 는 관절 거리
        v_dist = np.linalg.norm(A["v0"][idx] - v0, axis=1) + np.linalg.norm(A["vf"][idx] - vf, axis=1)
        # 종점 종류: exact(움직이는 종점 FK 등식) / rest(정지 종점: p_f 고정, 속도 0 — 상류 대기 hover·home 복귀) / linear(옛 1차 근사)
        want_mode = ("exact" if end_cart is not None else
                     ("rest" if (p_f is not None and (p_f_rate is None or not np.any(p_f_rate)) and not np.any(vf)) else "linear"))
        # 종점 공식화가 다르면 lam_g 를 못 쓰므로(primal-only) 같은 공식화 entry 를 앞에
        mode_mis = np.zeros(len(idx)) if p_f is None else (A["end_mode"][idx] != want_mode).astype(float)
        both = np.round((pf_dist + ps_dist) / 0.02)                         # 시작+종점 TCP 거리 합, 2 cm 단위
        # 경계속도(=벨트 속도·상승 속도) 차이는 1 rad/s 단위로 거리 합보다 먼저 본다: 벨트 1.2 entry 가 5 cm 더 가깝다고
        # 뽑히면 다른 basin(J 6~10, t_f 0.85 s)으로 수렴한다 (2026-09-21 실측). 같은 속도 안에서만 거리로 고른다.
        # 같은 xy 타겟에 조준 z 가 다른 entry 가 공존한다 (bin z −0.06 / −0.04, 2026-09-22) — 같은 조준 z 를 먼저 (1 cm 단위)
        tz = np.round(np.abs(A["target_z"][idx] - float(p_target[2])) / 0.01)
        order = np.lexsort((pf_dist, both, np.round(v_dist, 0), tz, mode_mis))   # 마지막 키가 1차 정렬
        near = [db[int(idx[k])] for k in order[:3]]
        k0 = int(order[0])
        # 2026-09-22 시뮬 실측: 시작·종점이 2 cm 안인 DB entry(촘촘한 rest 격자)는 0.4~0.9 s 에 풀리는데 warm-prev(직전 해)는 t_ref·시작이
        # 달라 1.0~1.5 s 걸렸다 → 가까운 entry 가 있으면 DB 먼저, warm-prev 는 DB 가 불발일 때만. 가까운 entry 가 없으면 종전대로 warm-prev 먼저.
        db_close = bool(ps_dist[k0] <= 0.02 and pf_dist[k0] <= 0.02 and mode_mis[k0] == 0)
    _dl_rel = (lambda t: "" if _DEADLINE is None else f"{t - _DEADLINE:+.2f}s")
    _log(logger, f"  NLP 시간: warm DB 준비 {time.time() - _t_db0:.2f}s (마감 대비 {_dl_rel(time.time())})")
    if have_prev and not db_close:
        _try(warm=warm_init, tag="warm-prev")
    if near and best is None:
        for k, ent in enumerate(near):
            if k > 0 and _over_deadline():
                fails.append(f"warm-db#{k}: 시간 예산 초과로 생략")
                break
            # 첫 entry 가 풀리더라도 J 가 비정상(다른 basin: 착지오차 수 cm → W_ACC 항이 수백)이면 다음 후보도 풀어 더 나은 쪽을 쓴다
            t_k = time.time()
            ok_k = _try(warm=ent, tag=f"warm-db#{k}")
            if not ok_k:                                                  # 왜·얼마나 걸려 실패했는지 남긴다 (첫 던지기 느림 진단, 2026-09-22)
                _log(logger, f"  NLP: warm-db#{k} 실패 ({time.time() - t_k:.2f}s, p_start 거리 "
                             f"{np.linalg.norm(np.asarray(ent['p_start'])[:2] - p_start[:2]) * 1e3:.0f}mm): {fails[-1] if fails else '?'}")
            if ok_k and best is not None and best[0]["J"] <= WARM_J_OK:
                break
        if best is not None:
            _log(logger, f"  NLP: warm DB polish 적중 (target 거리 {d_tgt * 1e3:.0f}mm, "
                         f"p_start 거리 {np.linalg.norm(np.asarray(near[0]['p_start'])[:2] - p_start[:2]) * 1e3:.0f}mm, "
                         f"종점 거리 {pf_dist[order[0]] * 1e3:.0f}mm, belt {near[0].get('belt')}, {near[0].get('end_mode', 'linear')})")
    if have_prev and db_close and best is None and not _over_deadline():
        _try(warm=warm_init, tag="warm-prev")
    if best is None and _over_deadline():
        raise TimeoutError("NLP 시간 예산 초과 (warm 후보 실패):\n  " + "\n  ".join(fails[-6:]))

    # ---- ② cold multistart (warm 이 없거나 전부 불발일 때) ----
    if best is None:
        for k, init in enumerate(INIT_VARIANTS[:max(0, NLP_MAX_INIT)]):
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
    q_of, qd_of, qdd_of = throw_nlp._spline_eval(res["P"], res["t_f"])
    ts = np.arange(0.0, res["t_f"] + DT / 2, DT)
    Q = np.array([q_of(t) for t in ts])
    Qd = np.array([qd_of(t) for t in ts])
    Qdd = np.array([qdd_of(t) for t in ts])
    rt = res["release_time"]
    _log(logger, f"  NLP: {tag} 채택 (J={res['J']:.3f}, t_f={res['t_f']:.3f}s, "
                 f"t*={res['t_star']:.3f}s, 윈도우 ±{rt / 2 * 1e3:.0f}ms, "
                 f"실패후보 {len(fails)}개)"
                 + (f" — {res.get('partial_reason', '')}에 멈춘 반복값 (제약 위반 {res.get('partial_viol', 0):.1e}, "
                    f"clamp {np.rad2deg(res.get('partial_clamp', 0.0)):.2f}°, 착지 {res.get('partial_land_err', 0) * 100:.1f} cm)"
                    if res.get("partial") else ""))
    if res.get("partial"):
        tag = tag + "/partial"
    return dict(ts=ts, Q=Q, Qd=Qd, Qdd=Qdd,
                release_window=(res["t_star"] - rt / 2, res["t_star"] + rt / 2),
                q_of=q_of, qd_of=qd_of, qdd_of=qdd_of,
                q_start=np.asarray(res["q_start"], float), v0=v0, vf=vf, p_f=p_f,
                q_end=res.get("q_end"), raw=res,          # raw = solver 출력 (P, lam_g, u_pos …; DB 빌더용)
                info=dict(model="nlp", t_f=float(res["t_f"]),
                          t_star=float(res["t_star"]), J=float(res["J"]),
                          release_time=float(rt), tag=tag, n_fail=len(fails)))


def prebuild_polish_solvers(logger=None) -> float:
    """파라메트릭 polish solver 를 미리 조립한다 (자유 종점·고정/선형 이동 종점·정확한 Cartesian 종점 3벌, 각 ~4.5 s 의 CasADi expand).
    안 하면 첫 계획이 조립 시간만큼 늦어 물체를 놓친다. 리턴: 걸린 시간 [s]."""
    db = load_warm_db(logger)
    if not db:
        return 0.0
    e = db[0]
    t0 = time.time()
    q_s = np.asarray(e["q_start"], float)
    for pf, ec in ((None, None), (q_s, None),
                   (q_s, dict(p=fk_pos(q_s), v=np.array([0.0, -0.1, 0.0]), t_ref=1.0, j6=float(q_s[5])))):
        try:
            throw_nlp.solve_throw_nlp(None, e["target"], q_start=e["q_start"], warm_data=e,
                                      p_f=pf, p_f_rate=(None if pf is None else np.full(6, 1e-3)), end_cart=ec)
        except (RuntimeError, ValueError):                       # 조립만 되면 충분
            pass
    dt = time.time() - t0
    _log(logger, f"  NLP solver 사전 조립 완료 ({dt:.1f}s)")
    return dt


def geometry_summary() -> str:
    """벤더링된 THR 기하/공식화 파라미터 요약 (로그·스크립트 헤더용)."""
    return (f"tool={GP8_DIMS['tool']:.3f} (link6→TCP "
            f"{GP8_DIMS['d6'] + GP8_DIMS['tool']:.3f}m), "
            f"rt={throw_nlp.RELEASE_TIME:.3f}s, W1={throw_nlp.W1}, "
            f"W_ACC={throw_nlp.W_ACC:.0e}, qdd={throw_nlp.QDD_LIM[0] / throw_nlp.GP8_QD_MAX[0]:g}×qd, "
            f"cart_constraints={throw_nlp.CART_CONSTRAINTS}, "
            f"GRIP_OFF={throw_nlp.GRIP_OFF:.3f}, "
            f"W_omega={throw_nlp.W_OMEGA_LADDER}, p_start="
            + f"grasp+{START_LIFT:.3f}")

