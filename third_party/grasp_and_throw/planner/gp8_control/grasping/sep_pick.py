"""분리 방식 잡기 (2026-10-30 사용자): 하강·접촉·상승을 하나의 LP 명령으로 후계산한다. 끝 속력 (v_in, v_out) 은 던지기 NLP 가 정한다.

명령 z_c[0..N] (2 ms, 절대 높이):
  z_c[0] = Z_H (hover), z_c[1] − z_c[0] = −v_in·dt  (직전 던지기가 v_in 으로 내려오며 hover 를 지남; 그 전 이력 = 같은 속력 직선)
  z_c[N] = Z_H,        z_c[N] − z_c[N−1] = +v_out·dt (던지기 시작 경계; 뒤는 v_out 으로 M_RISE 샘플 더 오른 뒤 정지로 모델)
실제 z_r = h * z_c (컨트롤러 필터: 불감 D + 이동평균 W₁, W₂):  z_r ≥ 윗면 (전 구간), 어느 k_t 에 z_r = 윗면 (접촉, 흡착 유지 없음)
명령 한계: 앱과 같은 수직선 IK (픽 자세, J6 고정) 의 관절 속도·가속을 z 로 환산 (벨트 추종 몫 빼고, 여유 0.95)
수평: 명령 xy = 물체 (하강·상승 내내 벨트 속도).
solve(…, N=None): 가장 짧은 N (이분법).  N 을 주면 그 길이로 (던지기 시작 시각을 먼저 정한 경우 — 이음 자세가 정확히 맞도록).
"""
import os
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog
import scipy.sparse as _sp

TS = 0.002
Z_H = float(os.environ.get("GP8_TRAVEL_Z", "0.12"))
M_CONT = 160
M_RISE = 25
# 잡기 LP 여유 (한계 × LIM_SCALE 에 곱함). 11-04 사용자 "다 90프로로 맞춰" → 기본 1.0 (종전 속도 0.95·가속 0.90 = LP 선형화 잔차 여유).
# 11-04 05:01 lim90 DB 는 0.95·0.90 으로 만들어 그 DB 로 돌릴 때는 GP8_SEP_LP_VMARGIN=0.95 GP8_SEP_LP_AMARGIN=0.90. 속도 여유는 수직선 캐시 지문에 들어간다
MARGIN = float(os.environ.get("GP8_SEP_LP_VMARGIN", "1.0"))
AMARGIN = float(os.environ.get("GP8_SEP_LP_AMARGIN", "1.0"))
LIM_SCALE = float(os.environ.get("GP8_LIM_SCALE", "0.9"))      # 계획 한계 배율 (11-03, throw_nlp 와 같은 값) — 관절 속도·가속 × 이 값
LINE_Z_LO = -0.18   # 수직선 아래쪽 끝 [m] — 윗면 −0.02 물체의 잡기 명령은 −0.12 까지 내려간다 (11-02; 종전 −0.10 은 그 밑을 끝값으로 잘랐다)
VLIM_MARGIN = float(os.environ.get("GP8_SEP_VLIM_MARGIN", "0.03"))   # 끝 속력 상한 표 값에서 뺄 여유 [m/s] (표는 0.01 m/s 이분법)
TABLE = os.environ.get("GP8_SEP_TABLE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "learning", "models",
                                                     "sep_pick_table_b012_z.npz"))


def kernel(dt: float = TS) -> np.ndarray:
    """필터 임펄스 응답 (2 ms 격자): 불감 D 뒤 길이 W₁, W₂ 이동평균 — throw_nlp.FILTER (없으면 실측 box200_80)."""
    try:
        from gp8_control.throwing.nlp import throw_nlp
        D, Ws = throw_nlp.FILTER if throw_nlp.FILTER is not None else (0.04, (0.20, 0.08))
    except Exception:                                                        # noqa: BLE001
        D, Ws = 0.04, (0.20, 0.08)
    h = np.zeros(int(round(D / dt)) + 1); h[-1] = 1.0
    for w in Ws:
        n = max(1, int(round(w / dt)))
        h = np.convolve(h, np.full(n, 1.0 / n))
    return h


_H = None
_PP = {}


def _h():
    global _H
    if _H is None:
        _H = kernel()
    return _H


def _robot_j6(robot, j6):
    if robot is None:
        from gp8_control.robot.gp8 import GP8
        robot = GP8()
    if j6 is None:
        from gp8_control.config import Config
        j6 = float(Config().PICK_WRIST_J6)
    return robot, float(j6)


def line_data(r: float, belt: float = 0.12, robot=None, j6: Optional[float] = None, z_lo: float = None, z_hi: float = Z_H + 0.05):
    """반지름 r 수직선(픽 자세, J6 고정) 의 (zs, v_c, q', q'') — line_bounds 의 재료 (파일 캐시에 이 꼴로 저장)."""
    from gp8_control.grasping.pick_profile import PickProfiles
    from gp8_control.throwing.throw_planner import belt_joint_velocity
    robot, j6 = _robot_j6(robot, j6)
    z_lo = LINE_Z_LO if z_lo is None else z_lo
    key = (id(robot), round(j6, 6))
    if key not in _PP:
        _PP[key] = PickProfiles(robot, j6, Z_H + 0.01, 1.2, 20.0)
    zs_desc, Q_desc = _PP[key]._path(r, z_hi, z_lo)
    zs, Q = zs_desc[::-1], Q_desc[::-1]
    qp = np.gradient(Q, zs, axis=0); qpp = np.gradient(qp, zs, axis=0)
    VJ = MARGIN * LIM_SCALE * np.asarray(robot.velocity_limits, float)
    vb = np.array([np.maximum(np.abs(belt_joint_velocity(q, np.array([belt, 0.0, 0.0]))), np.abs(belt_joint_velocity(q, np.array([0.0, belt, 0.0]))))
                   for q in Q])
    v_c = np.min((VJ[None, :] - vb) / np.maximum(np.abs(qp), 1e-9), axis=1)
    return zs, v_c, qp, qpp


def line_bounds(r: float, belt: float = 0.12, robot=None, j6: Optional[float] = None, z_lo: float = None, z_hi: float = Z_H + 0.05, data=None):
    """반지름 r 수직선(픽 자세, J6 고정)의 명령 z 속도 상한 v_c(z), 가속 범위 a_c(z, ż) — pick_profile 과 같은 규칙.
    data: line_data 결과 (캐시) — 주면 IK 경로를 다시 풀지 않는다."""
    robot, j6 = _robot_j6(robot, j6)
    zs, v_c, qp, qpp = data if data is not None else line_data(r, belt, robot, j6, z_lo, z_hi)
    AJ = 20.0 * AMARGIN * LIM_SCALE * np.asarray(robot.velocity_limits, float)   # 가속 한계 × LP 여유 × 계획 한계 배율

    def a_bounds(z, zd):
        qpz = np.stack([np.interp(z, zs, qp[:, j]) for j in range(6)], -1); qppz = np.stack([np.interp(z, zs, qpp[:, j]) for j in range(6)], -1)
        c = qppz * zd[..., None] ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = (AJ - c) / qpz; b2 = (-AJ - c) / qpz
        lo = np.max(np.where(np.abs(qpz) > 1e-6, np.minimum(b1, b2), -np.inf), axis=-1)
        hi = np.min(np.where(np.abs(qpz) > 1e-6, np.maximum(b1, b2), np.inf), axis=-1)
        return lo, hi
    return (lambda z: np.interp(z, zs, v_c)), a_bounds


LINE_STEP = 0.001   # 잡기 LP 한계를 계산하는 수직선 간격 [m]


class LineCache:
    """물체의 실제 반경 (1 mm 로 반올림) 수직선의 잡기 LP 한계 (11-02 사용자: "LP 를 실제 반경의 수직선에서 계산해").
    종전엔 1 cm 칸 아래쪽 끝 반경의 선을 써서, 실제 반경에서 보간한 끝 속력 상한(표)보다 LP 자신의 hover 속도 한계가 최대 0.06 m/s
    낮았다 → Isaac head 실패 136 건 중 106 건이 LP 를 풀기도 전에 거부. 한 선을 푸는 데 0.11 s 라 head 에서 매번 풀지 않고, 범위 전체를
    1 mm 마다 한 번 풀어 설정 지문별 파일 (learning/models/sep_line_cache_<지문>.pkl) 에 둔다 — 다음 실행부터는 읽기만."""

    def __init__(self, belt: float, robot=None, j6: Optional[float] = None):
        self.robot, self.j6 = _robot_j6(robot, j6)
        self.belt = float(belt)
        # 캐시 이름(지문)의 벨트 속도는 설정값 GP8_BELT_SPEED (11-03 사용자): 실물 엔코더가 0.121 을 읽자 0.12 캐시를 못 찾고 세 계획
        # 프로세스가 수직선 421 개를 동시에 새로 풀어 예열이 90 s 를 넘겼다 (KeyError 'plan' 로 앱 종료). 1 % 차이는 선 한계에 무시할 만하다.
        self.belt_key = float(os.environ.get("GP8_BELT_SPEED") or belt)
        self.data = {}; self.bnd = {}

    def _fingerprint(self, r_lo: float, r_hi: float) -> str:
        import hashlib
        parts = [np.asarray(self.robot.velocity_limits, float), np.array([self.belt_key, self.j6, Z_H, MARGIN, LINE_STEP, LINE_Z_LO, r_lo, r_hi, LIM_SCALE])]
        return hashlib.sha1(b"".join(np.round(p, 9).tobytes() for p in parts)).hexdigest()[:12]

    def prewarm(self, r_lo: float, r_hi: float, cache_dir: Optional[str] = None) -> float:
        import pickle
        import time
        t0 = time.time()
        d = cache_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "learning", "models")
        path = os.path.join(d, f"sep_line_cache_{self._fingerprint(r_lo, r_hi)}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.data.update(pickle.load(f))
                return time.time() - t0
            except Exception:                                          # noqa: BLE001 — 깨졌으면 다시 만든다
                pass
        for r in np.round(np.arange(r_lo, r_hi + 1e-9, LINE_STEP), 3):
            self.data[float(r)] = line_data(float(r), self.belt, self.robot, self.j6)
        try:
            tmp = path + f".tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump(dict(self.data), f)
            os.replace(tmp, path)
        except OSError:
            pass
        return time.time() - t0

    def get(self, r: float):
        """(v_c, a_c) — 반경 r 을 1 mm 로 반올림한 수직선. 캐시 범위 밖이면 그 자리에서 푼다 (0.11 s)."""
        key = float(np.round(round(float(r) / LINE_STEP) * LINE_STEP, 3))
        if key not in self.bnd:
            if key not in self.data:
                self.data[key] = line_data(key, self.belt, self.robot, self.j6)
            self.bnd[key] = line_bounds(key, self.belt, self.robot, self.j6, data=self.data[key])
        return self.bnd[key]


def realized_affine(N: int, v_in: float, M: int = M_CONT, z0: float = Z_H) -> Tuple[np.ndarray, np.ndarray]:
    """z_r[0..N+M] = A z_c + b  (b = k<0 이력: 시작 높이 z0 위에서 v_in 으로 내려오던 직선)."""
    h = _h(); pad = len(h); n = N + 1; L = n + M
    hist = z0 + v_in * TS * np.arange(pad, 0, -1)

    def run(zc, hz):
        s1 = zc[N] - zc[N - 1]
        cont = zc[N] + s1 * np.minimum(np.arange(1, M + 1), M_RISE)
        return np.convolve(np.concatenate([hz, zc, cont]), h)[pad:pad + L]
    A = np.zeros((L, n))
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0; A[:, i] = run(e, np.zeros(pad))
    return A, run(np.zeros(n), hist)


def _solve_fixed(N, k_t, v_c, a_c, zp, z_o, v_in, v_out, Ab, z0=Z_H):
    n = N + 1; A_r, b_r = Ab
    vk = v_c(0.5 * (zp[:-1] + zp[1:])); zd = np.gradient(zp, TS)[1:-1]; a_lo, a_hi = a_c(zp[1:-1], zd)
    a_lo = np.maximum(a_lo, -400.0); a_hi = np.minimum(a_hi, 400.0)
    if np.any(a_hi < a_lo):
        return None
    D1 = np.zeros((N, n)); D1[np.arange(N), np.arange(N)] = -1; D1[np.arange(N), np.arange(N) + 1] = 1
    D2 = np.zeros((N - 1, n)); D2[np.arange(N - 1), np.arange(N - 1)] = 1; D2[np.arange(N - 1), np.arange(N - 1) + 1] = -2
    D2[np.arange(N - 1), np.arange(N - 1) + 2] = 1
    A_ub = np.vstack([D1, -D1, D2, -D2, -A_r]); b_ub = np.concatenate([vk * TS, vk * TS, a_hi * TS * TS, -a_lo * TS * TS, b_r - z_o])
    A_eq = np.zeros((5, n)); b_eq = np.array([z0, z0 - v_in * TS, Z_H, Z_H - v_out * TS, z_o - b_r[k_t]])
    A_eq[0, 0] = 1; A_eq[1, 1] = 1; A_eq[2, N] = 1; A_eq[3, N - 1] = 1; A_eq[4] = A_r[k_t]
    # 명령 높이 범위 없음 (11-02 사용자: −0.20 ≤ z_c ≤ Z_H 를 뺐다 — 높이 조건은 실제 z ≥ 윗면·접촉 등식만)
    r = linprog(-0.01 * np.ones(n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)] * n, method="highs")
    return r.x if r.status == 0 else None


def _feasible(N, v_c, a_c, z_o, v_in, v_out, iters=3, z0=Z_H, near: bool = False):
    Ab = realized_affine(N, v_in, z0=z0)
    kf = int(0.45 * N)
    zp0 = np.concatenate([np.linspace(z0, z_o - 0.04, kf + 1), np.linspace(z_o - 0.04, Z_H, N - kf + 1)[1:]])
    k0 = min(int(1.25 * N), N + M_CONT - 6)                                  # 실제 접촉은 대개 1.31~1.46 N (DB 5318 건) — 거기부터 훑는다
    ks = list(range(k0, N + M_CONT - 5, 3)) + list(range(10, k0, 3))
    if near:
        # 앱 (10-30): 접촉 시점 후보를 k0 에서 가까운 순으로, ±30 샘플 안만 — 종전 순서는 접촉이 k0 바로 아래면 위쪽 끝까지 훑고 내려와
        # LP 92 번 (0.9 s) 을 써서 head 가 마감을 넘겼다. 가까운 순이면 7~13 번 (0.11~0.16 s).
        ks = sorted((k for k in ks if abs(k - k0) <= 30), key=lambda k: (abs(k - k0), k))
    for k_t in ks:
        zp = zp0; z = None
        for _ in range(iters):
            z = _solve_fixed(N, k_t, v_c, a_c, zp, z_o, v_in, v_out, Ab, z0)
            if z is None:
                break
            zp = z
        if z is not None:
            return k_t, z
    return None


def solve(r: float, v_in: float, v_out: float, z_o: float = 0.02, N: Optional[int] = None, N_lo: int = 20, N_hi: int = 260, bounds=None,
          z0: float = Z_H, near: bool = False):
    """(N, k_t, z_c) 또는 None. N·TS = hover(하강 시작) → z_top 통과(상승 끝) 시간. z0: 시작 명령 높이 (앞 던지기의 마지막 샘플 —
    던지기는 t_f 를 2 ms 격자로 올려 끝나므로 hover 보다 v_in·(≤2 ms) 낮을 수 있다)."""
    v_c, a_c = bounds if bounds is not None else line_bounds(r)
    if v_in > v_c(Z_H) + 1e-9 or v_out > v_c(Z_H) + 1e-9:
        return None
    if N is not None:
        f = _feasible(int(N), v_c, a_c, z_o, v_in, v_out, z0=z0, near=near)
        return None if f is None else (int(N),) + f
    if _feasible(N_hi, v_c, a_c, z_o, v_in, v_out, z0=z0) is None:
        return None
    lo, hi = N_lo, N_hi; best = None
    while hi - lo > 1:
        mid = (lo + hi) // 2; f = _feasible(mid, v_c, a_c, z_o, v_in, v_out, z0=z0)
        if f is None:
            lo = mid
        else:
            hi = mid; best = (mid,) + f
    return best if best is not None else (hi,) + _feasible(hi, v_c, a_c, z_o, v_in, v_out, z0=z0)


class Table:
    """잡기 시간 맞춤 T ≈ c0 + c1 v_in + c2 v_in² + c3 v_out + c4 v_out² 와 끝 속력 상한 (sep_table).
    반경 × 물체 높이 표 (11-02 사용자: 물체 윗면 −0.02~0.05 — 높을수록 감속 거리가 짧아 끝 속력 상한이 낮다): 격자점 (r, z) 마다
    잡기 LP 로 구한 값을 r·z 쌍선형 보간 (계수도 보간 = T 를 보간). 높이를 안 주면 0.02, 격자 밖 높이는 끝 높이로.
    z_grid 가 없는 예전 표 (높이 0.02 하나, 1 cm 칸 아래쪽 끝) 도 읽는다."""
    Z_DEF = 0.02

    def __init__(self, path: str = TABLE):
        z = np.load(path, allow_pickle=True)
        self.z_grid = np.asarray(z["z_grid"], float) if "z_grid" in z else None
        if self.z_grid is not None:
            self.r_grid = np.asarray(z["r_grid"], float)
            self.vin_grid, self.vout_grid = np.asarray(z["vin_grid"], float), np.asarray(z["vout_grid"], float)   # (반경, 높이)
            self.vc_grid, self.c_grid = np.asarray(z["vc_grid"], float), np.asarray(z["c_grid"], float)           # c: (반경, 높이, 5)
            # head 길이 추정용 잡기 시간 표본 (반경, 높이, v_in 3, v_out 3): v_in ∈ {0, ½, 1}·vi_s, v_out ∈ {0.3, 중간, vo_s}. 높은 물체는 v_in·v_out
            # 이 둘 다 크면 따로 더한 맞춤식보다 0.1 s 길다 (r 0.58·윗면 0.04: 맞춤 0.16 s, 실제 0.27 s) → t_pick 은 표본을 2차 보간
            self.t_grid = np.asarray(z["t_grid"], float) if "t_grid" in z else None
            self.vi_s = np.asarray(z["vi_s"], float) if "vi_s" in z else None
            self.vo_s = np.asarray(z["vo_s"], float) if "vo_s" in z else None
            self.rows = {int(round(r * 100)): None for r in self.r_grid}       # 반경 범위 (cm) — 앱 예열이 쓴다
            return
        self.rows = {int(k): (float(v), np.asarray(c, float)) for k, v, c in zip(z["bins"], z["vmax"], z["c"])}
        # 잡기 LP 가 실제로 풀리는 끝 속력 상한 (sep_vlim, 10-30 사용자): 반경 r [cm 격자점] 에서 vin_max(r), vout_max(r) — 감속 거리(관절
        # 가속 한계), 필터 통과 실제 z ≥ 윗면, 잡기 시간 ≤ 0.52 s 까지 본 값. 없으면 None (종전 규칙 vmax − 0.10).
        self.r_grid = np.asarray(z["r_grid"], float) if "r_grid" in z else None
        self.vin_grid = np.asarray(z["vin_grid"], float) if "vin_grid" in z else None
        self.vout_grid = np.asarray(z["vout_grid"], float) if "vout_grid" in z else None

    def _rz(self, grid, r: float, z: Optional[float]):
        """(값, d/dr) — 반경 × 높이 격자 쌍선형 보간. 반경이 격자 밖이면 None."""
        rg, zg = self.r_grid, self.z_grid
        if not (rg[0] - 1e-9 <= r <= rg[-1] + 1e-9):
            return None
        i = int(np.clip(np.searchsorted(rg, r) - 1, 0, len(rg) - 2)); u = (r - rg[i]) / (rg[i + 1] - rg[i])
        zz = float(np.clip(self.Z_DEF if z is None else z, zg[0], zg[-1]))
        j = int(np.clip(np.searchsorted(zg, zz) - 1, 0, len(zg) - 2)); w = (zz - zg[j]) / (zg[j + 1] - zg[j])
        g0 = (1 - w) * grid[i, j] + w * grid[i, j + 1]; g1 = (1 - w) * grid[i + 1, j] + w * grid[i + 1, j + 1]
        return (1 - u) * g0 + u * g1, (g1 - g0) / (rg[i + 1] - rg[i])

    def _lin(self, grid, r: float, margin: float):
        """(값, 기울기) — 격자 선형 보간 (NLP 가 종점 반경의 1차식으로 쓴다)."""
        rg = self.r_grid
        if rg is None or grid is None or not (rg[0] <= r <= rg[-1]):
            return None
        i = int(np.clip(np.searchsorted(rg, r) - 1, 0, len(rg) - 2))
        s = (grid[i + 1] - grid[i]) / (rg[i + 1] - rg[i])
        return float(grid[i] + s * (r - rg[i]) - margin), float(s)

    def _lim(self, grid, r: float, z: Optional[float], margin: float):
        if self.z_grid is not None:
            v = self._rz(grid, r, z)
            return None if v is None else (float(v[0] - margin), float(v[1]))
        v = self._lin(grid, r, margin)
        if v is None:
            vm = self.vmax(r)
            return None if vm is None else (vm - 0.10, 0.0)
        return v

    def vin_lim(self, r: float, z: Optional[float] = None, margin: float = VLIM_MARGIN):
        """반경 r·윗면 높이 z 물체를 잡으러 hover 를 지날 수 있는 최대 하강 속력 (값, d/dr). 표에 없으면 (vmax − 0.10, 0)."""
        return self._lim(self.vin_grid, r, z, margin)

    def vout_lim(self, r: float, z: Optional[float] = None, margin: float = VLIM_MARGIN):
        return self._lim(self.vout_grid, r, z, margin)

    def row(self, r: float, z: Optional[float] = None):
        """(hover 명령 속도 한계, 잡기 시간 계수 c[5]) 또는 None."""
        if self.z_grid is not None:
            vc = self._rz(self.vc_grid, r, z)
            if vc is None:
                return None
            c = np.array([self._rz(self.c_grid[:, :, m], r, z)[0] for m in range(5)])
            return float(vc[0]), c
        k = int(np.floor(r / 0.01 + 1e-9))
        return self.rows.get(k)

    def vmax(self, r: float, z: Optional[float] = None) -> Optional[float]:
        rw = self.row(r, z)
        return None if rw is None else rw[0]

    @staticmethod
    def _lag3(x0: float, x2: float, x: float) -> np.ndarray:
        """세 점 {x0, (x0+x2)/2, x2} 2차 라그랑주 가중 (x 는 그 범위로 자름)."""
        if x2 - x0 < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        t = (float(np.clip(x, x0, x2)) - x0) / (x2 - x0)
        return np.array([2 * (t - 0.5) * (t - 1), -4 * t * (t - 1), 2 * t * (t - 0.5)])

    def t_pick(self, r: float, v_in: float, v_out: float, z: Optional[float] = None) -> Optional[float]:
        if self.z_grid is not None and self.t_grid is not None:
            rg, zg = self.r_grid, self.z_grid
            if not (rg[0] - 1e-9 <= r <= rg[-1] + 1e-9):
                return None
            i = int(np.clip(np.searchsorted(rg, r) - 1, 0, len(rg) - 2)); u = (r - rg[i]) / (rg[i + 1] - rg[i])
            zz = float(np.clip(self.Z_DEF if z is None else z, zg[0], zg[-1]))
            j = int(np.clip(np.searchsorted(zg, zz) - 1, 0, len(zg) - 2)); w = (zz - zg[j]) / (zg[j + 1] - zg[j])
            out = 0.0
            for ii, jj, wt in ((i, j, (1 - u) * (1 - w)), (i, j + 1, (1 - u) * w), (i + 1, j, u * (1 - w)), (i + 1, j + 1, u * w)):
                out += wt * float(self._lag3(0.0, self.vi_s[ii, jj], v_in) @ self.t_grid[ii, jj] @ self._lag3(0.3, self.vo_s[ii, jj], v_out))
            return out
        rw = self.row(r, z)
        if rw is None:
            return None
        c = rw[1]
        return float(c[0] + c[1] * v_in + c[2] * v_in ** 2 + c[3] * v_out + c[4] * v_out ** 2)


# -- 명령 보정 (2026-10-30 사용자: "상승 하강 시에 명령을 잘 조절해서 최대한 잡기 오차 없게, 의도한 실제 궤적이 나오게") ----------------
# 위 LP 는 z 만, 그것도 "앞 이력 = 같은 속력 수직선" 이라는 가정으로 실제 z 를 맞춘다. 실제로는 필터가 관절 공간에서 평균하고, 접촉 순간의
# 실제 자세는 앞 던지기의 끝(수평으로 휘어 들어오는 명령)과 뒤 던지기의 시작까지 섞인다 → 실제 툴이 물체에서 수 mm 비켜 닿는다.
# correct(): 앞뒤 명령(이력·던지기)을 그대로 두고 잡기 명령의 Cartesian 위치만 δp (양끝 값·기울기·곡률 0 인 3차 B-스플라인) 로 옮겨,
# 필터를 지난 실제 TCP 가 [물체 xy (스트림 선행 lead 만큼 이른 시각), LP 가 설계한 실제 z] 를 따르게 하는 QP 를 푼다.
#   선형화: Δp_r[k] = J_p(q_r[k]) Σ_j h[k−j] M[j] δp[j],  M[j] = ∂IK/∂p (툴 수직·J6 고정 픽 IK 의 위치 민감도)
#   목적  : Σ_k w_k ‖p*[k] − p_r[k] − Δp_r[k]‖² + λ‖c‖²   (w: 하강·상승 구간, 윗면 가까울수록 큼)
#   제약  : 명령 관절 속도·가속 ≤ 한계(앱 clamp 와 같은 20·q̇_max), 실제 z ≥ 윗면 (하강·상승 내내)
# 풀고 나면 IK 를 정확히 다시 풀어 (선형화 잔차) 한 번 더. 나빠지면 원래 명령을 돌려준다.
CORR_KNOT_S = float(os.environ.get("GP8_SEP_CORR_KNOT", "0.016"))
CORR_LAM = float(os.environ.get("GP8_SEP_CORR_LAM", "1e-3"))
CORR_AMARGIN = float(os.environ.get("GP8_SEP_CORR_AMARGIN", "0.90"))   # QP 의 가속 한계 비 (선형화 잔차 몫), 최종 검사는 1.0
CORR_MSTEP = int(os.environ.get("GP8_SEP_CORR_MSTEP", "2"))            # IK 민감도 M 을 몇 샘플마다 (사이는 보간)
# 반복 1 회 (10-30 시험 4건): 접촉 xy 5.1→1.1, 5.6→1.6, 4.8→2.1, 6.4→3.9 mm, 0.2~0.5 s. 3 회는 가중 목적(접촉 밖 하강·상승 포함)만 더 줄이고
# 접촉 오차는 2.4~4.5 mm 로 오히려 커졌고 0.5~0.6 s.
# 10-30 사용자: 보정 QP 없앰 — 수직선 명령 그대로도 필터 모델 접촉 xy 오차가 10 mm 기준 안 (픽 단독 r 0.20~0.60 ×
# 끝 속력 0~한계 81 건 0.9~7.9 mm, 중앙 4.6, 거의 반경 바깥쪽; 실제 z 는 윗면 아래 ≤ 1.0 mm. 앞뒤 던지기 포함 4 건 4.8~6.4 mm).
# 기본 0 회 = 보정 없이 접촉 오차만 재서 info 에 남기고 명령은 그대로 (1 이면 종전 보정).
CORR_ITERS = int(os.environ.get("GP8_SEP_CORR_ITERS", "0"))


def lead_s() -> float:
    """계획 시각 τ 의 실제(필터 통과) 자세가 만나야 할 물체 = 계획 물체 궤적의 τ − lead 시각.
    lead = 스트림 선행(GP8_SERVO_LAG_S: 계획 τ 의 실제 자세가 벽시계 τ − 선행에 나타남) + 물체 선행(GP8_PICK_LEAD_S: 계획의 물체 위치를 실제보다
    이만큼 하류로 둠). 10-30 사용자 ("0.18 s 먼저 보내지 말고 평균 필터를 그대로"): 스트림 선행 0, 물체 선행 = 필터 평균 지연 — 명령이
    물체를 그만큼 앞서 따라가는 기본 경로에서 출발해, 보정이 필터를 지난 실제 자세를 실제 시각의 물체에 맞춘다."""
    return float(os.environ.get("GP8_SERVO_LAG_S", "0.18")) + float(os.environ.get("GP8_PICK_LEAD_S", "0.0"))


CORR_STOP = None                                                            # 보정을 끝내야 하는 벽시계 시각 (앱 꼬리: 잡기 시작 직전)


def set_corr_stop(t_wall) -> None:
    global CORR_STOP
    CORR_STOP = None if t_wall is None else float(t_wall)


def _basis(n: int, dt: float, knot: float) -> np.ndarray:
    """(n, nb) 균일 3차 B-스플라인 — 지지가 [2, n−3] 샘플 안 (양끝 두 샘플은 0: 이음 자세·속도·가속 그대로)."""
    a, b = 2 * dt, (n - 3) * dt
    span = b - a
    nb = max(2, int(np.floor(span / knot)) - 3)
    d = span / (nb + 3)
    c = a + 2 * d + d * np.arange(nb)
    t = np.arange(n) * dt
    x = np.abs((t[:, None] - c[None, :]) / d)
    B = np.where(x < 1, 2 / 3 - x ** 2 + x ** 3 / 2, np.where(x < 2, (2 - x) ** 3 / 6, 0.0))
    return B


def _qp(H, g, A, lba, uba):
    """작은 밀집 QP — qpOASES (proxqp 의 4 배 빠름). qpOASES 는 printLevel none 이어도 저작권 문구를 표준출력에 찍어 그동안만 막는다."""
    import casadi as ca
    n = H.shape[0]
    fd = os.dup(1); nul = os.open(os.devnull, os.O_WRONLY); os.dup2(nul, 1)
    try:
        S = ca.conic("S", "qpoases", {"h": ca.Sparsity.dense(n, n), "a": ca.Sparsity.dense(A.shape[0], n)}, {"printLevel": "none", "error_on_fail": False})
        r = S(h=H, g=g, a=A, lba=np.maximum(lba, -1e20), uba=np.minimum(uba, 1e20))
        ok = S.stats().get("success", False)
    finally:
        os.dup2(fd, 1); os.close(fd); os.close(nul)
    return np.asarray(r["x"]).ravel() if ok else None


def _realized(U: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.stack([np.convolve(U[:, j], h)[:len(U)] for j in range(U.shape[1])], -1)


def _ik_all(ik, P, q0):
    Qo = np.empty((len(P), 6)); q_prev = q0
    for j in range(len(P)):
        q = ik(P[j], q_prev)
        if q is None:
            return None
        Qo[j] = q; q_prev = q
    return Qo


def _lim_ratio(Qc, v_lim, dt):
    """명령의 max(속도/한계, 가속/(20·속도한계)) — 잡기 구간 안 (양끝 이음은 바꾸지 않으므로 제외 안 해도 같다)."""
    vl = np.asarray(v_lim, float)
    return float(max((np.abs(np.diff(Qc, axis=0)) / dt / vl).max(), (np.abs(np.diff(Qc, 2, axis=0)) / dt ** 2 / (20 * vl)).max()))


def correct(P_cmd, Q, ik, tcp_batch, hist, fut, xy_tgt, z_lp, z_top, v_lim, iters: int = CORR_ITERS, dt: float = TS):
    """잡기 명령 보정. P_cmd (N+1,3) 명령 TCP, Q (N+1,6) 그 IK, ik(p, q_init) → q, tcp_batch(Q) → (·,3).
    hist (≥pad−1, 6): Q[0] 앞 명령 (없으면 Q[0] 정지), fut (·,6): Q[N] 뒤 명령 (던지기·정지), xy_tgt (K,2): 실제 시각 k (잡기 기준
    샘플) 의 목표 xy, z_lp (K,): LP 가 설계한 실제 z (NaN = 목표 없음), v_lim (6,) [rad/s].
    리턴 (P_new, Q_new, info) — 나빠지거나 못 풀면 (P_cmd, Q, info(ok=False))."""
    h = _h(); pad = len(h); n = len(Q); N = n - 1
    hist = np.asarray(hist, float) if hist is not None and len(hist) else Q[:1]
    if len(hist) < pad - 1:
        hist = np.vstack([np.repeat(hist[:1], pad - 1 - len(hist), 0), hist])
    hist = hist[-(pad - 1):]
    fut = np.asarray(fut, float) if fut is not None and len(fut) else Q[-1:]
    if n + len(fut) < len(z_lp):                                            # 계획이 끝나면 팔은 마지막 자세에 머문다 (head 의 잡기 + 정지)
        fut = np.vstack([fut, np.repeat(fut[-1:], len(z_lp) - n - len(fut), 0)])
    K = min(len(xy_tgt), len(z_lp), n + len(fut))
    z_lp = np.asarray(z_lp, float)[:K]; xy_tgt = np.asarray(xy_tgt, float)[:K]
    win = np.isfinite(z_lp) & (z_lp < Z_H - 0.002)                         # LP 가 설계한 실제 하강·상승 (hover 아래)
    dz = np.where(win, z_lp - z_top, np.inf)
    w = np.where(win, 1.0 + 50.0 * np.exp(-np.maximum(dz, 0.0) / 0.01), 0.0)
    B = _basis(n, dt, CORR_KNOT_S); nb = B.shape[1]
    VL = 0.95 * np.asarray(v_lim, float); AL = 20.0 * CORR_AMARGIN * np.asarray(v_lim, float)
    Toe = np.zeros((K, n))                                                  # Toe[k, j] = h[k − j]
    for j in range(n):
        m = np.arange(K) - j
        ok = (m >= 0) & (m < pad)
        Toe[ok, j] = h[m[ok]]

    def evaluate(Qc):
        U = np.vstack([hist, Qc, fut])
        qr = _realized(U, h)[pad - 1:pad - 1 + K]
        return qr, tcp_batch(qr)

    def score(pr):
        e_xy = np.linalg.norm(pr[:, :2] - xy_tgt, axis=1); e_z = pr[:, 2] - z_lp
        k_c = int(np.nanargmin(np.where(win, pr[:, 2], np.inf)))
        return dict(J=float(np.sum(w * (e_xy ** 2 + np.where(win, e_z, 0.0) ** 2))), e_contact=float(e_xy[k_c]),
                    z_min=float(np.min(pr[win, 2])) if win.any() else np.nan, k_c=k_c)

    qr0, pr0 = evaluate(Q); s0 = score(pr0)
    info = dict(ok=False, before=s0, lim_before=_lim_ratio(Q, v_lim, dt))
    P_cur, Q_cur, s_cur = np.asarray(P_cmd, float).copy(), np.asarray(Q, float).copy(), s0
    for _ in range(iters):
        qr, pr = evaluate(Q_cur)
        eps = 1e-5                                                          # 실제 TCP 위치 자코비안 (유한차분, 한 번에)
        Jp = np.empty((K, 3, 6))
        for i in range(6):
            dq = np.zeros(6); dq[i] = eps
            Jp[:, :, i] = (tcp_batch(qr + dq) - tcp_batch(qr - dq)) / (2 * eps)
        idx = np.unique(np.r_[np.arange(0, n, CORR_MSTEP), n - 1])          # IK 위치 민감도 M = ∂q/∂p (CORR_MSTEP 샘플마다 + 보간)
        Ms = np.empty((len(idx), 6, 3)); e3 = 1e-4
        for a, j in enumerate(idx):
            for ax in range(3):
                d = np.zeros(3); d[ax] = e3
                qa, qb = ik(P_cur[j] + d, Q_cur[j]), ik(P_cur[j] - d, Q_cur[j])
                if qa is None or qb is None:
                    return P_cmd, Q, info
                Ms[a, :, ax] = (np.asarray(qa) - np.asarray(qb)) / (2 * e3)
        M = np.stack([np.stack([np.interp(np.arange(n), idx, Ms[:, r, c]) for c in range(3)], -1) for r in range(6)], 1)   # (n,6,3)
        L = np.einsum("jrc,jb->jrcb", M, B).reshape(n, 6, 3 * nb)           # Q_new ≈ Q + L x  (x = [c_x, c_y, c_z] 계수, 축 우선)
        Lr = np.einsum("kj,jrx->krx", Toe, L)                               # 실제 관절 변화
        G = np.einsum("kar,krx->kax", Jp, Lr)                               # 실제 TCP 변화 (K,3,3nb)
        e = np.concatenate([xy_tgt - pr[:, :2], np.where(win, z_lp - pr[:, 2], 0.0)[:, None]], 1)
        sw = np.sqrt(w)[:, None, None]
        Gw = (G * sw).reshape(-1, 3 * nb); ew = (e * sw[:, :, 0]).ravel()
        Hq = 2 * (Gw.T @ Gw) + 2 * CORR_LAM * np.eye(3 * nb); gq = -2 * Gw.T @ ew
        D1 = (L[1:] - L[:-1]) / dt; v0 = (Q_cur[1:] - Q_cur[:-1]) / dt
        D2 = (L[2:] - 2 * L[1:-1] + L[:-2]) / dt ** 2; a0 = (Q_cur[2:] - 2 * Q_cur[1:-1] + Q_cur[:-2]) / dt ** 2
        A = [D1.reshape(-1, 3 * nb), D2.reshape(-1, 3 * nb)]
        lb = [(-VL[None] - v0).ravel(), (-AL[None] - a0).ravel()]
        ub = [(VL[None] - v0).ravel(), (AL[None] - a0).ravel()]
        A.append(G[win, 2, :]); lb.append(z_top - 3e-4 - pr[win, 2]); ub.append(np.full(int(win.sum()), np.inf))
        k_t = int(np.nanargmin(np.where(win, z_lp, np.inf)))                # 닿기: LP 의 접촉 순간 실제 z ≤ 윗면 + 0.3 mm (xy 와 맞바꾸지 않게)
        A.append(G[k_t:k_t + 1, 2, :]); lb.append(np.array([-np.inf])); ub.append(np.array([z_top + 3e-4 - pr[k_t, 2]]))
        A = np.vstack(A); lb = np.concatenate(lb); ub = np.concatenate(ub)
        lb = np.minimum(lb, 0.0); ub = np.maximum(ub, 0.0)                  # 이미 한계 밖인 성분(LP 잔차)은 더 나빠지지만 않게
        x = _qp(Hq, gq, A, lb, ub)
        if x is None:
            info["fail"] = "QP"
            break
        dP = np.stack([B @ x[i * nb:(i + 1) * nb] for i in range(3)], -1)
        P_new = P_cur + dP
        Q_new = None
        for _ls in range(4):                                                # 정확한 IK 로 한계를 넘으면 걸음을 줄인다
            Q_new = _ik_all(ik, P_new, Q_cur[0])
            if Q_new is None:
                info["fail"] = "IK"
                return (P_cur, Q_cur, info) if info["ok"] else (P_cmd, Q, info)
            Q_new[0], Q_new[-1] = Q_cur[0], Q_cur[-1]
            if _lim_ratio(Q_new, v_lim, dt) <= max(1.0, _lim_ratio(Q_cur, v_lim, dt)):
                break
            P_new = P_cur + 0.5 * (P_new - P_cur); Q_new = None
        if Q_new is None:
            info["fail"] = "limits"
            break
        _, pr_n = evaluate(Q_new); s_n = score(pr_n)
        if s_n["J"] >= s_cur["J"]:
            break
        P_cur, Q_cur, s_cur = P_new, Q_new, s_n; info.update(ok=True, after=s_n, dP_max=float(np.abs(P_cur - P_cmd).max()))
    if info["ok"]:
        vq = np.abs(np.diff(Q_cur, axis=0)) / dt / np.asarray(v_lim, float); aq = np.abs(np.diff(Q_cur, 2, axis=0)) / dt ** 2 / (20 * np.asarray(v_lim, float))
        info.update(v_ratio=float(vq.max()), a_ratio=float(aq.max()))
        return P_cur, Q_cur, info
    return P_cmd, Q, info


# -- 잡기 v3 (2026-11-04 사용자, 논문 version2 식 7): 변수 = 6축 명령 + 끝 상승 속력 v_out, 조건은 전부 필터를 지난 실제 궤적에 --------------
# 실제 관절 q[k] = Σ ψ_m u[k−m] (이력 = 앞 계획의 실제 명령) — 명령의 정확한 1차식. 11-04 사용자 "던지기가 처음 0.05 s 동안 계속 올라간다 이건
# 휴리스틱": 끝 뒤 이어짐 가정 없음 — 실제 샘플은 잡기 명령까지로 정해지는 k ≤ N + N_D (불감 20 샘플) 만 쓰고, 그 안에서 실제 접촉 (최저점 뒤 상승).
# 실제 TCP 위치만 선형화 (FK 와 그 자코비안만, IK 없음). 실제 자세: 잡기 자세가 관절의 1차식 (J4 0, J5 = J2 − J3 − π/2, J6 고정) 이라
# 실제 관절의 그 식 어긋남도 명령의 1차식 — 정확히 LP 에. 목적 = 최단 길이 (호출부 사다리), 각 길이는 최대 위반 최소화 LP 를 반복.
V3 = os.environ.get("GP8_SEP_V3", "0") not in ("0", "", "false")
V3_HC = float(os.environ.get("GP8_SEP_V3_HC", "0.005"))           # 접촉부 = 실제 z ≤ 윗면 + h_c (기록용)
V3_SBAR = float(os.environ.get("GP8_SEP_V3_SBAR", "0.005"))       # 잡기 구간 전체 실제 xy 허용 s̄ [m] (11-04 사용자: 5 mm)
V3_ALPHA = float(os.environ.get("GP8_SEP_V3_ALPHA", "0.01745"))   # 실제 자세 허용 α [rad] (11-04 사용자: 1°)
V3_KNOT = int(os.environ.get("GP8_SEP_V3_KNOT", "4"))             # 명령 변화 B-스플라인 간격 [샘플]
V3_ITERS = int(os.environ.get("GP8_SEP_V3_ITERS", "6"))
V3_TR = float(os.environ.get("GP8_SEP_V3_TR", "0.03"))            # 반복당 계수 변화 상한 [rad]
V3_TOL = 2e-4                                                     # 최대 위반이 이 아래면 그 길이는 가능


def _v3_basis(n_var: int, knot: int) -> np.ndarray:
    """(n_var, nb) 균일 3차 B-스플라인 — 샘플 1..n_var (샘플 0 = 이음, 0 으로 고정)."""
    t = np.arange(1, n_var + 1, dtype=float)
    centers = np.arange(knot, n_var + 2 * knot, knot, dtype=float)
    x = np.abs((t[:, None] - centers[None, :]) / knot)
    B = np.where(x < 1, 2 / 3 - x ** 2 + x ** 3 / 2, np.where(x < 2, (2 - x) ** 3 / 6, 0.0))
    return B[:, B.sum(0) > 1e-9]


def solve_v3(hist, Q0, obj_xy, z_o: float, z_h: float, v_out0: float, v_lim, j6: float, v_out_free: bool = True,
             dt: float = TS, iters: int = None) -> dict:
    """hist (≥K−1, 6): 잡기 앞 명령 (마지막 = Q0[0] 바로 앞), Q0 (N+1, 6): 초기 명령 (Q0[0] = 이음, 고정),
    obj_xy (N+1+M, 2): 실제 시각 k 의 물체 xy, z_o: 윗면, z_h: 끝 명령 높이 (hover), v_out0: 초기 상승 속력, v_lim (6,) 관절 속도 한계.
    리턴 dict(ok, Q, v_out, k_t, e_contact, e_c, e_w, e_rot, z_gap, viol, vr, ar, it, why)."""
    from gp8_control.grasping.workspace_gate import tcp_batch
    iters = V3_ITERS if iters is None else iters
    h = _h(); pad = len(h); Q = np.asarray(Q0, float).copy(); n = len(Q); N = n - 1
    M = int(np.argmax(np.abs(h) > 1e-15)); K = n + M                            # M = 불감 N_D: 실제 k ≤ N + N_D 는 잡기 명령까지로 정해진다
    hist = np.asarray(hist, float)
    if len(hist) < pad - 1:
        hist = np.vstack([np.repeat(hist[:1], pad - 1 - len(hist), 0), hist])
    hist = hist[-(pad - 1):]
    obj_xy = np.asarray(obj_xy, float)[:K]
    VJ = LIM_SCALE * np.asarray(v_lim, float); AJ = 20.0 * VJ
    m_c = np.zeros(M)                                                             # 끝 뒤 명령은 실제 k ≤ N + N_D 에 무게 0 (값 무관)
    B = _v3_basis(N, V3_KNOT); nb = B.shape[1]; nx = 6 * nb + 1                 # 마지막 변수 = Δv_out
    L = np.zeros((K, 6, nx))                                                      # 명령 변화 (잡기 0..N + 이어짐) = L x
    for j in range(6):
        L[1:n, j, j * nb:(j + 1) * nb] = B
    for i in range(M):
        L[n + i] = (1 + m_c[i]) * L[N] - m_c[i] * L[N - 1]
    Toe = np.zeros((K, K))
    for jj in range(K):
        mm = np.arange(K) - jj; ok = (mm >= 0) & (mm < pad); Toe[ok, jj] = h[mm[ok]]
    Lr = np.einsum("kj,jax->kax", Toe, L)                                         # 실제 관절 변화 (K, 6, nx)
    ctrl = np.abs(Lr).reshape(K, -1).max(1) > 1e-12                               # 잡기 명령이 닿는 실제 샘플 (불감 뒤)
    P_rot = np.zeros((3, 6)); P_rot[0, 3] = 1; P_rot[1, 4] = 1; P_rot[1, 1] = -1; P_rot[1, 2] = 1; P_rot[2, 5] = 1   # 자세 어긋남 = P_rot q + c
    c_rot = np.array([0.0, np.pi / 2, -j6])
    v_out = float(v_out0)

    def evaluate(Qc):
        C = np.vstack([Qc, Qc[-1] + (Qc[-1] - Qc[-2]) * m_c[:, None]])
        U = np.vstack([hist, C])
        qr = np.stack([np.convolve(U[:, j], h)[:len(U)] for j in range(6)], -1)[pad - 1:pad - 1 + K]
        return C, qr, tcp_batch(qr)

    def measures(Qc, vo):
        C, qr, pr = evaluate(Qc)
        k_t = int(np.argmin(pr[:, 2])); k_e = K - 1
        # 잡기 구간 (11-04 사용자 "잡기 명령 시간이 아니라 실제로 잡기 궤적이 발현되는 시간 동안"): 실제 TCP 가 h 아래로 처음 내려온 샘플부터 끝까지
        dn = np.flatnonzero(pr[:, 2] <= z_h); k_s = int(dn[0]) if len(dn) else K - 1
        W = np.arange(k_s, K); Wc = W[ctrl[W]]
        e = np.abs(pr[:, :2] - obj_xy).max(1)
        cz = (pr[:, 2] <= z_o + V3_HC)
        rot = np.abs(qr @ P_rot.T + c_rot).max(1)
        Cf = np.vstack([hist[-2:], Qc])
        vr = float((np.abs(np.diff(Cf, axis=0)) / dt / VJ).max()); ar = float((np.abs(np.diff(Cf, 2, axis=0)) / dt ** 2 / AJ).max())
        pc = tcp_batch(Qc[-2:])
        return dict(C=C, qr=qr, pr=pr, k_t=k_t, k_e=k_e, k_s=k_s, rise=float(pr[K - 1, 2] - pr[K - 2, 2]), W=W, Wc=Wc, e=e, cz=cz, rot=rot, vr=vr, ar=ar,
                    e_c=float(e[Wc][cz[Wc]].max()) if cz[Wc].any() else float("nan"), e_w=float(e[Wc].max()) if len(Wc) else 0.0,
                    e_rot=float(rot[Wc].max()) if len(Wc) else 0.0, z_gap=float(pr[k_t, 2] - z_o),
                    z_min=float(pr[ctrl, 2].min() - z_o), end_z=float(pc[1, 2] - z_h), end_v=float((pc[1, 2] - pc[0, 2]) / dt - vo))

    it = 0; viol = np.inf
    for it in range(1, iters + 1):
        m = measures(Q, v_out); qr, pr = m["qr"], m["pr"]; k_t, Wc, cz = m["k_t"], m["Wc"], m["cz"]
        eps = 1e-6; Jp = np.empty((K, 3, 6))
        for i in range(6):
            dq = np.zeros(6); dq[i] = eps
            Jp[:, :, i] = (tcp_batch(qr + dq) - tcp_batch(qr - dq)) / (2 * eps)
        G = np.einsum("kab,kbx->kax", Jp, Lr)                                     # 실제 TCP 변화 (K, 3, nx)
        Gr = np.einsum("ab,kbx->kax", P_rot, Lr)                                  # 실제 자세 어긋남 변화
        rows, rhs, slack, kind = [], [], [], []

        def add(A, b, s, k=0):                                                    # A x ≤ b + s·t_k  (k 0: 위치 [m], 1: 관절 한계 [한계 비], 2: xy 최대 어긋남 d)
            rows.append(A); rhs.append(b); slack.append(np.full(len(b), s)); kind.append(np.full(len(b), k))
        # 관절 한계 (명령, 이음 포함) — 한계로 나눈 행: 넘은 비율도 최대 위반 t_l 로 함께 줄인다 (초기 경로가 이미 몇 % 넘을 수 있다)
        Cf = np.vstack([hist[-2:], Q]); Lf = np.concatenate([np.zeros((2, 6, nx)), L[:n]], 0)
        D1 = (Lf[1:] - Lf[:-1]).reshape(-1, nx); v0 = (Cf[1:] - Cf[:-1]).ravel()
        D2 = (Lf[2:] - 2 * Lf[1:-1] + Lf[:-2]).reshape(-1, nx); a0 = (Cf[2:] - 2 * Cf[1:-1] + Cf[:-2]).ravel()
        vlim = np.tile(VJ * dt, len(Cf) - 1); alim = np.tile(AJ * dt * dt, len(Cf) - 2)
        add(D1 / vlim[:, None], (vlim - v0) / vlim, 1.0, 1); add(-D1 / vlim[:, None], (vlim + v0) / vlim, 1.0, 1)
        add(D2 / alim[:, None], (alim - a0) / alim, 1.0, 1); add(-D2 / alim[:, None], (alim + a0) / alim, 1.0, 1)
        # 실제 z ≥ 윗면, 접촉
        kz = np.flatnonzero(ctrl)
        add(-G[kz, 2], pr[kz, 2] - z_o, 1.0)
        add(G[k_t:k_t + 1, 2], np.array([z_o + 3e-4 - pr[k_t, 2]]), 1.0)
        add(-(G[K - 1:K, 2] - G[K - 2:K - 1, 2]), np.array([pr[K - 1, 2] - pr[K - 2, 2]]), 1.0)   # 접촉은 k ≤ N + N_D 안: 끝 샘플에서 실제 z 상승
        # 실제 xy (11-04 사용자 "전 구간에 대해서 최대한 물체 xy에 일치하도록 한개의 제약식만"): 잡기 구간 전체에서 |실제 xy − 물체 xy| ≤ d,
        # d ≤ s̄ (넘으면 위반 t_g) — 길이는 그대로 두고 d 를 줄이는 쪽을 고른다
        if len(Wc):
            ex = obj_xy[Wc] - pr[Wc, :2]
            for a in (0, 1):
                add(G[Wc, a], ex[:, a], 1.0, 2); add(-G[Wc, a], -ex[:, a], 1.0, 2)
        # 실제 자세
        if len(Wc):
            er = qr[Wc] @ P_rot.T + c_rot
            for a in range(3):
                add(Gr[Wc, a], V3_ALPHA - er[:, a], 1.0); add(-Gr[Wc, a], V3_ALPHA + er[:, a], 1.0)
        A = np.vstack(rows); b = np.concatenate(rhs); sc = np.concatenate(slack); kd = np.concatenate(kind)
        # 끝: 명령 z[N] = z_h, z[N] − z[N−1] = v_out·dt  (명령 FK 선형화; v_out 은 변수 또는 고정)
        pc = tcp_batch(Q[-2:]); Jc = np.empty((2, 6))
        for i in range(6):
            dq = np.zeros(6); dq[i] = eps
            Jc[:, i] = (tcp_batch(Q[-2:] + dq)[:, 2] - tcp_batch(Q[-2:] - dq)[:, 2]) / (2 * eps)
        Aeq = np.zeros((2, nx)); beq = np.zeros(2)
        Aeq[0] = Jc[1] @ L[N]; beq[0] = z_h - pc[1, 2]
        Aeq[1] = Jc[1] @ L[N] - Jc[0] @ L[N - 1]; Aeq[1, -1] = -dt; beq[1] = v_out * dt - (pc[1, 2] - pc[0, 2])
        # 변수 [x (nx), t_g (위치 위반 m), t_l (한계 위반 비), |Δv_out| 보조, d (xy 최대 어긋남 m)]:
        # 최소화 1000·t_g + 50·t_l (1 mm ≈ 한계 2 %) + 1e-4·|Δv_out| + 1·d — 위반을 먼저 없애고, 그 안에서 xy 어긋남을 줄인다
        # (v_out 은 목적이 없어 꼭짓점 아무 값 — 초기 추정에 가까운 것을 고르는 타이브레이크)
        A_full = np.hstack([A, -(sc * (kd == 0))[:, None], -(sc * (kd == 1))[:, None], np.zeros((len(b), 1)), -(sc * (kd == 2))[:, None]])
        A_vo = np.zeros((2, nx + 4)); A_vo[0, nx - 1] = 1; A_vo[0, nx + 2] = -1; A_vo[1, nx - 1] = -1; A_vo[1, nx + 2] = -1
        A_d = np.zeros((1, nx + 4)); A_d[0, nx + 3] = 1; A_d[0, nx] = -1                         # d − t_g ≤ s̄
        bounds = [(-V3_TR, V3_TR)] * (nx - 1) + [((-1.0, 1.0) if v_out_free else (0.0, 0.0))] + [(0, None)] * 4
        cost = np.zeros(nx + 4); cost[nx] = 1000.0; cost[nx + 1] = 50.0; cost[nx + 2] = 1e-4; cost[nx + 3] = 1.0
        r = linprog(cost, A_ub=np.vstack([A_full, A_vo, A_d]), b_ub=np.r_[b, 0.0, 0.0, V3_SBAR], A_eq=np.hstack([Aeq, np.zeros((2, 4))]),
                    b_eq=beq, bounds=bounds, method="highs")
        if r.status != 0:
            return dict(ok=False, why=f"LP {r.status}", it=it, Q=Q, v_out=v_out)
        x = r.x[:nx]; viol = float(max(r.x[nx], r.x[nx + 1] * 1e-3 / 0.02))
        dQ = np.einsum("kax,x->ka", L[:n], x)
        Q = Q + dQ; Q[0] = np.asarray(Q0, float)[0]; v_out += float(x[-1])
        if viol < V3_TOL and np.abs(x[:-1]).max() < 2e-5:
            break
    m = measures(Q, v_out)
    ok = m["e_w"] <= V3_SBAR + 3e-4 and m["e_rot"] <= V3_ALPHA + 1e-3 and m["rise"] >= -1e-6 and m["k_t"] < K - 1 \
        and abs(m["z_gap"]) <= 1e-3 and m["z_min"] >= -1e-3 and m["vr"] <= 1.02 and m["ar"] <= 1.05 and abs(m["end_z"]) < 1e-3
    return dict(ok=bool(ok), Q=Q, v_out=float(v_out), k_t=m["k_t"], e_contact=float(m["e"][m["k_t"]]), e_c=m["e_c"], e_w=m["e_w"],
                e_rot=m["e_rot"], z_gap=m["z_gap"], z_min=m["z_min"], viol=viol, vr=m["vr"], ar=m["ar"], it=it,
                why=None if ok else "조건 미달")


# -- 잡기 v4 (2026-11-04 사용자 "등식은 모두 명령을 실제 궤적으로 변환한 값으로 … 시간도 실제 궤적이 발현되는 시간으로", 논문 식 5) ------------
# 명령 샘플 k 는 실제 k + N_D (불감) 부터 보인다 → 잡기 = 실제 구간 𝒦 = {N_D, …, N + N_D}. 명령에는 𝒰 (이음 포함) 만, 나머지는 모두 실제:
# 𝒦 에서 z ≥ 윗면, 접촉 z = 윗면, |xy − 물체| ≤ d ≤ s̄, 자세 ≤ α, 끝 (N + N_D) 실제 z = h·실제 상승 속력 = v_out.
def solve_v4(hist, Q0, obj_xy, z_o: float, z_h: float, v_out0: float, v_lim, j6: float, v_out_free: bool = True,
             dt: float = TS, iters: int = None) -> dict:
    """hist (≥K−1, 6): 잡기 앞 명령 (마지막 = Q0[0] 바로 앞), Q0 (N+1, 6): 초기 명령 (Q0[0] = 이음, 고정),
    obj_xy (≥ N+1+N_D, 2): 실제 샘플 k 의 물체 xy, z_o: 윗면, z_h: 끝 실제 높이 h, v_out0: 초기 실제 상승 속력."""
    from gp8_control.grasping.workspace_gate import tcp_batch
    iters = V3_ITERS if iters is None else iters
    h = _h(); pad = len(h); Q = np.asarray(Q0, float).copy(); n = len(Q); N = n - 1
    ND = int(np.argmax(np.abs(h) > 1e-15)); K = n + ND
    hist = np.asarray(hist, float)
    if len(hist) < pad - 1:
        hist = np.vstack([np.repeat(hist[:1], pad - 1 - len(hist), 0), hist])
    hist = hist[-(pad - 1):]
    obj_xy = np.asarray(obj_xy, float)[:K]
    VJ = LIM_SCALE * np.asarray(v_lim, float); AJ = 20.0 * VJ
    B = _v3_basis(N, V3_KNOT); nb = B.shape[1]; nx = 6 * nb + 1                 # 마지막 변수 = Δv_out
    L = np.zeros((K, 6, nx))
    for j in range(6):
        L[1:n, j, j * nb:(j + 1) * nb] = B
    L[n:] = L[N]                                                                  # 끝 뒤 명령은 실제 k ≤ N + N_D 에 무게 0 (값 무관)
    Toe = np.zeros((K, K))
    for jj in range(K):
        mm = np.arange(K) - jj; ok = (mm >= 0) & (mm < pad); Toe[ok, jj] = h[mm[ok]]
    Lr = np.einsum("kj,jax->kax", Toe, L)
    Kw = np.arange(ND, K)                                                         # 실제 잡기 구간 𝒦
    P_rot = np.zeros((3, 6)); P_rot[0, 3] = 1; P_rot[1, 4] = 1; P_rot[1, 1] = -1; P_rot[1, 2] = 1; P_rot[2, 5] = 1
    c_rot = np.array([0.0, np.pi / 2, -j6])
    v_out = float(v_out0)

    def evaluate(Qc):
        U = np.vstack([hist, Qc, np.repeat(Qc[-1:], ND, 0)])
        qr = np.stack([np.convolve(U[:, j], h)[:len(U)] for j in range(6)], -1)[pad - 1:pad - 1 + K]
        return qr, tcp_batch(qr)

    def measures(Qc, vo):
        qr, pr = evaluate(Qc)
        k_t = int(ND + np.argmin(pr[ND:, 2]))
        e = np.abs(pr[:, :2] - obj_xy).max(1)
        rot = np.abs(qr @ P_rot.T + c_rot).max(1)
        Cf = np.vstack([hist[-2:], Qc])
        vr = float((np.abs(np.diff(Cf, axis=0)) / dt / VJ).max()); ar = float((np.abs(np.diff(Cf, 2, axis=0)) / dt ** 2 / AJ).max())
        return dict(qr=qr, pr=pr, k_t=k_t, e=e, rot=rot, vr=vr, ar=ar, e_w=float(e[Kw].max()), e_rot=float(rot[Kw].max()),
                    z_gap=float(pr[k_t, 2] - z_o), z_min=float(pr[Kw, 2].min() - z_o),
                    end_z=float(pr[K - 1, 2] - z_h), end_v=float((pr[K - 1, 2] - pr[K - 2, 2]) / dt - vo))

    it = 0; viol = np.inf
    for it in range(1, iters + 1):
        m = measures(Q, v_out); qr, pr, k_t = m["qr"], m["pr"], m["k_t"]
        eps = 1e-6; Jp = np.empty((K, 3, 6))
        for i in range(6):
            dq = np.zeros(6); dq[i] = eps
            Jp[:, :, i] = (tcp_batch(qr + dq) - tcp_batch(qr - dq)) / (2 * eps)
        G = np.einsum("kab,kbx->kax", Jp, Lr)
        Gr = np.einsum("ab,kbx->kax", P_rot, Lr)
        rows, rhs, slack, kind = [], [], [], []

        def add(A, b, s, k=0):                                                    # A x ≤ b + s·t_k  (k 0: 위치 [m], 1: 관절 한계 [한계 비], 2: xy 최대 어긋남 d)
            rows.append(A); rhs.append(b); slack.append(np.full(len(b), s)); kind.append(np.full(len(b), k))
        Cf = np.vstack([hist[-2:], Q]); Lf = np.concatenate([np.zeros((2, 6, nx)), L[:n]], 0)
        D1 = (Lf[1:] - Lf[:-1]).reshape(-1, nx); v0 = (Cf[1:] - Cf[:-1]).ravel()
        D2 = (Lf[2:] - 2 * Lf[1:-1] + Lf[:-2]).reshape(-1, nx); a0 = (Cf[2:] - 2 * Cf[1:-1] + Cf[:-2]).ravel()
        vlim = np.tile(VJ * dt, len(Cf) - 1); alim = np.tile(AJ * dt * dt, len(Cf) - 2)
        add(D1 / vlim[:, None], (vlim - v0) / vlim, 1.0, 1); add(-D1 / vlim[:, None], (vlim + v0) / vlim, 1.0, 1)
        add(D2 / alim[:, None], (alim - a0) / alim, 1.0, 1); add(-D2 / alim[:, None], (alim + a0) / alim, 1.0, 1)
        add(-G[Kw, 2], pr[Kw, 2] - z_o, 1.0)                                      # 실제 z ≥ 윗면 (𝒦)
        add(G[k_t:k_t + 1, 2], np.array([z_o + 3e-4 - pr[k_t, 2]]), 1.0)          # 접촉
        ex = obj_xy[Kw] - pr[Kw, :2]
        for a in (0, 1):
            add(G[Kw, a], ex[:, a], 1.0, 2); add(-G[Kw, a], -ex[:, a], 1.0, 2)
        er = qr[Kw] @ P_rot.T + c_rot
        for a in range(3):
            add(Gr[Kw, a], V3_ALPHA - er[:, a], 1.0); add(-Gr[Kw, a], V3_ALPHA + er[:, a], 1.0)
        A = np.vstack(rows); b = np.concatenate(rhs); sc = np.concatenate(slack); kd = np.concatenate(kind)
        # 끝 (실제 N + N_D): z = h, 상승 속력 = v_out — 위반은 t_g 로 (등식 ±)
        Ae = np.zeros((2, nx)); be = np.zeros(2)
        Ae[0] = G[K - 1, 2]; be[0] = z_h - pr[K - 1, 2]
        Ae[1] = G[K - 1, 2] - G[K - 2, 2]; Ae[1, -1] = -dt; be[1] = v_out * dt - (pr[K - 1, 2] - pr[K - 2, 2])
        A = np.vstack([A, Ae, -Ae]); b = np.r_[b, be, -be]; sc = np.r_[sc, np.ones(4)]; kd = np.r_[kd, np.zeros(4, int)]
        A_full = np.hstack([A, -(sc * (kd == 0))[:, None], -(sc * (kd == 1))[:, None], np.zeros((len(b), 1)), -(sc * (kd == 2))[:, None]])
        A_vo = np.zeros((2, nx + 4)); A_vo[0, nx - 1] = 1; A_vo[0, nx + 2] = -1; A_vo[1, nx - 1] = -1; A_vo[1, nx + 2] = -1
        A_d = np.zeros((1, nx + 4)); A_d[0, nx + 3] = 1; A_d[0, nx] = -1
        bounds = [(-V3_TR, V3_TR)] * (nx - 1) + [((-1.0, 1.0) if v_out_free else (0.0, 0.0))] + [(0, None)] * 4
        cost = np.zeros(nx + 4); cost[nx] = 1000.0; cost[nx + 1] = 50.0; cost[nx + 2] = 1e-4; cost[nx + 3] = 1.0
        r = linprog(cost, A_ub=np.vstack([A_full, A_vo, A_d]), b_ub=np.r_[b, 0.0, 0.0, V3_SBAR], bounds=bounds, method="highs")
        if r.status != 0:
            return dict(ok=False, why=f"LP {r.status}", it=it, Q=Q, v_out=v_out)
        x = r.x[:nx]; viol = float(max(r.x[nx], r.x[nx + 1] * 1e-3 / 0.02))
        Q = Q + np.einsum("kax,x->ka", L[:n], x); Q[0] = np.asarray(Q0, float)[0]; v_out += float(x[-1])
        if viol < V3_TOL and np.abs(x[:-1]).max() < 2e-5:
            break
    m = measures(Q, v_out)
    chk = dict(xy=m["e_w"] <= V3_SBAR + 3e-4, rot=m["e_rot"] <= V3_ALPHA + 1e-3, contact=abs(m["z_gap"]) <= 1e-3, above=m["z_min"] >= -1e-3,
               lim=m["vr"] <= 1.02 and m["ar"] <= 1.05, end_z=abs(m["end_z"]) < 1e-3, end_v=abs(m["end_v"]) < 0.02)
    ok = all(chk.values())
    return dict(ok=bool(ok), Q=Q, v_out=float(v_out), k_t=m["k_t"], e_w=m["e_w"], e_rot=m["e_rot"], z_gap=m["z_gap"], z_min=m["z_min"],
                end_z=m["end_z"], end_v=m["end_v"], viol=viol, vr=m["vr"], ar=m["ar"], it=it, pr=m["pr"],
                why=None if ok else ",".join(k for k, v in chk.items() if not v))


# -- 잡기 B-스플라인 (2026-11-04 사용자 "bspline 쓰면 되잖아", 논문 version2 식 1·6, GP8_SEP_RT=1) -------------------------------
# 명령 u(t) = Σ Q_k N_k(t/T), 제어점 수 고정 (Q_0 = 앞 던지기 마지막 명령), 잡기 시간 T 연속 변수. 실제 잡기 = [D, T + D].
# 순차 LP: 매 단계 실제 TCP 를 (ΔQ, ΔT) 에 1차 근사 — 위반 → T → xy 최대 어긋남 순으로 줄인다. 끝: 실제 T + D 에서 z = h, 상승 속력 = v_out.
RT = os.environ.get("GP8_SEP_RT", "0") not in ("0", "", "false")
BS_NC = int(os.environ.get("GP8_SEP_BS_NC", "40"))                # 제어점 수
BS_ITERS = int(os.environ.get("GP8_SEP_BS_ITERS", "30"))
BS_CMD_XMIN = float(os.environ.get("GP8_SEP_BS_CMD_XMIN", "0.24"))   # 잡기 명령 TCP x 하한 [m] (0 이면 끔)
BS_W_STEP = float(os.environ.get("GP8_SEP_BS_W_STEP", "1e-4"))      # 제어점 변화 |ΔP| 벌점 [/rad] (T 1 s = 1)
BS_VOUT_MAX = float(os.environ.get("GP8_SEP_BS_VOUT_MAX", "1.5"))   # 실제 상승 속력 상한 [m/s] (11-04 RT 시뮬: 2.4~3.6 이면 잡기 끝 명령이 h 위 0.5 m, 뒤 정지 구간이 베이스 쪽으로)
BS_STRIDE = int(os.environ.get("GP8_SEP_BS_STRIDE", "5"))                     # 11-05: SLP 반복 중 실제 궤적 행 간격 [샘플] (명령 행까지 줄이면 4 는 수렴 안 함, 8 은 관절 한계 위반 — 명령 행은 늘 매 샘플)
BS_CP_LIM = os.environ.get("GP8_SEP_BS_CPLIM", "1") not in ("0", "", "false")   # 11-05 사용자: 명령 관절 속도·가속도·위치 한계를 제어점에
BS_CP_JUNC = int(os.environ.get("GP8_SEP_BS_CPJUNC", "8"))                       # 그래도 샘플 행으로 두는 처음 샘플 수 (이력과의 이음)
_BS = {}


def _bs_basis():
    if "B" not in _BS:
        from scipy.interpolate import BSpline
        kn = np.r_[[0.0] * 4, np.linspace(0, 1, BS_NC - 2)[1:-1], [1.0] * 4]
        B = BSpline(kn, np.eye(BS_NC), 3)
        _BS["B"] = (B, B.derivative())
    return _BS["B"]


def _bs_dmat():
    """제어점 → 1·2 계 미분 (d/ds) 제어점 행렬 (BS_NC−1, BS_NC), (BS_NC−2, BS_NC). 3 차 clamped: Q_i = 3 (P_{i+1} − P_i)/(u_{i+4} − u_{i+1}),
    R_i = 2 (Q_{i+1} − Q_i)/(u_{i+4} − u_{i+2}). 곡선의 미분은 이 제어점들의 볼록 껍질 안 → 제어점에 건 한계가 곡선 전체의 한계."""
    if "D" not in _BS:
        kn = np.r_[[0.0] * 4, np.linspace(0, 1, BS_NC - 2)[1:-1], [1.0] * 4]
        DV = np.zeros((BS_NC - 1, BS_NC)); DA = np.zeros((BS_NC - 2, BS_NC - 1))
        for i in range(BS_NC - 1):
            c = 3.0 / (kn[i + 4] - kn[i + 1]); DV[i, i] = -c; DV[i, i + 1] = c
        for i in range(BS_NC - 2):
            c = 2.0 / (kn[i + 4] - kn[i + 2]); DA[i, i] = -c; DA[i, i + 1] = c
        _BS["D"] = (DV, DA @ DV)
    return _BS["D"]


def bs_fit(Q_samples, T: float, q0) -> np.ndarray:
    """격자 명령 샘플 (0..N, Q[0] = q0) → 제어점 (BS_NC, 6), 첫 점 = q0 고정 최소제곱."""
    B, _ = _bs_basis(); Q_samples = np.asarray(Q_samples, float); n = len(Q_samples) - 1
    s = np.minimum(np.arange(1, n + 1) * TS / T, 1.0); Bm = B(s)
    P = np.zeros((BS_NC, 6)); P[0] = q0
    P[1:] = np.linalg.lstsq(Bm[:, 1:], Q_samples[1:] - Bm[:, :1] * np.asarray(q0)[None], rcond=None)[0]
    return P


def bs_samples(P, T: float) -> np.ndarray:
    """제어점 → 격자 명령 샘플 0..⌊T/TS⌋ (호출부가 T 를 격자에 맞춘다)."""
    B, _ = _bs_basis(); n = int(np.floor(T / TS + 1e-9))
    return B(np.minimum(np.arange(0, n + 1) * TS / T, 1.0)) @ np.asarray(P, float)


def solve_bs(hist, P, T: float, obj_at, z_o: float, z_h: float, v_out: float, v_lim, j6: float, iters: int = None,
             tr_p: float = 0.03, tr_t: float = 0.01, fix_T: bool = False, v_out_free: bool = True, qlim=None) -> dict:
    """hist: 잡기 앞 명령 (마지막 = P[0] 바로 앞), P (BS_NC, 6): 초기 제어점 (P[0] 고정), T: 초기 잡기 시간 [s],
    obj_at(t): 잡기 명령 시작 기준 시각 t 의 물체 위치, z_o 윗면, z_h 끝 실제 높이 h, v_out 초기 실제 상승 속력."""
    from gp8_control.grasping.workspace_gate import tcp_batch
    iters = BS_ITERS if iters is None else iters
    B, Bdv = _bs_basis(); hk = _h(); pad = len(hk); ND = int(np.argmax(np.abs(hk) > 1e-15)); Dd = ND * TS
    VJ = LIM_SCALE * np.asarray(v_lim, float); AJ = 20.0 * VJ
    c_rot = np.array([0.0, np.pi / 2, -j6])
    P_rot = np.zeros((3, 6)); P_rot[0, 3] = 1; P_rot[1, 4] = 1; P_rot[1, 1] = -1; P_rot[1, 2] = 1; P_rot[2, 5] = 1
    hist = np.asarray(hist, float)[:, :6]
    if len(hist) < pad - 1:
        hist = np.vstack([np.repeat(hist[:1], pad - 1 - len(hist), 0), hist])
    hist = hist[-(pad - 1):]
    P = np.asarray(P, float).copy(); nP = BS_NC - 1; nx = 6 * nP + 2

    def build(P, T):
        Nc = int(np.floor(T / TS + 1e-9)); s = np.arange(1, Nc + 1) * TS / T
        Bm = B(s); Bd = Bdv(s)
        U = Bm @ P; dUT = -(s / T)[:, None] * (Bd @ P)
        nh = ND + 3
        U = np.vstack([U, np.repeat(P[-1:], nh, 0)]); dUT = np.vstack([dUT, np.zeros((nh, 6))])
        BmF = np.vstack([Bm, np.tile(np.eye(BS_NC)[-1], (nh, 1))])[:, 1:]
        Uf = np.vstack([hist, P[:1], U]); K = len(U) + 1
        qr = np.stack([np.convolve(Uf[:, j], hk)[:len(Uf)] for j in range(6)], -1)[pad - 1:pad - 1 + K]
        Bz = np.vstack([np.zeros((1, nP)), BmF]); dTz = np.vstack([np.zeros((1, 6)), dUT])
        R = np.stack([np.convolve(Bz[:, i], hk)[:K] for i in range(nP)], -1)
        RTm = np.stack([np.convolve(dTz[:, j], hk)[:K] for j in range(6)], -1)
        return Nc, U, BmF, dUT, qr, R, RTm

    def metrics(P, T, vo):
        Nc, U, BmF, dUT, qr, R, RTm = build(P, T); pr = tcp_batch(qr)
        e = (T + Dd) / TS; i = int(np.floor(e + 1e-9)); f = e - i
        Kw = np.arange(ND, i + 1)
        ob = np.array([obj_at(k * TS)[:2] for k in range(len(qr))])
        ew = np.abs(pr[:, :2] - ob).max(1); rot = np.abs(qr @ P_rot.T + c_rot).max(1)
        k_t = int(Kw[np.argmin(pr[Kw, 2])])
        z_end = (1 - f) * pr[i, 2] + f * pr[i + 1, 2]
        v_end = ((1 - f) * (pr[i, 2] - pr[i - 1, 2]) + f * (pr[i + 1, 2] - pr[i, 2])) / TS
        Cf = np.vstack([hist[-2:], P[:1], U[:Nc]])
        vr = float((np.abs(np.diff(Cf, axis=0)) / TS / VJ).max()); ar = float((np.abs(np.diff(Cf, 2, axis=0)) / TS ** 2 / AJ).max())
        return dict(Nc=Nc, U=U, BmF=BmF, dUT=dUT, qr=qr, R=R, RT=RTm, pr=pr, i=i, f=f, Kw=Kw, ob=ob, k_t=k_t, z_end=z_end, v_end=v_end,
                    e_w=float(ew[Kw].max()), e_rot=float(rot[Kw].max()), z_gap=float(pr[k_t, 2] - z_o), z_min=float(pr[Kw, 2].min() - z_o),
                    end_z=float(z_end - z_h), end_v=float(v_end - vo), vr=vr, ar=ar)

    it = 0; viol = np.inf; st = max(1, BS_STRIDE)
    for it in range(1, iters + 1):
        m = metrics(P, T, v_out); qr, pr, R, RTm, Kw, i, f, k_t = m["qr"], m["pr"], m["R"], m["RT"], m["Kw"], m["i"], m["f"], m["k_t"]
        if st > 1:                                                              # 11-05: 실제 (필터, 0.2 s 박스로 매끄러움) 궤적 행만 st 샘플마다 (접촉 순간·끝은 그대로).
            Kw = np.union1d(Kw[::st], [k_t, Kw[-1]])                            # 명령 행 (관절 속도·가속도·위치, 명령 x) 은 매 샘플 — 수렴 뒤 전체 행으로 마무리
        K = len(qr); eps = 1e-6; Jp = np.empty((K, 3, 6))
        for j in range(6):
            dq = np.zeros(6); dq[j] = eps
            Jp[:, :, j] = (tcp_batch(qr + dq) - tcp_batch(qr - dq)) / (2 * eps)
        G = np.zeros((K, 3, nx)); Gr = np.zeros((K, 3, nx))
        for j in range(6):
            G[:, :, j * nP:(j + 1) * nP] = Jp[:, :, j][:, :, None] * R[:, None, :]
            Gr[:, :, j * nP:(j + 1) * nP] = P_rot[:, j][None, :, None] * R[:, None, :]
        G[:, :, 6 * nP] = np.einsum("kab,kb->ka", Jp, RTm); Gr[:, :, 6 * nP] = RTm @ P_rot.T
        rows, rhs, kind = [], [], []

        def add(A, b, k=0):                                                    # A x ≤ b + t_k  (k 0: 위치 [m], 1: 관절 한계 [한계 비], 2: xy 최대 어긋남 d)
            rows.append(np.atleast_2d(A)); rhs.append(np.atleast_1d(b)); kind.append(np.full(len(np.atleast_1d(b)), k))
        Nc = m["Nc"]; Cv = np.vstack([hist[-2:], P[:1], m["U"][:Nc]])
        dC = np.zeros((Nc + 3, 6, nx))
        for j in range(6):
            dC[3:, j, j * nP:(j + 1) * nP] = m["BmF"][:Nc]
        dC[3:, :, 6 * nP] = m["dUT"][:Nc]
        D1 = (dC[1:] - dC[:-1]).reshape(-1, nx); v0 = (Cv[1:] - Cv[:-1]).ravel(); vl = np.tile(VJ * TS, len(Cv) - 1)
        D2 = (dC[2:] - 2 * dC[1:-1] + dC[:-2]).reshape(-1, nx); a0 = (Cv[2:] - 2 * Cv[1:-1] + Cv[:-2]).ravel(); al = np.tile(AJ * TS * TS, len(Cv) - 2)
        if BS_CP_LIM:                                                           # 11-05 사용자: 명령 관절 한계를 제어점에 (볼록 껍질) — 샘플 행 ~6,500 → ~1,000.
            nj = 6 * BS_CP_JUNC                                                 # 이력과 이어지는 처음 몇 샘플은 샘플 행 그대로 (이음의 속도·가속도)
            D1, v0, vl, D2, a0, al = D1[:nj], v0[:nj], vl[:nj], D2[:nj], a0[:nj], al[:nj]
        add(D1 / vl[:, None], (vl - v0) / vl, 1); add(-D1 / vl[:, None], (vl + v0) / vl, 1)
        add(D2 / al[:, None], (al - a0) / al, 1); add(-D2 / al[:, None], (al + a0) / al, 1)
        if BS_CP_LIM:                                                           # |Q_i| ≤ v̄ T (T 에 선형),  |R_i| ≤ ā T² (T 에 선형화)
            DV, DA = _bs_dmat(); Qc = DV @ P; Rc = DA @ P
            for j in range(6):
                cs_ = slice(j * nP, (j + 1) * nP); sv = VJ[j] * T; sa = AJ[j] * T * T
                for sg in (1.0, -1.0):
                    Av = np.zeros((BS_NC - 1, nx)); Av[:, cs_] = sg * DV[:, 1:]; Av[:, 6 * nP] = -VJ[j]
                    add(Av / sv, (sv - sg * Qc[:, j]) / sv, 1)
                    Aa = np.zeros((BS_NC - 2, nx)); Aa[:, cs_] = sg * DA[:, 1:]; Aa[:, 6 * nP] = -2.0 * AJ[j] * T
                    add(Aa / sa, (sa - sg * Rc[:, j]) / sa, 1)
        if BS_CMD_XMIN > 0:                                                     # 명령 TCP x ≥ x_min (11-04 RT 시뮬: 실제 구간 밖 끝 명령이 베이스 쪽으로 — 실제가 따라감)
            Uc_ = m["U"][:Nc]; xc_ = tcp_batch(Uc_)[:, 0]; Jx = np.empty((Nc, 6))
            for j in range(6):
                dq = np.zeros(6); dq[j] = eps
                Jx[:, j] = (tcp_batch(Uc_ + dq)[:, 0] - tcp_batch(Uc_ - dq)[:, 0]) / (2 * eps)
            add(-np.einsum("kj,kjx->kx", Jx, dC[3:]), xc_ - BS_CMD_XMIN)
        if qlim is not None and BS_CP_LIM:                                      # 𝒰 의 관절 위치 범위 — 제어점에 (볼록 껍질)
            for j in range(6):
                Ap = np.zeros((nP, nx)); Ap[:, j * nP:(j + 1) * nP] = np.eye(nP)
                add(Ap, qlim[1][j] - P[1:, j]); add(-Ap, P[1:, j] - qlim[0][j])
        elif qlim is not None:                                                  # 𝒰 의 관절 위치 범위 (던지기 NLP 상자 — 잡기 끝이 던지기 시작)
            Dp = dC[3:].reshape(-1, nx); u0 = Cv[3:].ravel()
            add(Dp, np.tile(qlim[1], Nc) - u0); add(-Dp, u0 - np.tile(qlim[0], Nc))
        add(-G[Kw, 2], pr[Kw, 2] - z_o)
        add(G[k_t, 2], z_o + 3e-4 - pr[k_t, 2])
        ex = m["ob"][Kw] - pr[Kw, :2]
        for a in (0, 1):
            add(G[Kw, a], ex[:, a], 2); add(-G[Kw, a], -ex[:, a], 2)
        er = qr[Kw] @ P_rot.T + c_rot
        for a in range(3):
            add(Gr[Kw, a], V3_ALPHA - er[:, a]); add(-Gr[Kw, a], V3_ALPHA + er[:, a])
        gz = (1 - f) * G[i, 2] + f * G[i + 1, 2]; gz[6 * nP] += (pr[i + 1, 2] - pr[i, 2]) / TS
        gv = ((1 - f) * (G[i, 2] - G[i - 1, 2]) + f * (G[i + 1, 2] - G[i, 2])) / TS
        gv[6 * nP] += ((pr[i + 1, 2] - pr[i, 2]) - (pr[i, 2] - pr[i - 1, 2])) / TS ** 2
        gv = gv.copy(); gv[6 * nP + 1] -= 1.0
        for g, rr in ((gz, z_h - m["z_end"]), (gv, v_out - m["v_end"])):
            add(g, rr); add(-g, -rr)
        A = np.vstack(rows); b = np.concatenate(rhs); kd = np.concatenate(kind)
        # ΔP = p⁺ − p⁻ (각 ≥ 0) — |ΔP| 에 작은 벌점: 실제 구간에 거의 안 보이는 명령 (잡기 끝 ~0.1 s) 이 반복마다 떠다니지 않게 (11-04 RT 시뮬)
        nq = 6 * nP; Ap = A[:, :nq]
        A_full = np.hstack([Ap, -Ap, A[:, nq:], -(kd == 0)[:, None].astype(float), -(kd == 1)[:, None].astype(float),
                            -(kd == 2)[:, None].astype(float)])
        nz = nx + nq
        A_d = np.zeros((1, nz + 3)); A_d[0, nz + 2] = 1; A_d[0, nz] = -1                       # d − t_g ≤ s̄
        tb = (0.0, 0.0) if fix_T else (max(-tr_t, 0.08 - T), tr_t)
        vb_ = (max(-0.5, 0.0 - v_out), min(0.5, BS_VOUT_MAX - v_out)) if v_out_free else (0.0, 0.0)
        bounds = [(0.0, tr_p)] * (2 * nq) + [tb, vb_] + [(0, None)] * 3
        cost = np.zeros(nz + 3); cost[:2 * nq] = BS_W_STEP; cost[2 * nq] = 1.0; cost[nz] = 1000.0; cost[nz + 1] = 50.0; cost[nz + 2] = 0.2
        Aub = np.vstack([A_full, A_d]); Aub[np.abs(Aub) < 1e-12] = 0.0                     # 11-05: 희소로 (관절 한계 행은 B-스플라인 국소 지지)
        r = linprog(cost, A_ub=_sp.csr_matrix(Aub), b_ub=np.r_[b, V3_SBAR], bounds=bounds, method="highs")
        if r.status != 0:
            return dict(ok=False, why=f"LP {r.status}", it=it, P=P, T=T, v_out=v_out)
        x = np.r_[r.x[:nq] - r.x[nq:2 * nq], r.x[2 * nq:]]
        for j in range(6):
            P[1:, j] += x[j * nP:(j + 1) * nP]
        T += x[6 * nP]; v_out += x[6 * nP + 1]
        viol = max(x[nx], x[nx + 1] * 1e-3 / 0.02)
        if viol < 2e-4 and abs(x[6 * nP]) < 2e-4 and np.abs(x[:6 * nP]).max() < 2e-4:
            if st == 1:
                break
            st = 1                                                              # 성긴 행으로 수렴 → 전체 행으로 마무리 반복
    m = metrics(P, T, v_out)
    chk = dict(xy=m["e_w"] <= V3_SBAR + 3e-4, rot=m["e_rot"] <= V3_ALPHA + 1e-3, contact=abs(m["z_gap"]) <= 1e-3,
               above=m["z_min"] >= -1e-3, lim=m["vr"] <= 1.02 and m["ar"] <= 1.05, end_z=abs(m["end_z"]) < 1e-3, end_v=abs(m["end_v"]) < 0.02)
    qpos_info = None
    if qlim is not None:
        Uc_ = m["U"][:m["Nc"]]
        chk["qpos"] = bool(np.all(Uc_ >= qlim[0] - 1e-3) and np.all(Uc_ <= qlim[1] + 1e-3))
        if not chk["qpos"]:                                                  # 진단: 가장 많이 넘은 관절·값·한계 [deg]
            over = np.maximum(qlim[0] - Uc_, Uc_ - qlim[1]); k_, j_ = np.unravel_index(int(np.argmax(over)), over.shape)
            qpos_info = (int(j_) + 1, round(float(np.degrees(Uc_[k_, j_])), 1), round(float(np.degrees(qlim[0][j_])), 1),
                         round(float(np.degrees(qlim[1][j_])), 1), int(k_))
    ok = all(chk.values())
    return dict(ok=bool(ok), why=None if ok else ",".join(k for k, v in chk.items() if not v), P=P, T=float(T), v_out=float(v_out), it=it,
                k_t=m["k_t"], e_w=m["e_w"], e_rot=m["e_rot"], z_gap=m["z_gap"], z_min=m["z_min"], end_z=m["end_z"], end_v=m["end_v"],
                vr=m["vr"], ar=m["ar"], viol=float(viol), qpos_info=qpos_info)


def sink_depth(v: float, u: float) -> float:
    """논문 식 11: 실제 툴이 h 를 v 로 지난 뒤, 명령을 가장 빨리 (가속 20u, 상승 속력 u 까지) 뒤집어도 더 내려가는 깊이 [m]."""
    W1, W2 = 0.20, 0.08
    dx = 1e-4

    def box(w):
        n = max(1, int(round(w / dx)))
        return np.ones(n) / n
    phi = np.convolve(np.convolve(box(W1), box(W2)), box((v + u) / (20.0 * u))) / dx
    x = np.arange(len(phi)) * dx; Psi = np.cumsum(phi) * dx
    k = int(np.searchsorted(Psi, v / (v + u)))
    return float((v + u) * np.sum(x[:k] * phi[:k]) * dx)


def vbar(gap: float, u: float) -> float:
    """논문 식 12: v̄ = Δ⁻¹(h − z_i) (u: 그 반경 잡기 자세의 최대 수직 상승 속력). Δ 가 단조라 이분법."""
    if gap <= 0:
        return 0.0
    lo, hi = 0.0, 5.0
    if sink_depth(hi, u) <= gap:
        return hi
    for _ in range(30):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if sink_depth(m, u) <= gap else (lo, m)
    return lo
