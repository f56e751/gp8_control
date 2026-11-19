"""실제 궤적 작업공간 게이트 (2026-10-29 사용자).

계획이 보내는 **명령**이 아니라 컨트롤러 필터를 지난 **실제 궤적**에 대해,
  - 하강·상승 구간을 뺀 나머지에서 TCP z > WS_Z_MIN (기본 0.10 m = 준비점 높이; 1 mm 허용),
  - 계획 전체에서 TCP x ≥ WS_X_MIN (기본 0.22 m)
인지 본다. 어기면 그 계획은 실패로 처리하고 보내지 않는다 (sequence_app.gate_plan).

실제 궤적 = 명령을 D 늦춘 뒤 W₁, W₂ 이동평균 (throw_nlp.FILTER, 꺼져 있으면 실측 box200_80 = 0.04 s, 0.20 s, 0.08 s).
계획 시작 전 명령은 첫 샘플의 1차 외삽, 끝난 뒤는 마지막 명령 유지 (필터 기억 길이만큼 붙여 끝까지 본다).
하강·상승 구간 = 'descend'/'pick' 세그먼트 명령 시작 ~ 실제 z 최저점(실제 접촉; 잡기 명령 끝 + 필터 기억 안에서 찾음) 뒤 처음으로 WS_Z_MIN − 1 mm 에 다시 닿는 순간.
계획이 'throw' 로 시작하면 (직전 상승에 바로 이어 붙인 꼬리) 처음 필터 기억 안의 실제 최저점 뒤 WS_Z_MIN 에 닿을 때까지가 상승이다.
"""
import os
from typing import Optional, Sequence, Tuple

import numpy as np

WS_Z_MIN: float = float(os.environ.get("GP8_WS_Z_MIN", "0.10"))      # 10-29 사용자: 0.12 → 0.10 (준비점 높이)
WS_X_MIN: float = float(os.environ.get("GP8_WS_X_MIN", "0.22"))
WS_Z_MIN_RT_HEAD: float = float(os.environ.get("GP8_WS_Z_MIN_RT_HEAD", "0.055"))   # RT 계획 첫 필터 기억 구간의 z 하한 (가장 높은 물체 윗면 0.05 위)
WS_GATE: bool = os.environ.get("GP8_WS_GATE", "1") not in ("0", "", "false")
Z_TOL = 1e-3
# 11-04 사용자 "DB 만들때 던지기 궤적이 항상 z >= h 이상이도록 gate": 던지기 구간의 명령 TCP z ≥ 시작·끝 높이 (= h) 중 낮은 쪽
THROW_H: bool = os.environ.get("GP8_WS_THROW_H", os.environ.get("GP8_SEP_V2", "0")) not in ("0", "", "false")
DT = 0.002


def _filter_params() -> Tuple[float, Tuple[float, ...]]:
    try:
        from gp8_control.throwing.nlp import throw_nlp
        if throw_nlp.FILTER is not None:
            return throw_nlp.FILTER
    except Exception:                                                        # noqa: BLE001
        pass
    return 0.04, (0.20, 0.08)


def realized(q_cmd: np.ndarray, qd0: np.ndarray, dt: float = DT, hist: Optional[np.ndarray] = None) -> np.ndarray:
    """명령 (N,6) → 실제 (N+M,6): 지연 D + 이동평균 W₁, W₂. 앞은 이력 (hist = 이 계획 앞에 실제로 나간 명령; 없으면 1차 외삽),
    뒤는 마지막 명령 유지 (M = 필터 기억)."""
    D, Ws = _filter_params()
    nd = int(round(D / dt)); ns = [max(1, int(round(w / dt))) for w in Ws]
    mem = nd + sum(ns)
    q_cmd = np.asarray(q_cmd, float); qd0 = np.asarray(qd0, float)
    if hist is not None and len(hist) >= 1:
        hist = np.asarray(hist, float)[-mem:]
        if len(hist) < mem:                                                  # 모자라면 가장 오래된 명령을 그대로 앞에 채움
            hist = np.vstack([np.repeat(hist[:1], mem - len(hist), axis=0), hist])
    else:
        hist = q_cmd[0] + qd0 * (np.arange(-mem, 0)[:, None] * dt)
    tail = np.repeat(q_cmd[-1:], mem, axis=0)
    u = np.vstack([hist, q_cmd, tail])                                       # 인덱스 mem 이 계획 t=0
    y = u
    for n in ns:                                                             # 인과 이동평균 (길이 n·dt) — 샘플 사이 선형 보간의 정확한 적분 (사다리꼴)
        c = np.vstack([np.zeros((1, 6)), np.cumsum(0.5 * (y[:-1] + y[1:]), axis=0)])   # c[k] = ∫₀^{k dt} y / dt
        avg = (c[n:] - c[:-n]) / n                                           # avg[j] = 창 [j, j+n] 평균 → y_new[j+n]
        y = np.vstack([np.repeat(avg[:1], n, axis=0), avg])
    y = np.vstack([np.repeat(y[:1], nd, axis=0), y[:-nd]]) if nd else y      # 지연 D
    return y[mem:]                                                           # 계획 t=0 부터 (끝에 mem 샘플 더)


def tcp_batch(q_robot: np.ndarray) -> np.ndarray:
    """로봇 관절 (N,6) → TCP 위치 (N,3) — throw_planner.tcp_of (= throwing.fk_pos ∘ to_planner) 와 같은 FK 를 한 번에."""
    from gp8_control.throwing.nlp.throwing import _CHAIN
    from gp8_control.throwing.throw_planner import PLANNER_SIGN
    qp = np.asarray(q_robot, float) * PLANNER_SIGN
    n = len(qp)
    R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy(); p = np.zeros((n, 3))
    for (xyz, ax), i in zip(_CHAIN, range(6)):
        p = p + R @ np.asarray(xyz, float)
        c, s_ = np.cos(qp[:, i]), np.sin(qp[:, i]); Rot = np.zeros((n, 3, 3))
        if ax == "x":
            Rot[:, 0, 0] = 1; Rot[:, 1, 1] = c; Rot[:, 1, 2] = -s_; Rot[:, 2, 1] = s_; Rot[:, 2, 2] = c
        elif ax == "y":
            Rot[:, 1, 1] = 1; Rot[:, 0, 0] = c; Rot[:, 0, 2] = s_; Rot[:, 2, 0] = -s_; Rot[:, 2, 2] = c
        else:
            Rot[:, 2, 2] = 1; Rot[:, 0, 0] = c; Rot[:, 0, 1] = -s_; Rot[:, 1, 0] = s_; Rot[:, 1, 1] = c
        R = R @ Rot
    return p


def check(t: np.ndarray, q: np.ndarray, qd: np.ndarray, segments: Sequence, tcp_fn=None, hist: Optional[np.ndarray] = None) -> Optional[str]:
    """위반이면 사유 문자열, 통과면 None. t, q, qd: 계획 (N,), (N,6) 로봇 관절; segments: [(t0, Segment)]; tcp_fn(q_robot) → xyz."""
    if not WS_GATE or len(t) < 2:
        return None
    qr = realized(q, qd[0], hist=hist)
    n_all = len(qr)
    P = tcp_batch(qr) if tcp_fn is None else np.array([tcp_fn(qr[k]) for k in range(n_all)])
    tt = np.arange(n_all) * DT
    z, x = P[:, 2], P[:, 0]
    excl = np.zeros(len(tt), bool)

    D, Ws = _filter_params()
    mem = int(round((D + sum(Ws)) / DT))                                     # 필터 기억 (명령 끝 뒤 실제가 따라오는 시간)

    def _close_window(k_lo, k_hi):
        """[k_lo, k_hi + 필터 기억] 의 실제 z 최저점 (= 실제 접촉) 뒤에 처음으로 기준 높이에 다시 닿는 샘플."""
        k_hi = min(len(tt), k_hi + mem + 1)
        k_min = k_lo + int(np.argmin(z[k_lo:k_hi])) if k_hi > k_lo else k_lo
        up = np.nonzero(z[k_min:] >= WS_Z_MIN - Z_TOL)[0]
        return (k_min + int(up[0])) if len(up) else len(tt)

    # 11-09 사용자 "검사를 실행되는 사이클까지만": 통합 사이클 자리표시(jc_cert) 앞 = 이번에 실제로 나가는 명령의 실제 궤적만 본다.
    # 자리표시 뒤는 다음 사이클이 새로 계획해 자기 게이트(이력 포함)에서 검사한다.
    for _t0c, _sgc in segments:
        if (getattr(_sgc, "info", None) or {}).get("jc_cert"):
            _kc = int(np.searchsorted(tt, float(_t0c)))
            if 2 <= _kc < n_all:
                qr, P, tt, z, x, excl = qr[:_kc], P[:_kc], tt[:_kc], z[:_kc], x[:_kc], excl[:_kc]
            break
    names = [sg.name for _t0, sg in segments]
    if names and names[0] == "throw":                                        # 상승에 바로 이어 붙인 꼬리 → 실제 최저점 뒤 높이에 닿을 때까지 상승
        excl[:_close_window(0, 0)] = True
    i = 0
    while i < len(segments):
        t0, sg = segments[i]
        if sg.name in ("descend", "pick") or (getattr(sg, "info", None) or {}).get("ws_exclude"):
            t_end = t0 + sg.duration
            j = i
            while j + 1 < len(segments) and (segments[j + 1][1].name in ("descend", "pick") or (getattr(segments[j + 1][1], "info", None) or {}).get("ws_exclude")):
                j += 1; t_end = segments[j][0] + segments[j][1].duration
            k0 = int(np.searchsorted(tt, t0)); k1 = int(np.searchsorted(tt, t_end))
            excl[k0:_close_window(k0, k1)] = True
            i = j + 1
        else:
            i += 1
    if THROW_H and os.environ.get("GP8_SEP_RT", "0") not in ("0", "", "false"):   # 11-04 논문 식 9 gate: 던지기 실제 구간 [t0 + D, 끝] 실제 z ≥ h
        Dg = _filter_params()[0]
        for t0, sg in segments:
            if sg.name != "throw":
                continue
            k0 = int(np.searchsorted(tt, t0 + Dg)); k1 = int(np.searchsorted(tt, t0 + sg.duration))
            if k1 - k0 < 2:
                continue
            from gp8_control.grasping import sep_pick as _SPg
            hh = min(_SPg.Z_H, z[k0], z[k1 - 1])                              # travel height h (끝점이 그보다 낮으면 끝점)
            if z[k0:k1].min() < hh - Z_TOL:
                k = k0 + int(np.argmin(z[k0:k1]))
                return f"던지기 실제 z {z[k]:.3f} m < h {hh:.3f} m @ t={tt[k]:.2f}s"
    elif THROW_H:
        for t0, sg in segments:
            if sg.name != "throw":
                continue
            zc = (tcp_batch(np.asarray(sg.q, float)) if tcp_fn is None else np.array([tcp_fn(qq) for qq in sg.q]))[:, 2]
            hh = min(zc[0], zc[-1])
            if zc.min() < hh - Z_TOL:
                k = int(np.argmin(zc))
                return f"던지기 명령 z {zc[k]:.3f} m < h {hh:.3f} m @ t={t0 + float(sg.t[k]):.2f}s"
    zmin_k = np.full(len(tt), WS_Z_MIN)
    if os.environ.get("GP8_SEP_RT", "0") not in ("0", "", "false"):          # 11-04 RT: 앞 던지기가 실제 툴을 h 에서 하강 중으로 넘기므로, 계획 첫 필터 기억
        zmin_k[tt < mem * DT] = min(WS_Z_MIN, WS_Z_MIN_RT_HEAD)              # 구간은 이미 나간 명령이 정한다 (거부해도 못 바꿈) — 물체 윗면 위만 본다
    bad_z = (~excl) & (z < zmin_k - Z_TOL)
    if bad_z.any():
        k = int(np.argmax(bad_z))
        seg = next((sg.name for t0_, sg in reversed(list(segments)) if t0_ <= tt[k]), "?")
        return (f"실제 궤적 z {z[bad_z].min():.3f} m ≤ {float(zmin_k[k]):.3f} m (하강·상승 밖) @ t={tt[k]:.2f}s ({seg} 구간 부근, "
                f"위반 {int(bad_z.sum()) * DT * 1e3:.0f} ms)")
    if x.min() < WS_X_MIN:
        k = int(np.argmin(x))
        seg = next((sg.name for t0_, sg in reversed(list(segments)) if t0_ <= tt[k]), "?")
        return f"실제 궤적 x {x.min():.3f} m < {WS_X_MIN:.2f} m @ t={tt[k]:.2f}s ({seg} 구간 부근)"
    return None
