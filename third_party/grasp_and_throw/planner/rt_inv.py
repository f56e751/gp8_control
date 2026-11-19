"""rt_inv: one command chunk from a realized target (least squares, command limits, exact key points). Joints solved in parallel."""
import os
import numpy as np
import casadi as ca
import multiprocessing as mp

DT = 0.002; KB = 5; _C = {}; _POOL = [None]


def setup(M):
    _C.update(hk=np.asarray(M.hk, float), QD=np.asarray(M.V.QD, float), QDD=np.asarray(M.V.QDD, float), QLO=np.asarray(M.V.Q_LO, float) + 0.02, QHI=np.asarray(M.V.Q_HI, float) - 0.02, bs=M.bs_basis)
    _C["L"] = len(_C["hk"]); _C["D"] = int(np.argmax(_C["hk"] > 0))


_TW = float(os.environ.get("RT_TRACK_W", "1"))                                 # 11-18 (user): weight of the tracking term (realized against planned realized motion); 1 before
_VW = float(os.environ.get("RT_KEY_VW", "1e-4"))                                # 11-18 (user): weight of the velocity at a soft key point relative to its position (1e-4 before)
_SOLVER = os.environ.get("RT_INV_SOLVER", "qpoases")                          # 11-18 (user): "ipm" = interior-point solve of the joint QPs (setting_now.sh); "qpoases" = as before


def _ipm(a, tol=1e-9, itmax=60):
    """One joint QP  min 1/2 x'Hx + g'x  s.t. lba <= Ax <= uba, lo <= x <= hi  by a primal-dual interior-point method (Mehrotra predictor-corrector).
    The variables are scaled by the diagonal of H and every constraint row by its largest entry.  Returns x and whether it converged."""
    j, H, g, A, lba, uba, lo, hi, x0 = a; n = H.shape[0]; E = np.eye(n); g = np.asarray(g, float).ravel()
    G = np.vstack([A, -A, E, -E]); h = np.r_[np.asarray(uba, float).ravel(), -np.asarray(lba, float).ravel(), np.asarray(hi, float).ravel(), -np.asarray(lo, float).ravel()]
    d = 1.0 / np.sqrt(np.diag(H)); H = H * np.outer(d, d); g = g * d; G = G * d; rs = 1.0 / np.maximum(np.abs(G).max(1), 1e-12); G = G * rs[:, None]; h = h * rs; m = len(h)
    x = np.linalg.solve(H + 1e-9 * E, -g); s = np.maximum(h - G @ x, 1.0); z = np.ones(m); done = False

    def amax(v, dv):
        neg = dv < 0; return min(1.0, float((-v[neg] / dv[neg]).min())) if neg.any() else 1.0
    for _ in range(itmax):
        rd = H @ x + g + G.T @ z; rp = G @ x + s - h; mu = s @ z / m
        if max(np.abs(rd).max(), np.abs(rp).max()) < tol * 1e3 and mu < tol:
            done = True; break
        W = z / s; Lc = np.linalg.cholesky(H + (G.T * W) @ G + 1e-12 * E)

        def step(rc):
            dx = np.linalg.solve(Lc.T, np.linalg.solve(Lc, -rd - G.T @ (W * rp - rc / s))); ds = -rp - G @ dx; return dx, ds, -(rc + z * ds) / s
        dx, ds, dz = step(s * z); al = min(amax(s, ds), amax(z, dz)); sig = (((s + al * ds) @ (z + al * dz)) / m / mu) ** 3
        dx, ds, dz = step(s * z + ds * dz - sig * mu); al = 0.99 * min(amax(s, ds), amax(z, dz)); x = x + al * dx; s = s + al * ds; z = z + al * dz
    return x * d, bool(done and np.all(np.isfinite(x)))


def _one(a):
    j, H, g, A, lba, uba, lo, hi, x0 = a
    if _SOLVER == "ipm" and not np.any(np.asarray(lba, float).ravel() >= np.asarray(uba, float).ravel()):     # no equality rows
        try:
            x, ok = _ipm(a)
            if ok:
                return x, True
        except Exception:  # noqa: BLE001  (e.g. Cholesky breakdown) -> qpOASES below
            pass
    S = ca.conic("c", "qpoases", dict(h=ca.DM(H).sparsity(), a=ca.DM(A).sparsity()), {"printLevel": "none", "error_on_fail": False})
    r = S(h=ca.DM(H), g=ca.DM(g), a=ca.DM(A), lba=lba, uba=uba, lbx=lo, ubx=hi, x0=x0)
    return np.array(r["x"]).ravel(), bool(S.stats().get("success", True))


def pool():
    if _POOL[0] is None:
        _POOL[0] = mp.get_context("fork").Pool(6); _POOL[0].map(abs, range(6))
    return _POOL[0]


def chunk(hist, Qt, keys, n_cmd, still=None, soft=None, smooth=None):
    """hist: previous commands (>= L rows, last = sample k_a). Qt: realized targets for samples k_a+D+1 .. k_a+D+n_cmd (n_cmd rows).
    keys: [(tau, q, v)] or [(tau, q, v, w)] with tau = sample position relative to k_a (float) and w a factor on the key-point weight (soft keys only; 11-17). Returns commands for samples k_a+1 .. k_a+n_cmd and fit info."""
    hk, L, D = _C["hk"], _C["L"], _C["D"]; N = n_cmd; Bm = np.asarray(_C["bs"](N, KB).todense()); Mc = Bm.shape[1]; hist = np.asarray(hist, float)
    Rm = np.zeros((N, N)); r0 = np.zeros((N, 6))
    for i in range(N):                                                        # realized sample n = D + 1 + i (relative to k_a) = sum_k hk[k] u[n - k]
        n = D + 1 + i
        for k in range(D, L):
            m = n - k
            if m >= 1:
                Rm[i, m - 1] = hk[k]
            else:
                r0[i] += hk[k] * hist[len(hist) - 1 + m]
    RB = Rm @ Bm; E = np.eye(N); D1 = (E - np.vstack([np.zeros(N), E[:-1]])) / DT; D2 = (E - 2 * np.vstack([np.zeros(N), E[:-1]]) + np.vstack([np.zeros((2, N)), E[:-2]])) / DT ** 2
    c1 = np.zeros((N, 6)); c1[0] = -hist[-1] / DT; c2 = np.zeros((N, 6)); c2[0] = (-2 * hist[-1] + hist[-2]) / DT ** 2; c2[1] = hist[-1] / DT ** 2      # history terms of velocity / acceleration
    rows = []; vals = []; srows = []; svals = []; kinfo = []
    for key in keys:
        tau, q, v = key[:3]; wk = float(key[3]) if len(key) > 3 else 1.0
        i0 = int(np.clip(np.floor(tau - (D + 1)), 1, N - 3)); idx = np.arange(i0 - 1, i0 + 3); xs = (idx + D + 1) - tau; Vi = np.linalg.inv(np.vander(xs, 4, increasing=True)); wp = Vi[0]; wv = Vi[1] / DT; kinfo.append((idx, wp, wv, np.asarray(q, float), np.asarray(v, float)))
        if soft is None:
            rows += [wp @ RB[idx], wv @ RB[idx]]; vals += [(q - wp @ r0[idx]), (v - wv @ r0[idx])]
        else:                                                                 # key points as heavy weights instead of equalities (position weight soft, velocity weight soft * RT_KEY_VW)
            srows += [np.sqrt(soft * wk) * (wp @ RB[idx]), np.sqrt(soft * wk * _VW) * (wv @ RB[idx])]; svals += [np.sqrt(soft * wk) * (q - wp @ r0[idx]), np.sqrt(soft * wk * _VW) * (v - wv @ r0[idx])]
    if still is not None:                                                      # commands at samples still-1 and still are equal to the one before (so a hold can follow)
        for m_ in (still - 1, still):
            rows.append((D1 @ Bm)[m_ - 1]); vals.append(-c1[m_ - 1])
    A = np.vstack([D1 @ Bm, D2 @ Bm] + rows); SR = np.array(srows) if srows else np.zeros((0, Mc)); H = 2 * _TW * RB.T @ RB + 2 * SR.T @ SR + 1e-9 * np.eye(Mc); args = []
    for j in range(6):
        lba = np.r_[-_C["QD"][j] - c1[:, j], -_C["QDD"][j] - c2[:, j]] ; uba = np.r_[_C["QD"][j] - c1[:, j], _C["QDD"][j] - c2[:, j]]; kv = [vv[j] for vv in vals]
        gs = -2 * SR.T @ np.array([sv[j] for sv in svals]) if srows else 0.0; Hj = H
        if smooth:                                                             # smooth command: + smooth * sum (command acceleration / rated speed)^2   (rated speed = QD / 0.9)
            A2 = D2 @ Bm; ws = smooth / (_C["QD"][j] / 0.9) ** 2; Hj = H + 2 * ws * A2.T @ A2; gs = gs + 2 * ws * A2.T @ c2[:, j]
        args.append((j, Hj, -2 * _TW * RB.T @ (Qt[:, j] - r0[:, j]) + gs, A, np.r_[lba, kv], np.r_[uba, kv], np.full(Mc, _C["QLO"][j]), np.full(Mc, _C["QHI"][j]), np.full(Mc, hist[-1, j])))
    res = pool().map(_one, args, chunksize=1); C = np.stack([r[0] for r in res], 1); U = Bm @ C; Qr = Rm @ U + r0
    Qk = [wp @ Qr[idx] for idx, wp, wv, q, v in kinfo]; Vk = [wv @ Qr[idx] for idx, wp, wv, q, v in kinfo]      # realized position / velocity at the key points
    return U, dict(ok=all(r[1] for r in res), err_rad=float(np.abs(Qr - Qt).max()), Qr=Qr, Qk=Qk, Vk=Vk, key_err=float(max([np.abs(a - k[3]).max() for a, k in zip(Qk, kinfo)] or [0.0])),
                   key_verr=float(max([np.abs(a - k[4]).max() for a, k in zip(Vk, kinfo)] or [0.0])))
