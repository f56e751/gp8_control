"""ordfull: the expert that follows EVERY order (user 11-12: "전부 따져서 신경망 학습 데이터로").
full_values(S, tm, a, ts, vis, ok): for each candidate ok[:K], take it first and then search all orders of the objects visible now, to the end
(at every cycle every candidate, up to K, is tried; no candidate -> wait point and restart, as in ordsim).  Value = (objects picked, end time), most objects first.
Level-by-level search over states (first candidate, object in hand, set of remaining objects), all states of a level expanded at once with numpy.
Of the orders that reach the same state, only the one that reaches it earliest is continued, and a state is dropped when even with the shortest cycle
for every remaining object it cannot beat the best order found so far (both rules checked against the plain search on small streams)."""
import numpy as np
import ordsim as O
T_MIN = 0.46                                                                   # below the shortest cycle of the library (0.485 s): only for the bound


def tm_batch(tm, xa, ya, xb, yb, b, za=None, zb=None):                          # 11-18: tops of A and B for a 9-input model
    n = len(tm.m); X = np.zeros((len(xa), n)); X[:, 0] = xa; X[:, 1] = ya; X[:, 2] = xb; X[:, 3] = yb; X[np.arange(len(xa)), 4 + b] = 1.0
    if n >= 9:
        X[:, 7] = O.ZTOP if za is None else za; X[:, 8] = O.ZTOP if zb is None else zb
    h = (X - tm.m) / tm.s
    for k, (w, bb) in enumerate(zip(tm.W, tm.b)):
        h = h @ w + bb
        if k < len(tm.W) - 1:
            h = np.tanh(h)
    return h[:, 0]


def full_values(S, tm, a, ts, vis, ok, cap=400000, stat=None, dedupe=True):
    c = ok[:O.K]; ids = sorted(vis, key=lambda j: S.y0[j]); n = len(ids); pos = {j: i for i, j in enumerate(ids)}; x = S.x[ids]; y0 = S.y0[ids]; z = S.z[ids]; bb = S.b[ids].astype(int); w = O.yr(x); bit = (1 << np.arange(n)).astype(np.int64)
    full = (1 << n) - 1; R = len(c); best_n = np.zeros(R, int); best_t = np.full(R, np.inf); nst = 0; hit = False
    inc = O.roll_values(S, tm, a, ts, vis, ok); best_n[:] = [v[0] for v in inc]; best_t[:] = [v[1] for v in inc]          # a known order (candidate, then FIFO): states that cannot beat it are dropped
    ia, ib = np.nonzero(~np.eye(n, dtype=bool)); tg = ts + np.arange(0.0, max(1.0, float(best_t.max() - ts)) + 0.5, 0.5); ya = y0[ia][None, :] + O.VY * tg[:, None]; yb_ = y0[ib][None, :] + O.VY * tg[:, None]
    inr = (np.abs(ya) <= w[ia][None, :] + 0.05) & (yb_ >= -w[ib][None, :] - 0.05) if n > 1 else np.zeros((len(tg), 0), bool); gi, pi = np.nonzero(inr)                                # shortest cycle between any two of these objects while both are near the reach
    T_MIN = max(0.46, float(tm_batch(tm, x[ia[pi]], ya[gi, pi], x[ib[pi]], yb_[gi, pi], bb[ia[pi]], z[ia[pi]], z[ib[pi]]).min()) - 0.01) if len(gi) else 0.46
    tout = (y0 + w) / (-O.VY); eo = np.argsort(tout)                                                                          # the time at which each object leaves the reach, and that order
    root = np.arange(R); cur = np.array([pos[j] for j, _T in c]); t = np.array([ts + T + O.TDW for _j, T in c]); mask = np.array([full & ~(1 << pos[j]) for j, _T in c], np.int64); level = 1

    def final(r, nn, tt):
        for i in range(len(r)):
            if nn > best_n[r[i]] or (nn == best_n[r[i]] and tt[i] < best_t[r[i]]):
                best_n[r[i]] = nn; best_t[r[i]] = tt[i]

    while len(root):
        nst += len(root); M = (mask[:, None] & bit[None, :]) != 0; yt = y0[None, :] + O.VY * t[:, None]; M &= yt >= -w[None, :]; mask = (M * bit[None, :]).sum(1).astype(np.int64)         # drop the objects that have left the reach
        e = ~M.any(1)
        if e.any():
            final(root[e], level, t[e])
        k = ~e; root, cur, t, mask, M, yt = root[k], cur[k], t[k], mask[k], M[k], yt[k]
        if not len(root):
            break
        cnt = np.zeros(len(root), int)                                                                                        # bound: how many more objects can still be contacted before they leave, at the shortest cycle (T_MIN) each
        for q in eo:
            cnt += M[:, q] & (t + T_MIN + cnt * (T_MIN + O.TDW) <= tout[q] + 1e-9)
        ub = level + cnt; kp = (ub > best_n[root]) | ((ub == best_n[root]) & (t + (best_n[root] - level) * (T_MIN + O.TDW) < best_t[root] - 1e-9))
        root, cur, t, mask, M, yt = root[kp], cur[kp], t[kp], mask[kp], M[kp], yt[kp]
        if not len(root):
            break
        si, oi = np.nonzero(M); T = np.zeros(M.shape); T[si, oi] = tm_batch(tm, x[cur[si]], y0[cur[si]] + O.VY * t[si], x[oi], yt[si, oi], bb[cur[si]], z[cur[si]], z[oi]); yc = yt + O.VY * T; okm = M & (yc >= -w[None, :]) & (yc <= w[None, :])
        okm &= np.cumsum(okm, 1) <= O.K                                                                                       # the K most downstream candidates
        nr, nc_, nt, nm = [], [], [], []
        no = ~okm.any(1)
        for i in np.nonzero(no)[0]:                                                                                           # no candidate: wait point, then the start rule
            t2 = t[i] + O.T_END; r2 = [ids[q] for q in np.nonzero(M[i])[0] if y0[q] + O.VY * t2 >= -w[q]]
            if not r2:
                final(root[i:i + 1], level, np.array([t2])); continue
            f = O.first(S, t2 + O.T_RESTART, [q for q in r2 if S.y0[q] + O.VY * (t2 + O.T_RESTART) >= -O.yr(S.x[q])], tm)
            if f is None:
                final(root[i:i + 1], level, np.array([t2])); continue
            m2 = 0
            for q in r2:
                if q != f[0]:
                    m2 |= 1 << pos[q]
            nr.append(root[i]); nc_.append(pos[f[0]]); nt.append(f[1] + O.TDW); nm.append(m2)
        si, oi = np.nonzero(okm)
        root = np.r_[root[si], np.array(nr, int)]; cur = np.r_[oi, np.array(nc_, int)]; t = np.r_[t[si] + T[si, oi] + O.TDW, np.array(nt, float)]; mask = np.r_[mask[si] & ~bit[oi], np.array(nm, np.int64)].astype(np.int64); level += 1
        if dedupe and len(root) > 1:                                                                                          # same (first candidate, object in hand, remaining set): keep the earliest
            key = root * 64 + cur; o = np.lexsort((t, mask, key)); keep = np.ones(len(o), bool); keep[1:] = (key[o][1:] != key[o][:-1]) | (mask[o][1:] != mask[o][:-1]); o = o[keep]; root, cur, t, mask = root[o], cur[o], t[o], mask[o]
        if len(root) > cap:                                                                                                   # too many states: keep the earliest ones
            hit = True; o = np.argsort(t)[:cap]; root, cur, t, mask = root[o], cur[o], t[o], mask[o]
    if stat is not None:
        stat.append((nst, hit))
    return [(int(best_n[i]), float(best_t[i])) for i in range(R)]


def choose_full(S, tm, a, ts, vis, ok, passed):
    v = full_values(S, tm, a, ts, vis, ok); return int(min(range(len(v)), key=lambda i: (-v[i][0], v[i][1])))
