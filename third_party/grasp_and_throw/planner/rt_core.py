"""rt_core: helper pieces for the sim runner — script server client, piece sampling, planned-realized buffer on the 2 ms sample grid."""
import os, sys, json, subprocess, threading
import numpy as np
DT = 0.002; HERE = os.path.dirname(os.path.abspath(__file__))


class Srv:
    def __init__(self, py):
        self.p = subprocess.Popen([py, "-W", "ignore", os.path.join(HERE, "srv.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=HERE); self._read()

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("srv ended")
            if line.startswith("{"):
                return json.loads(line)

    def call(self, script, argv, env=None, unset=()):
        self.p.stdin.write(json.dumps(dict(script=script, argv=argv, env=env or {}, unset=list(unset))) + "\n"); self.p.stdin.flush(); return self._read()

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(5)
        except Exception:  # noqa: BLE001
            self.p.kill()


def sample(P, ts):
    """Piece P (tt, Q, V, A, J: constant jerk per interval) at local times ts → (q, v)."""
    tt = P["tt"]; m = np.clip(np.searchsorted(tt, ts, side="right") - 1, 0, len(tt) - 2); s = (ts - tt[m])[:, None]
    return P["Q"][m] + P["V"][m] * s + P["A"][m] * s ** 2 / 2 + P["J"][m] * s ** 3 / 6, P["V"][m] + P["A"][m] * s + P["J"][m] * s ** 2 / 2


class Buf:
    """Planned realized joint positions per sample (index k - k0), filled piece by piece; gaps are held at the last pose."""

    def __init__(self, k0, q):
        self.k0 = int(k0); self.Q = [np.asarray(q, float)]; self.keys = []; self.lock = threading.RLock()

    @property
    def k_end(self):
        return self.k0 + len(self.Q) - 1

    def hold(self, k):
        self.Q.extend([self.Q[-1]] * max(0, int(k) - self.k_end))

    def add(self, t_abs, P, t_cut=None):
        """Writes the piece on samples floor(t_abs/DT)+1 .. floor((t_abs+T)/DT). Samples that were only held there (filled by rows() before the piece was known) are overwritten."""
        T = float(P["tt"][-1] if t_cut is None else t_cut); ka = int(np.floor(t_abs / DT)); ks = np.arange(ka + 1, int(np.floor((t_abs + T) / DT)) + 1)
        with self.lock:
            self.hold(ka)
            if len(ks):
                q, _ = sample(P, ks * DT - t_abs); i0 = ka + 1 - self.k0; self.Q[i0:i0 + len(ks)] = list(q)
        return t_abs + T

    def key(self, t_abs, q, v):
        self.keys.append((float(t_abs), np.asarray(q, float), np.asarray(v, float)))

    def rows(self, ka, n):
        with self.lock:
            self.hold(ka + n); return np.array(self.Q[ka - self.k0:ka - self.k0 + n])
