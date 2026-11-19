"""mk_keys_belt (2026-11-16, user: belt speeds 0.10, 0.14, 0.16, 0.18, 0.20 — 5 cm library and networks as for 0.12):
the cells of mk_keys_ring.py (whole reach ring, 5 cm) and the random off-grid tasks of mk_keys_rand2.py for a belt speed V.
Grid: A at the cycle start ya = 0.05 i, B's contact yc = 0.05 j, key yb = yc + SHIFT with SHIFT = V x 0.6 s rounded to 1 cm (0.07 m at 0.12 m/s, the value
of the 0.12 library: B is about one cycle upstream of its contact when the cycle starts).  Same selection as mk_keys_ring.py: A inside the ring (one extra row
downstream), B's contact inside the ring (half a step of slack downstream), A and B at least 7.5 cm apart.  Cells in random order (seed 0), as a6_5cm.sh.
Random tasks: A's and B's contact uniform inside the ring of their lanes; key = positions at the cycle start: A 0.12 s after its contact (V x 0.12),
B 0.45-0.80 s before its contact (V x U(0.45, 0.80)); seed 12 as mk_keys_rand2.py.
usage: python mk_keys_belt.py V GRID_OUT.txt RAND_OUT.txt N_RAND"""
import sys, json, random
import numpy as np
V = float(sys.argv[1]); SHIFT = round(V * 0.6, 2); R = json.load(open("logs/reach_r.json")); yr = lambda x: float(np.sqrt(max(R["r_hi"] ** 2 - x * x, 0.0)))
XS = [round(0.25 + 0.05 * i, 2) for i in range(8)]; YS = [round(-0.70 + 0.05 * i, 2) + 0.0 for i in range(29)]; K = []; nA = set(); nB = set()
for b in (0, 1, 2):
    for xa in XS:
        for ya in YS:
            if not (-yr(xa) - 0.04 <= ya <= yr(xa) + 1e-6):
                continue
            nA.add((xa, ya))
            for xb in XS:
                for yc in YS:
                    yb = round(yc + SHIFT, 2)
                    if not (-yr(xb) - 0.025 <= yc <= yr(xb) + 1e-6) or np.hypot(xa - xb, yb - ya) < 0.075:
                        continue
                    nB.add((xb, yc)); K.append("%.2f,%.2f,%.2f,%.2f,%d" % (xa, ya, xb, yb, b))
n = len(K); random.Random(0).shuffle(K); open(sys.argv[2], "w").write("".join(k + "\n" for k in K))
rng = np.random.default_rng(12); Q = []
while len(Q) < int(sys.argv[4]):
    xa, xb = rng.uniform(0.25, 0.595, 2); ya = rng.uniform(-yr(xa), yr(xa)) - V * 0.12; yb = rng.uniform(-yr(xb), yr(xb)) + V * rng.uniform(0.45, 0.80)
    if np.hypot(xa - xb, ya - yb) >= 0.08:
        Q.append("%.4f,%.4f,%.4f,%.4f,%d" % (xa, ya, xb, yb, rng.integers(0, 3)))
open(sys.argv[3], "w").write("\n".join(Q) + "\n")
print("belt %.2f: shift %.2f m, A positions %d, B contact positions %d, cells %d, random tasks %d" % (V, SHIFT, len(nA), len(nB), n, len(Q)))
