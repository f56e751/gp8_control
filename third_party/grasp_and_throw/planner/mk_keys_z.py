"""mk_keys_z (2026-11-18, user: object height as an input of the cycle-time model and of the order network):
random off-grid tasks as mk_keys_belt.py (belt 0.12: contact of A and of B uniform inside the ring of their lanes, A 0.12 s after its contact, B 0.45-0.80 s before its
contact, at least 8 cm apart, random bin) with the tops of A and B drawn uniformly in [ZLO, ZHI] = [-0.02, 0.05] m (the range of object tops of 11-02; the four
heights of the Isaac runs are -0.02, 0.0033, 0.0267, 0.05).  Key "xa,ya,xb,yb,b,zA,zB" (rn2.py takes the two tops as key[5], key[6]).
usage: python mk_keys_z.py OUT.txt N"""
import sys, json
import numpy as np
V = 0.12; ZLO, ZHI = -0.02, 0.05; N = int(sys.argv[2]); R = json.load(open("logs/reach_r.json")); yr = lambda x: float(np.sqrt(max(R["r_hi"] ** 2 - x * x, 0.0)))
rng = np.random.default_rng(2118); Q = []
while len(Q) < N:
    xa, xb = rng.uniform(0.25, 0.595, 2); ya = rng.uniform(-yr(xa), yr(xa)) - V * 0.12; yb = rng.uniform(-yr(xb), yr(xb)) + V * rng.uniform(0.45, 0.80); za, zb = rng.uniform(ZLO, ZHI, 2); b = rng.integers(0, 3)
    if np.hypot(xa - xb, ya - yb) >= 0.08:
        Q.append("%.4f,%.4f,%.4f,%.4f,%d,%.4f,%.4f" % (xa, ya, xb, yb, b, za, zb))
open(sys.argv[1], "w").write("\n".join(Q) + "\n"); print("random tasks with heights %d, tops %.3f..%.3f" % (len(Q), ZLO, ZHI))
