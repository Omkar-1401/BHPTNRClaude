"""What form does b_E(q,chi1) take?  MEASURE it from the step-1 curves.

pn_anchored inspiral:  beta = X1^(6/5) [1 + nu (b1 x + b2 x^2)]
If spin enters the DRIFT coefficients, b_k -> b_k + c_k(chi1), then

    (R_beta(x) - 1)/nu  ~  db1(chi1) x + db2(chi1) x^2     with NO constant term

Three testable claims: (i) no constant, (ii) db_k independent of q, (iii) quadratic in x
suffices.  Fits with and without a constant to test (i).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "spin_measure_results" / "curves_win20.json").read_text())
QS = [3.0, 4.0, 5.0, 6.0, 8.0]; CHIS = [-0.6, -0.3, 0.3, 0.6, 0.8]
nu_of = lambda q: q / (1 + q) ** 2
X0, X1B = 0.05, 0.10


def ratio(q, chi, key="beta"):
    b, d = data[f"{q:.1f}_+0.00"], data[f"{q:.1f}_{chi:+.2f}"]
    xb, yb = np.array(b["x"]), np.array(b[key]); ob = np.argsort(xb)
    xd, yd = np.array(d["x"]), np.array(d[key]); od = np.argsort(xd)
    lo = max(xb.min(), xd.min(), X0); hi = min(xb.max(), xd.max(), X1B)
    xg = np.linspace(lo, hi, 200)
    return xg, np.interp(xg, xd[od], yd[od]) / np.interp(xg, xb[ob], yb[ob])


print("TEST (i): is a constant term needed?  RMS residual of the fit to (R-1)/nu")
print(f"{'q':>4} {'chi1':>6} {'no const (db1,db2)':>22} {'rms':>9} | {'with const':>10} {'rms':>9}")
DB = {}
for q in QS:
    for c in CHIS:
        xg, R = ratio(q, c); y = (R - 1.0) / nu_of(q)
        V2 = np.vstack([xg, xg ** 2]).T
        p2, *_ = np.linalg.lstsq(V2, y, rcond=None); r2 = np.sqrt(np.mean((V2 @ p2 - y) ** 2))
        V3 = np.vstack([np.ones_like(xg), xg, xg ** 2]).T
        p3, *_ = np.linalg.lstsq(V3, y, rcond=None); r3 = np.sqrt(np.mean((V3 @ p3 - y) ** 2))
        DB[(q, c)] = p2
        print(f"{q:>4g} {c:>+6.2f}   {p2[0]:+8.3f} {p2[1]:+8.3f}   {r2:9.2e} | "
              f"c={p3[0]:+7.4f} {r3:9.2e}")

print("\nTEST (ii): are db1, db2 independent of q?  (PP-anchored drift form predicts yes)")
for k, nm in ((0, "db1"), (1, "db2")):
    print(f"\n  {nm}:")
    print(f"{'chi1':>7} " + " ".join(f"{'q='+str(int(q)):>9}" for q in QS) + f" {'spread':>9}")
    for c in CHIS:
        v = np.array([DB[(q, c)][k] for q in QS])
        sp = (v.max() - v.min()) / abs(v.mean()) * 100 if abs(v.mean()) > 1e-9 else np.nan
        print(f"{c:>+7.2f} " + " ".join(f"{x:9.3f}" for x in v) + f" {sp:8.1f}%")

print("\nTEST (iii): fit db_k(chi1) = c_k1 chi + c_k2 chi^2  (q-averaged; c_k(0)=0 by form)")
for k, nm in ((0, "db1"), (1, "db2")):
    y = np.array([np.mean([DB[(q, c)][k] for q in QS]) for c in CHIS])
    V = np.vstack([np.array(CHIS), np.array(CHIS) ** 2]).T
    p, *_ = np.linalg.lstsq(V, y, rcond=None)
    pred = V @ p
    print(f"  {nm}(chi) = {p[0]:+.4f} chi {p[1]:+.4f} chi^2   "
          f"rms {np.sqrt(np.mean((pred-y)**2)):.4f}")
    print(f"      measured {np.array2string(y, precision=3)}")
    print(f"      fitted   {np.array2string(pred, precision=3)}")
