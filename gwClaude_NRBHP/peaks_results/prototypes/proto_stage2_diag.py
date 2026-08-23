"""Diagnose why the 2-stage regression blows up at q=2.
We HAVE measured alpha,beta at q=2 (proto_diag2 ceiling 0.066%).  Compare the
regressed alpha(x;nu),beta(x;nu) against the measured q=2 point cloud, and inspect
how the per-q x-coefficients and the inspiral value alpha(x_lo) behave vs nu.
"""
import sys, warnings
from pathlib import Path
import numpy as np
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_peaks as peaks
PER_Q = ROOT / "peaks_results" / "per_q"
cases = {float(p.stem.replace("peaks_q","")): dict(np.load(p))
         for p in sorted(PER_Q.glob("peaks_q*.npz"))}
QS = sorted(cases)


def fit_xpoly(x, y, deg):
    A = np.vstack([x**m for m in range(deg+1)]).T
    c, *_ = np.linalg.lstsq(A, y-1.0, rcond=None); return c
def eval_xpoly(c, x): return 1.0 + sum(c[m]*x**m for m in range(len(c)))

# ---- inspiral value: does alpha(x->x_lo) track q/(1+q) ?
print("=== inspiral alpha vs mass-scale q/(1+q) ===")
print(f"{'q':>5} {'nu':>7} {'a(x_lo)':>8} {'q/(1+q)':>8} {'ratio':>7}   "
      f"{'b(x_lo)':>8}")
for q in [3.0, 4.0, 5.0, 6.0, 8.0]:
    c = cases[q]; o = np.argsort(c["x"])
    a_lo = c["alpha"][o][0]; b_lo = c["beta_abs"][o][0]
    ms = q/(1+q)
    print(f"{q:5.1f} {c['nu']:7.4f} {a_lo:8.4f} {ms:8.4f} {a_lo/ms:7.4f}   {b_lo:8.4f}")

# ---- per-q x-coeffs vs nu (deg 5 alpha, deg 3 beta)
DA, DB = 5, 3
A = np.array([fit_xpoly(cases[q]["x"], cases[q]["alpha"], DA) for q in QS])
B = np.array([fit_xpoly(cases[q]["x"], cases[q]["beta_abs"], DB) for q in QS])
nus = np.array([cases[q]["nu"] for q in QS])
print(f"\n=== alpha x-coeffs vs nu (deg {DA}); columns a0..a{DA} ===")
print(" nu     " + " ".join(f"a{m:>9d}" for m in range(DA+1)))
for i in range(0, len(QS), 8):
    print(f"{nus[i]:.4f} " + " ".join(f"{A[i,m]:10.3f}" for m in range(DA+1)))
print(f"\n=== beta_abs x-coeffs vs nu (deg {DB}); columns b0..b{DB} ===")
print(" nu     " + " ".join(f"b{m:>9d}" for m in range(DB+1)))
for i in range(0, len(QS), 8):
    print(f"{nus[i]:.4f} " + " ".join(f"{B[i,m]:10.3f}" for m in range(DB+1)))

# ---- regress coeffs in nu (PP-anchored) and compare to MEASURED q=2
def regress(C, nus, dnu):
    Anu = np.vstack([nus**k for k in range(1, dnu+1)]).T
    return [np.linalg.lstsq(Anu, C[:,m], rcond=None)[0] for m in range(C.shape[1])]
def evalc(G, nu): return np.array([sum(g[k]*nu**(k+1) for k in range(len(g))) for g in G])

GA = regress(A, nus, 4); GB = regress(B, nus, 4)
x_all = np.concatenate([cases[q]["x"] for q in QS]); XLO, XHI = x_all.min(), x_all.max()

for q2 in [2.0, 2.5]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); m = peaks.extract(q2)
    nu2 = q2/(1+q2)**2; o = np.argsort(m["x"])
    xg = m["x"][o]; a_meas = m["alpha"][o]; b_meas = m["beta_abs"][o]
    ca = evalc(GA, nu2); cb = evalc(GB, nu2)
    a_reg = eval_xpoly(ca, np.clip(xg, XLO, XHI))
    b_reg = eval_xpoly(cb, np.clip(xg, XLO, XHI))
    print(f"\n=== q={q2} (nu={nu2:.4f}): regressed vs MEASURED ===")
    print(f"  x-range measured [{xg.min():.3f},{xg.max():.3f}]  (fit XHI={XHI:.3f})")
    print(f"  alpha: max|reg-meas|={np.max(np.abs(a_reg-a_meas)):.3e}  "
          f"a_meas[{a_meas[0]:.3f}->{a_meas[-1]:.3f}] a_reg[{a_reg[0]:.3f}->{a_reg[-1]:.3f}]")
    print(f"  beta : max|reg-meas|={np.max(np.abs(b_reg-b_meas)):.3e}  "
          f"b_meas[{b_meas[0]:.3f}->{b_meas[-1]:.3f}] b_reg[{b_reg[0]:.3f}->{b_reg[-1]:.3f}]")
