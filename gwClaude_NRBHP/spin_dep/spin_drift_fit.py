"""Fit the measured spin correction to beta with the indicated form, all 25 (q,chi)
points x the clean x-band at once.

    R_beta(x;q,chi) - 1  =  nu^p [ (a1 + a2 x) chi + (a3 + a4 x) chi^2 ]

p free.  Compare against restricted variants to see what each ingredient buys.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "spin_measure_results" / "curves_win20.json").read_text())
QS = [3.0, 4.0, 5.0, 6.0, 8.0]; CHIS = [-0.6, -0.3, 0.3, 0.6, 0.8]
nu_of = lambda q: q / (1 + q) ** 2

X, NU, CHI, D = [], [], [], []
for q in QS:
    b = data[f"{q:.1f}_+0.00"]; xb, yb = np.array(b["x"]), np.array(b["beta"])
    ob = np.argsort(xb)
    for c in CHIS:
        d = data[f"{q:.1f}_{c:+.2f}"]; xd, yd = np.array(d["x"]), np.array(d["beta"])
        od = np.argsort(xd)
        xg = np.linspace(max(xb.min(), xd.min(), 0.05),
                         min(xb.max(), xd.max(), 0.10), 40)
        R = np.interp(xg, xd[od], yd[od]) / np.interp(xg, xb[ob], yb[ob])
        X.append(xg); NU.append(np.full_like(xg, nu_of(q)))
        CHI.append(np.full_like(xg, c)); D.append(R - 1.0)
X, NU, CHI, D = map(np.concatenate, (X, NU, CHI, D))
print(f"{len(D)} data points, |D| up to {np.abs(D).max():.4f}")

def model(th, full=True, pfix=None):
    if pfix is None: p, a = th[0], th[1:]
    else: p, a = pfix, th
    g = (a[0] + a[1]*X)*CHI + (a[2] + a[3]*X)*CHI**2 if full else a[0]*CHI + a[1]*CHI**2
    return NU**p * g

def run(label, full=True, pfix=None, n=None):
    n = n or (5 if full else 3)
    x0 = ([1.2] if pfix is None else []) + [0.1, 0.0, 0.05, 1.0][:n-(0 if pfix is None else 1)]
    r = least_squares(lambda th: model(th, full, pfix) - D, x0)
    rms = np.sqrt(np.mean(r.fun**2))
    print(f"{label:<42} rms {rms:.5f}  ({100*rms/np.abs(D).max():5.1f}% of max |D|)  "
          f"params {np.array2string(r.x, precision=3)}")
    return r

print()
run("full: nu^p[(a1+a2 x)chi + (a3+a4 x)chi^2]")
run("p fixed = 1 (pn_anchored's explicit nu)", pfix=1.0, n=5)
run("p fixed = 1.5", pfix=1.5, n=5)
run("no x-dependence: nu^p[a1 chi + a2 chi^2]", full=False)
run("no x-dep, p=1", full=False, pfix=1.0, n=3)
r = run("full (repeat, for the quoted numbers)")
p, a1, a2, a3, a4 = r.x
print(f"\nBEST FORM:  R_beta - 1 = nu^{p:.2f} [ ({a1:+.4f} {a2:+.3f} x) chi1 "
      f"+ ({a3:+.4f} {a4:+.3f} x) chi1^2 ]")
print("evaluated:")
print(f"{'q':>4} {'chi1':>6} {'x':>6} {'measured':>10} {'model':>10} {'diff':>9}")
for q in (3.0, 8.0):
    for c in (-0.6, 0.8):
        for xs in (0.06, 0.09):
            m = (np.abs(NU - nu_of(q)) < 1e-9) & (np.abs(CHI - c) < 1e-9)
            i = np.argmin(np.abs(X[m] - xs))
            dm, mm = D[m][i], model(r.x)[m][i]
            print(f"{q:>4g} {c:>+6.2f} {xs:>6.3f} {dm:>10.4f} {mm:>10.4f} {dm-mm:>+9.4f}")
