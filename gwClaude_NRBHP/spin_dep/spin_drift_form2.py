"""Well-conditioned version: what IS the form of the spin correction to beta?

x and x^2 are near-degenerate over x in [0.05,0.10], so db1/db2 are individually
meaningless.  Use instead two robust, directly-measured observables:

    D(x*) = R_beta(x*) - 1     at x* = 0.06 (early) and 0.095 (late)

and ask, for each:  how does D scale with nu?  how does it depend on chi1?
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "spin_measure_results" / "curves_win20.json").read_text())
QS = [3.0, 4.0, 5.0, 6.0, 8.0]; CHIS = [-0.6, -0.3, 0.3, 0.6, 0.8]
nu_of = lambda q: q / (1 + q) ** 2
XLO, XHI = 0.06, 0.095


def Rb(q, chi, xstar):
    b, d = data[f"{q:.1f}_+0.00"], data[f"{q:.1f}_{chi:+.2f}"]
    xb, yb = np.array(b["x"]), np.array(b["beta"]); ob = np.argsort(xb)
    xd, yd = np.array(d["x"]), np.array(d["beta"]); od = np.argsort(xd)
    return float(np.interp(xstar, xd[od], yd[od]) / np.interp(xstar, xb[ob], yb[ob]))


print("condition number of [x, x^2] over the band:",
      f"{np.linalg.cond(np.vstack([(g:=np.linspace(0.05,0.10,200)), g**2]).T):.1f}"
      "  <- why db1/db2 split is unstable\n")

print("nu-SCALING:  fit D(x*) = K(chi) * nu^p  across q,  per chi1")
print(f"{'chi1':>6} | {'x*=0.060':>26} | {'x*=0.095':>26}")
print(f"{'':>6} | {'p':>7} {'K':>9} {'fit r':>7} | {'p':>7} {'K':>9} {'fit r':>7}")
nus = np.array([nu_of(q) for q in QS])
P = {}
for c in CHIS:
    row = []
    for xs in (XLO, XHI):
        D = np.array([Rb(q, c, xs) - 1.0 for q in QS])
        s = np.sign(np.mean(D))
        good = (s * D) > 1e-6
        if good.sum() >= 3:
            co = np.polyfit(np.log(nus[good]), np.log(s * D[good]), 1)
            pred = np.exp(np.polyval(co, np.log(nus[good])))
            r = np.corrcoef(np.log(s*D[good]), np.log(nus[good]))[0, 1]
            row.append((co[0], s * np.exp(co[1]), r))
        else:
            row.append((np.nan, np.nan, np.nan))
    P[c] = row
    print(f"{c:>+6.2f} | {row[0][0]:7.2f} {row[0][1]:9.3f} {row[0][2]:7.3f} | "
          f"{row[1][0]:7.2f} {row[1][1]:9.3f} {row[1][2]:7.3f}")

pl = np.nanmean([P[c][0][0] for c in CHIS]); ph = np.nanmean([P[c][1][0] for c in CHIS])
print(f"\nmean exponent p:  early {pl:.2f}   late {ph:.2f}   "
      f"(p=1 would mean the spin term rides nu exactly, as pn_anchored's nu*(...) assumes)")

print(f"\nchi1-DEPENDENCE at fixed nu-scaling: D/nu^p vs chi1  (p fixed at the mean)")
for xs, p, tag in ((XLO, pl, "x*=0.060"), (XHI, ph, "x*=0.095")):
    print(f"\n  {tag}:  g(chi) = D / nu^{p:.2f}")
    print(f"{'chi1':>7} " + " ".join(f"{'q='+str(int(q)):>9}" for q in QS)
          + f" {'mean':>9} {'spread':>8}")
    G = {}
    for c in CHIS:
        v = np.array([(Rb(q, c, xs) - 1.0) / nu_of(q) ** p for q in QS])
        G[c] = np.mean(v)
        print(f"{c:>+7.2f} " + " ".join(f"{z:9.4f}" for z in v)
              + f" {np.mean(v):9.4f} {(v.max()-v.min())/abs(np.mean(v))*100:7.1f}%")
    cs = np.array(CHIS); g = np.array([G[c] for c in CHIS])
    V = np.vstack([cs, cs ** 2]).T
    co, *_ = np.linalg.lstsq(V, g, rcond=None)
    print(f"    -> g(chi) = {co[0]:+.4f} chi {co[1]:+.4f} chi^2   "
          f"rms {np.sqrt(np.mean((V@co-g)**2)):.4f}  (max |g| {np.abs(g).max():.4f})")
