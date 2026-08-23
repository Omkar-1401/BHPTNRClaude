"""Stage-2 two-stage regression prototype for the peaks model.

Plan (from peaks_results/RESUME.md NEXT STEP):
  A. Per q in [3,8]: fit alpha(x)-1 and beta_abs(x)-1 as low-order x-polynomials
     to the measured peak point cloud.  Record per-q coeffs + per-q fit RMS.
  B. Regress each x-coefficient smoothly in nu, PP-anchored: coeff(nu) = nu*poly(nu)
     so every correction vanishes as nu->0 (q->inf, point-particle => alpha,beta->1).
  C. Reconstruct waveforms (abs-ratio map, merger phase-align + 1-D phi0 refine),
     in-range [3,8] and low-q extrapolation q=2.75..1.5.  Report mismatch.
     (No QNM ringdown branch yet -- x clipped to measured range, inspiral only.)

Goal: preserve the per-q method ceiling (q2 0.066%) through the regression.
The joint (x,nu) poly threw it away (q2 34%); does the 2-stage survive?
"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTNRSurrogate" / "surrogates"))
import fit_scaling_PN_opt_creative as creative
WAVE = ROOT / ".cache" / "q_dep"
PER_Q = ROOT / "peaks_results" / "per_q"

cases = {float(p.stem.replace("peaks_q", "")): dict(np.load(p))
         for p in sorted(PER_Q.glob("peaks_q*.npz"))}
QS = sorted(cases)
print(f"{len(QS)} q cases in [{QS[0]},{QS[-1]}]")


# ---------------------------------------------------------------- Stage A
def fit_xpoly(x, y, deg):
    """fit y-1 = sum_{m=0..deg} c_m x^m  (constant term allowed: alpha!=1 at merger)."""
    A = np.vstack([x ** m for m in range(deg + 1)]).T
    c, *_ = np.linalg.lstsq(A, y - 1.0, rcond=None)
    return c


def eval_xpoly(c, x):
    return 1.0 + sum(c[m] * x ** m for m in range(len(c)))


def stageA(deg_a, deg_b):
    """per-q coeffs + per-q fit RMS for alpha and beta_abs."""
    A, B, rmsA, rmsB, nus = [], [], [], [], []
    for q in QS:
        c = cases[q]
        x, al, be = c["x"], c["alpha"], c["beta_abs"]
        ca = fit_xpoly(x, al, deg_a)
        cb = fit_xpoly(x, be, deg_b)
        A.append(ca); B.append(cb)
        rmsA.append(np.sqrt(np.mean((al - eval_xpoly(ca, x)) ** 2)))
        rmsB.append(np.sqrt(np.mean((be - eval_xpoly(cb, x)) ** 2)))
        nus.append(float(c["nu"]))
    return (np.array(A), np.array(B), np.array(rmsA), np.array(rmsB),
            np.array(nus))


print("\n=== Stage A: per-q x-polynomial fit RMS (median over q) ===")
print(f"{'deg_a':>5} {'deg_b':>5} {'rmsA_med':>10} {'rmsA_max':>10} "
      f"{'rmsB_med':>10} {'rmsB_max':>10}")
for da, db in [(3, 2), (4, 2), (4, 3), (5, 3), (5, 4), (6, 4)]:
    A, B, rA, rB, nus = stageA(da, db)
    print(f"{da:5d} {db:5d} {np.median(rA):10.3e} {rA.max():10.3e} "
          f"{np.median(rB):10.3e} {rB.max():10.3e}")


# ---------------------------------------------------------------- Stage B
def regress_coeffs(C, nus, deg_nu):
    """Each column c_m(nu) = nu * poly_{deg_nu-1}(nu)  (PP-anchored: ->0 at nu=0).
    Returns list of nu-poly coeff arrays g[m] (length deg_nu), value = sum g_k nu^(k+1)."""
    Anu = np.vstack([nus ** k for k in range(1, deg_nu + 1)]).T
    G = []
    for m in range(C.shape[1]):
        g, *_ = np.linalg.lstsq(Anu, C[:, m], rcond=None)
        G.append(g)
    return G


def eval_coeffs(G, nu):
    """evaluate all x-coeffs at a given nu."""
    return np.array([sum(g[k] * nu ** (k + 1) for k in range(len(g))) for g in G])


# ---------------------------------------------------------------- recon
def load(q):
    d = np.load(WAVE / f"waveforms_q{q:.10f}.npz")
    return (d["t_bhpt"], d["h_bhpt_re"] + 1j * d["h_bhpt_im"],
            d["t_nr"], d["h_nr_re"] + 1j * d["h_nr_im"])


def recon(q, GA, GB, XLO, XHI):
    t_b, h_b, t_n, h_n = load(q)
    nu = q / (1 + q) ** 2
    psib = np.unwrap(np.angle(h_b))
    om = np.abs(np.gradient(psib, t_b)); xb = (om / 2) ** (2 / 3)
    m_b = int(np.argmax(np.abs(h_b))); m_n = int(np.argmax(np.abs(h_n)))
    trel = t_b - t_b[m_b]
    ca = eval_coeffs(GA, nu); cb = eval_coeffs(GB, nu)
    xc = np.clip(xb, XLO, XHI)
    a = eval_xpoly(ca, xc); b = eval_xpoly(cb, xc)
    tau = b * trel; tau_abs = tau + t_n[m_n]
    mono = np.r_[True, np.diff(tau) > 0]
    use = (tau_abs >= t_n[0]) & (tau_abs <= t_n[-1]) & mono
    phi0 = np.unwrap(np.angle(h_n))[m_n] - psib[m_b]

    def f(d):
        hs = a * np.exp(1j * (phi0 + d)) * h_b
        hi = creative.interp_complex(tau_abs[use], hs[use], t_n)
        cov = (t_n >= tau_abs[use][0]) & (t_n <= tau_abs[use][-1])
        return creative.mathcalE_error(h_n[cov], hi[cov]), np.mean(cov)
    r = minimize_scalar(lambda d: f(d)[0], bounds=(-np.pi, np.pi), method="bounded")
    return f(r.x)


# global x range for clipping
x_all = np.concatenate([cases[q]["x"] for q in QS])
XLO, XHI = x_all.min(), x_all.max()

print(f"\nfit x-range [{XLO:.3f},{XHI:.3f}]")
print("\n=== Stage B bake-off + reconstruction ===")
DEG_A, DEG_B = 5, 3           # x-degrees chosen from Stage A table
A, B, rA, rB, nus = stageA(DEG_A, DEG_B)
print(f"using deg_a={DEG_A} deg_b={DEG_B}  (ceiling: per-q x-fit "
      f"rmsA_med={np.median(rA):.2e}, rmsB_med={np.median(rB):.2e})")

for dnu_a, dnu_b in [(3, 3), (4, 3), (4, 4), (5, 4)]:
    GA = regress_coeffs(A, nus, dnu_a)
    GB = regress_coeffs(B, nus, dnu_b)
    ir = []
    for q in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e, cov = recon(q, GA, GB, XLO, XHI)
        ir.append(e)
    lo = {}
    for q in [2.75, 2.5, 2.0, 1.5]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e, cov = recon(q, GA, GB, XLO, XHI)
        lo[q] = e
    print(f"\n dnu_a={dnu_a} dnu_b={dnu_b}: in-range[3,8] median={np.median(ir):.3e} "
          f"max={np.max(ir):.3e}")
    print(f"   low-q: q2.75={100*lo[2.75]:.3f}%  q2.5={100*lo[2.5]:.3f}%  "
          f"q2.0={100*lo[2.0]:.3f}%  q1.5={100*lo[1.5]:.3f}%  "
          f"{'[q2 GATE OK]' if lo[2.0] < 0.01 else '[q2 FAIL >1%]'}")
