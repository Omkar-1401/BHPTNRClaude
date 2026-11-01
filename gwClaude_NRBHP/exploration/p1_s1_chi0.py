"""EXPLORATION pass 1 (S1) -- zero-parameter 1PA inspiral beta, chi1 = 0, all 12 T96 q.

Equal-phase map between the ppBHPT (m1 clock, its OWN measured chirp) and the binary
(M clock, 1PAT1 chirp of Wardell et al. 2023 via sf1pa.py):

    dOmega_pp/dw = [ w / wdot_NR(w) ] / [ Omega_pp / Omegadot_pp(Omega_pp) ]
    beta(t) = Omega_pp(t) / w(Omega_pp(t))           (orbital frequencies, each on its own clock)

    wdot_NR(w)          = nu F0(w) + nu^2 F1(w)              1PAT1, M = 1
    Omegadot_pp(Om_pp)  = measured from the ppBHPT (2,2) phase (exact for the input)

The integration constant (one per case) is the gauge mode c x^(5/2) of an equal-phase map; it is
fixed by starting on the local fixed point (dy/dw = 0, y = Omega_pp/w) at x_NR = 1/30.

Arms, all scored on the INSPIRAL error E_I (NR window [-5000.1, t_s], t_s = -150 M before the NR
peak; t0, phi0 optimised on that window alone -- alignment (c) of EXPLORATION 1.2):
  L    leader inspiral branch: alpha, beta = pn_anchored_spin with S = 0 (b1, b2, a2 fitted)
  S1   beta from the map with 1PAT1 (0 parameters), alpha = leader alpha_insp
  S0   beta from the map with 0PA only (F1 = 0) -- shows what the 2SF/first-law term buys
  S1c  S1 plus one global gauge coefficient c (beta *= 1 + c nu x^(5/2)), fitted over the 12 cases
Also a gauge-free check of both chirps (omega^2/omegadot vs omega): measured NR against 1PAT1,
measured ppBHPT against 0PA with eps = 1/q.

Writes exploration/p1/p1_s1_chi0.json.
"""
from __future__ import annotations
import json, sys, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "spin_dep")); sys.path.insert(0, str(REPO))
import sf1pa                                   # noqa: E402
import fit_pn_anchored_spin as S               # noqa: E402
import test_guard                              # noqa: E402,F401  (refuses sealed test points)
import physics_common as PC                    # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p1"; OUT.mkdir(exist_ok=True)
TS = -150.0                    # inspiral window end, NR time relative to the NR peak
X_LO, X_HI = 1 / 30, 0.145     # 1PAT1 data range (WaSABI stop condition ~ x = 0.146 at nu = 0.19)


def orb(t, h, win):
    dt = float(np.median(np.diff(t)))
    n = int(round(win / dt)); n += (n % 2 == 0)
    ph = np.unwrap(np.angle(h)); ph = ph if ph[-1] > ph[0] else -ph
    w = savgol_filter(ph, n, 3, deriv=1, delta=dt, mode="interp") / 2
    wd = savgol_filter(ph, n, 3, deriv=2, delta=dt, mode="interp") / 2
    return w, wd


def pp_chirp(t, h):
    """Omegadot_pp as a function of Omega_pp (m1 units), inspiral part, monotone."""
    w, wd = orb(t, h, 200.0)
    ip = int(np.argmax(np.abs(h)))
    m = (np.arange(len(t)) < ip) & (wd > 0)
    w, wd = w[m], wd[m]
    keep = np.r_[True, np.diff(w) > 0]
    w, wd = w[keep], wd[keep]
    return lambda Om: np.interp(Om, w, wd), w


def map_beta(q, t, h, with_F1=True):
    nu, eps = q / (1 + q) ** 2, 1.0 / q
    chirp_pp, wgrid = pp_chirp(t, h)
    def wdot(w):
        r0 = w ** (-2 / 3)
        return nu * sf1pa.F0(r0) + (nu ** 2 * sf1pa.F1(r0) if with_F1 else 0.0)
    rhs = lambda w, Om: [(w / wdot(w)) / (Om[0] / chirp_pp(Om[0]))]
    yN = (q / (1 + q)) ** 1.2
    w_lo = max(X_LO ** 1.5, 1.1 * wgrid[0] / yN)          # start inside the ppBHPT data
    w_hi = X_HI ** 1.5
    # local fixed point at w_lo: y such that rhs(w_lo, y w_lo) = y
    from scipy.optimize import brentq
    g = lambda y: rhs(w_lo, [y * w_lo])[0] - y
    y0 = brentq(g, 1.0001 * wgrid[0] / w_lo, 1.5 * yN)
    sol = solve_ivp(rhs, (w_lo, w_hi), [y0 * w_lo], rtol=1e-10, atol=1e-14, dense_output=True,
                    max_step=(w_hi - w_lo) / 400)
    W = np.linspace(w_lo, w_hi, 2000); Om = sol.sol(W)[0]
    # beta as a function of Omega_pp; frozen outside the covered band
    w_pp, _ = orb(t, h, 20.0)
    b = np.interp(w_pp, Om, Om / W, left=(Om / W)[0], right=(Om / W)[-1])
    return b, float(y0), float(Om[-1])


def score_I(alpha, beta, case, seed=-50.0):
    """Inspiral error on NR [-5000.1, t_peak + TS], own t0 / phi0."""
    t, h, tn, hn = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]
    inr = int(np.argmax(np.abs(hn)))
    m = tn <= tn[inr] + TS
    tn, hn = tn[m][::5], hn[m][::5]
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
    f = lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn)
    grid = np.linspace(seed - 250, seed + 250, 101)
    c = grid[int(np.argmin([f(g) for g in grid]))]
    r = minimize_scalar(f, bounds=(c - 6, c + 6), method="bounded", options={"xatol": 1e-3})
    return float(r.fun)


def one(q):
    th9 = S.model9(); ths = np.array(json.loads(S.COEFFS.read_text())["theta_spin"], float)
    case = S.build_case(q, 0.0)
    t, h = case["t_bhpt"], case["h_bhpt"]
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    b1, b2, a2 = th9[:3]
    xc = PC.x_scale(q) * np.minimum(case["x"], S.XCLIP)
    base = X1 ** 1.2
    aL = base * (1 + S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
    bL = base * (1 + nu * (b1 * xc + b2 * xc ** 2))
    bS1, y0, _ = map_beta(q, t, h, True)
    bS0, y00, _ = map_beta(q, t, h, False)
    # gauge-free chirp checks: omega^2/omegadot (orbital) vs omega
    wn, wnd = orb(case["t_nr"], case["h_nr"], 50.0)
    inr = int(np.argmax(np.abs(case["h_nr"])))
    xs = np.array([0.05, 0.07, 0.09, 0.11, 0.13])
    Wn = xs ** 1.5
    mm = (np.arange(len(wn)) < inr) & (wnd > 0)
    o = np.argsort(wn[mm]); meas_nr = np.interp(Wn, wn[mm][o], (wn[mm] ** 2 / wnd[mm])[o])
    th_nr = Wn ** 2 / sf1pa.chirp_1pa(Wn, nu)
    chirp_pp, _ = pp_chirp(t, h)
    th_pp = Wn ** 2 / ((1 / q) * sf1pa.F0(Wn ** (-2 / 3)))
    meas_pp = Wn ** 2 / chirp_pp(Wn)
    xsc = PC.x_scale(q) * np.minimum(case["x"], S.XCLIP)
    gauge = nu * xsc ** 2.5
    return dict(q=q, y0_S1=y0, y0_S0=y00, X1_65=base,
                E_L=score_I(aL, bL, case), E_S1=score_I(aL, bS1, case), E_S0=score_I(aL, bS0, case),
                chirp_nr_meas_over_1PAT1=list(meas_nr / th_nr), chirp_pp_meas_over_0PA=list(meas_pp / th_pp),
                _arrays=(aL, bS1, gauge))


def main():
    qs = list(S.TRAIN_Q)
    with mp.get_context("fork").Pool(len(qs)) as pool:
        res = pool.map(one, qs)
    # S1c: one global gauge coefficient
    cases = {q: S.build_case(q, 0.0) for q in qs}
    def tot(c):
        return float(np.mean([score_I(r["_arrays"][0], r["_arrays"][1] * (1 + c * r["_arrays"][2]), cases[r["q"]]) for r in res]))
    rc = minimize_scalar(tot, bounds=(-60, 60), method="bounded", options={"xatol": 1e-2})
    c = float(rc.x)
    for r in res:
        r["E_S1c"] = score_I(r["_arrays"][0], r["_arrays"][1] * (1 + c * r["_arrays"][2]), cases[r["q"]])
        del r["_arrays"]
    summ = {k: dict(mean=float(np.mean([r[k] for r in res])), median=float(np.median([r[k] for r in res])),
                    max=float(np.max([r[k] for r in res]))) for k in ("E_L", "E_S1", "E_S0", "E_S1c")}
    out = dict(TS=TS, c_gauge=c, summary=summ, cases=res)
    (OUT / "p1_s1_chi0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1)); print("c_gauge", c)
    for r in res:
        print(f"q={r['q']:.3f}  E_L {r['E_L']:.3e}  E_S1 {r['E_S1']:.3e}  E_S0 {r['E_S0']:.3e}  E_S1c {r['E_S1c']:.3e}  "
              f"y0/X1^1.2 {r['y0_S1']/r['X1_65']:.4f}  NRchirp/1PAT1 {np.round(r['chirp_nr_meas_over_1PAT1'],3)}  "
              f"ppchirp/0PA {np.round(r['chirp_pp_meas_over_0PA'],3)}")


if __name__ == "__main__":
    main()
