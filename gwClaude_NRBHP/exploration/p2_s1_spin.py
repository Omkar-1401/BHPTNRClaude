"""EXPLORATION pass 2 -- S1 with spin, on T96, inspiral error E_I.

Binary chirp on the M clock, built from the ppBHPT's OWN (Kerr chi1, exact 0PA) chirp:

    wdot_NR(w) = nu * [Omegadot_pp(w)/eps] * [1 + nu F1(w)/F0(w)] * [1 - chi1 D x^(3/2)]
    eps = 1/q;  F0, F1 = Schwarzschild 1PAT1 (sf1pa.py);  x = w^(2/3)
    D = [113 (X1^2 - 1) + 75 nu] / 12      (1.5PN spin-orbit: binary (113 X1^2 + 75 nu)/12 minus test mass 113/12)

i.e. Kuechler's eps -> nu, m1 -> M identification of the 0PA chirp, the Schwarzschild 1PA correction, and the
spin-orbit difference between the binary and the test-mass limit.  At leading order this reproduces the
installed kappa_SO factor exactly (derivation in EXPLORATION Log, pass 2).  beta then follows from the
equal-phase map ODE of pass 1 (p1_s1_chi0.map_beta, generalised).  Zero parameters.

Arms on E_I (NR [-5000.1, peak - 150 M], own t0 / phi0), all 96 T96 cases:
  L    pn_anchored_spin, full installed model
  F    flux_anc_spin_with_orbit, full installed model
  S1   alpha = pn_anchored_spin alpha_insp (no spin), beta = map  (0 parameters)
  S1c  S1 * (1 + c nu x^(5/2)), one global c
  per-case best c (diagnostic: is the gauge-like term universal?)
Chirp check: measured NR omega^2/omegadot over the constructed wdot_NR, at x = 0.07..0.13.
Writes exploration/p2/p2_s1_spin.json.
"""
from __future__ import annotations
import json, sys, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO, REPO / "spin_dep_split"):
    sys.path.insert(0, str(p))
import sf1pa                                   # noqa: E402
import p1_s1_chi0 as P1                        # noqa: E402
import fit_pn_anchored_spin as S               # noqa: E402
import fit_flux_anc_spin_with_orbit as FO      # noqa: E402
import fit_fluxanchored_spin as FS             # noqa: E402
import test_guard                              # noqa: E402,F401  (refuses sealed test points)
import physics_common as PC                    # noqa: E402
import ot_join as OT                           # noqa: E402

OUT = HERE / "p2"; OUT.mkdir(exist_ok=True)
X_LO, X_HI = 1 / 30, 0.145


def map_beta_spin(q, chi1, t, h):
    nu, eps, X1 = q / (1 + q) ** 2, 1.0 / q, q / (1 + q)
    chirp_pp, wgrid = P1.pp_chirp(t, h)
    D = (113 * (X1 ** 2 - 1) + 75 * nu) / 12
    def wdot(w):
        r0 = w ** (-2 / 3)
        return nu * (chirp_pp(w) / eps) * (1 + nu * sf1pa.F1(r0) / sf1pa.F0(r0)) * (1 - chi1 * D * w)
    rhs = lambda w, Om: [(w / wdot(w)) / (Om[0] / chirp_pp(Om[0]))]
    yN = X1 ** 1.2
    _, Om_isco = OT.kerr_isco(chi1)
    x_hi = min(X_HI, 0.9 * Om_isco ** (2 / 3), (0.97 * wgrid[-1]) ** (2 / 3))
    w_lo = max(X_LO ** 1.5, 1.1 * wgrid[0] / yN)
    w_hi = x_hi ** 1.5
    g = lambda y: rhs(w_lo, [y * w_lo])[0] - y
    y0 = brentq(g, 1.0001 * wgrid[0] / w_lo, 1.5 * yN)
    sol = solve_ivp(rhs, (w_lo, w_hi), [y0 * w_lo], rtol=1e-10, atol=1e-14, dense_output=True,
                    max_step=(w_hi - w_lo) / 400)
    W = np.linspace(w_lo, w_hi, 2000); Om = sol.sol(W)[0]
    w_pp, _ = P1.orb(t, h, 20.0)
    b = np.interp(w_pp, Om, Om / W, left=(Om / W)[0], right=(Om / W)[-1])
    return b, wdot, float(y0 / yN), float(x_hi)


def one(k):
    q, chi1 = k
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    th9 = S.model9(); ths = np.array(json.loads(S.COEFFS.read_text())["theta_spin"], float)
    thF = FO.load_theta()
    case = FS.build_case(q, chi1)                    # = S.build_case + loss drives
    t, h = case["t_bhpt"], case["h_bhpt"]
    aL, bL = S.model_ab(th9, ths, q, chi1, case["x"], case["p_loss"])
    aF, bF = FO.model_ab(thF, q, chi1, case, "full")
    xc = PC.x_scale(q) * np.minimum(case["x"], S.XCLIP)
    a2 = th9[2]
    a_ins = X1 ** 1.2 * (1 + S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
    bS, wdot, y0r, x_hi = map_beta_spin(q, chi1, t, h)
    gauge = nu * xc ** 2.5
    E = dict(E_L=P1.score_I(aL, bL, case), E_F=P1.score_I(aF, bF, case), E_S1=P1.score_I(a_ins, bS, case))
    r = minimize_scalar(lambda c: P1.score_I(a_ins, bS * (1 + c * gauge), case), bounds=(-150, 250),
                        method="bounded", options={"xatol": 0.05})
    E["c_best"], E["E_S1_cbest"] = float(r.x), float(r.fun)
    # chirp check on the NR side
    wn, wnd = P1.orb(case["t_nr"], case["h_nr"], 50.0)
    inr = int(np.argmax(np.abs(case["h_nr"])))
    xs = np.array([0.07, 0.09, 0.11, 0.13]); Wn = xs ** 1.5
    mm = (np.arange(len(wn)) < inr) & (wnd > 0)
    o = np.argsort(wn[mm])
    meas = np.interp(Wn, wn[mm][o], wnd[mm][o])
    E["chirp_meas_over_model"] = [float(v) if xx <= x_hi else None for v, xx in zip(meas / wdot(Wn), xs)]
    return dict(q=q, chi1=chi1, y0_over_X1_65=y0r, x_hi=x_hi, **E), (a_ins, bS, gauge)


def stats(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), max=float(v.max()))


def main():
    keys = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
    with mp.get_context("fork").Pool(96) as pool:
        res = pool.map(one, keys)
    rows = [r[0] for r in res]
    arrs = {(r[0]["q"], r[0]["chi1"]): r[1] for r in res}
    cases = None
    # one global c, scored in parallel
    def tot(c):
        with mp.get_context("fork").Pool(96) as pool:
            v = pool.map(_sc, [(k, c) for k in keys])
        return float(np.mean(v))
    global _ARRS
    _ARRS = arrs
    rc = minimize_scalar(tot, bounds=(-100, 200), method="bounded", options={"xatol": 0.2})
    c = float(rc.x)
    with mp.get_context("fork").Pool(96) as pool:
        v = pool.map(_sc, [(k, c) for k in keys])
    for r, e in zip(rows, v):
        r["E_S1c"] = e
    out = dict(c_global=c, summary={k: stats([r[k] for r in rows]) for k in ("E_L", "E_F", "E_S1", "E_S1c", "E_S1_cbest")},
               cases=rows)
    (OUT / "p2_s1_spin.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1)); print("c_global", c)
    print("per-chi1 medians: chi  E_L  E_F  E_S1  E_S1c  | median c_best | chirp meas/model (x=.07,.09,.11,.13) median over q")
    for ch in S.CHIS:
        rr = [r for r in rows if r["chi1"] == ch]
        cm = np.array([[np.nan if v is None else v for v in r["chirp_meas_over_model"]] for r in rr])
        print(f"{ch:+.1f}  " + "  ".join(f"{np.median([r[k] for r in rr]):.2e}" for k in ("E_L", "E_F", "E_S1", "E_S1c"))
              + f"  | {np.median([r['c_best'] for r in rr]):6.1f} | {np.round(np.nanmedian(cm, 0), 3)}")
    print("per-q medians of c_best:", [round(float(np.median([r['c_best'] for r in rows if r['q'] == q])), 1) for q in S.TRAIN_Q])


_ARRS = {}


def _sc(a):
    k, c = a
    a_ins, bS, gauge = _ARRS[k]
    return P1.score_I(a_ins, bS * (1 + c * gauge), FS.build_case(*k))


if __name__ == "__main__":
    main()
