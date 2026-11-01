"""EXPLORATION pass 5b (from p5_floor.py: bounded blend, all T96) -- per-case representation floors of candidate forms on S18 (no regression).

Each form is refitted independently per (q, chi1) on the full-window mathcalE (fit-stride arrays inside the
optimiser, full resolution for the reported number).  The floor says what a form can reach at all; the gap between
an installed model and its floor is the regression's price (fig10 study: flux_anc_spin_with_orbit is within
1.05-1.15x of its floor at q = 4, so only a new FORM can help it).

Forms (all on the ppBHPT clock t, s = t - t_peak,pp; x = X1^(-4/5) min(x_pp, 0.26); E, F the despiked drives):
  FLUX (4):  alpha = a_pp (1 + a_F F),  beta = (b_pp (1 + b_e E)) (1 + kappa_SO chi1 x^(3/2))      [= the installed form]
  M2   (8):  FLUX, then for s in [s1, s2] a C2 log-blend W(s) to the ringdown branch
             ln beta  -> ln beta_r + rho_b                 (beta_r exact Kerr / remnant)
             ln alpha -> ln alpha_r + kappa_RD beta_r (t - t1)   (B5 slope, zero parameters; alpha_r free)
             params: FLUX 4 + s1, s2, rho_b, ln alpha_r
Writes exploration/p5/p5_floor.json.
"""
from __future__ import annotations
import json, sys, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO):
    sys.path.insert(0, str(p))
import fit_pn_anchored_spin as S              # noqa: E402
import fit_flux_anc_spin_with_orbit as FO     # noqa: E402
import fit_fluxanchored_spin as FS            # noqa: E402
import test_guard                              # noqa: E402,F401  (refuses sealed test points)
import p3_rd as P3                            # noqa: E402  (anchors, smooth5)
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p5b"; OUT.mkdir(exist_ok=True)
S18 = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]


class Case:
    def __init__(self, k):
        self.k = k; q, chi1 = k
        self.case = FS.build_case(q, chi1)
        self.br, self.kap = P3.anchors(q, chi1)
        self.tpk = float(self.case["t_bhpt"][int(np.argmax(np.abs(self.case["h_bhpt"])))])
        self.ks = FO.kappa_so(q) * chi1
        self.t0 = None

    def arrays(self, key):
        pre = "fit_" if key == "fit" else ""
        c = self.case
        x = S.PC.x_scale(self.k[0]) * np.minimum(c[pre + "x"], S.XCLIP)
        return (c[pre + "t_bhpt"], c[pre + "h_bhpt"], c[pre + "t_nr"], c[pre + "h_nr"],
                c[pre + "losses"]["e_oft"], c[pre + "losses"]["flux_hat"], x)

    def ab(self, p, form, key):
        t, h, tn, hn, E, F, x = self.arrays(key)
        a_pp, a_F, b_pp, b_e = p[:4]
        alpha = a_pp * (1 + a_F * F)
        beta = b_pp * (1 + b_e * E) * (1 + self.ks * x ** 1.5)
        if form == "M2":
            s1, s2, rho, lar = p[4:8]
            if not (-80 <= s1 <= 25 and s1 + 3 <= s2 <= 70 and abs(rho) < 0.3):
                return None, None
            s = t - self.tpk
            W = P3.smooth5((s - s1) / max(s2 - s1, 1.0))
            t1 = self.tpk + s1
            lb_r = np.log(self.br) + rho
            la_r = lar + self.kap * self.br * np.clip(t - t1, 0, None)
            with np.errstate(invalid="ignore", divide="ignore"):
                beta = np.exp((1 - W) * np.log(np.clip(beta, 1e-6, None)) + W * lb_r)
                alpha = np.exp((1 - W) * np.log(np.clip(alpha, 1e-6, None)) + W * la_r)
        return alpha, beta

    def err(self, p, form, key="fit", wide=False):
        t, h, tn, hn, *_ = self.arrays(key)
        alpha, beta = self.ab(p, form, key)
        if alpha is None:
            return 1.0
        if np.any(beta <= 0) or np.any(alpha <= 0) or not np.all(np.isfinite(beta)):
            return 1.0
        bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
        if np.min(np.diff(ts)) <= 0:
            return 1.0
        f = lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn)
        if self.t0 is None or wide:
            grid = np.linspace(-300, 200, 126); c0 = grid[int(np.argmin([f(g) for g in grid]))]; w = 6
        else:
            c0, w = self.t0, 15
        r = minimize_scalar(f, bounds=(c0 - w, c0 + w), method="bounded", options={"xatol": 1e-3})
        self.t0 = float(r.x)
        return float(r.fun)


def fit_case(k):
    C = Case(k); q, chi1 = k
    a_pp, a_F, b_pp, b_e = FO.params_at(q, chi1, FO.load_theta())
    p_inst = np.array([a_pp, a_F, b_pp, b_e])
    out = dict(q=q, chi1=chi1)
    out["E_installed"] = C.err(p_inst, "FLUX", "full", wide=True)
    C.t0 = None; C.err(p_inst, "FLUX", "fit", wide=True)
    r = minimize(lambda p: C.err(p, "FLUX"), p_inst, method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 1500, "adaptive": True})
    pF = r.x
    out["FLUX"] = dict(p=list(map(float, pF)), E=C.err(pF, "FLUX", "full", wide=True))
    # M2 from the FLUX floor
    tpk_a = np.interp(C.tpk, C.case["t_bhpt"], C.ab(pF, "FLUX", "full")[0])
    best = None
    for s1, s2 in ((-10.0, 33.0), (0.0, 15.0), (-25.0, 10.0)):
        p0 = np.r_[pF, s1, s2, 0.0, np.log(tpk_a)]
        C.t0 = None; C.err(p0, "M2", "fit", wide=True)
        rr = minimize(lambda p: C.err(p, "M2"), p0, method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 3000, "adaptive": True})
        if best is None or rr.fun < best.fun:
            best = rr
    out["M2"] = dict(p=list(map(float, best.x)), E=C.err(best.x, "M2", "full", wide=True))
    print(f"q {q:.1f} chi1 {chi1:+.1f}  installed {out['E_installed']:.3e}  FLUX floor {out['FLUX']['E']:.3e}  "
          f"M2 floor {out['M2']['E']:.3e}  M2 s1 {best.x[4]:.1f} s2 {best.x[5]:.1f} rho {best.x[6]:+.3f}", flush=True)
    return out


def main():
    T96 = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
    with mp.get_context("fork").Pool(96) as pool:
        rows = pool.map(fit_case, T96)
    (OUT / "p5b_floor.json").write_text(json.dumps(rows, indent=1))
    for key in ("E_installed",):
        v = [r[key] for r in rows]; print(key, f"mean {np.mean(v):.3e} median {np.median(v):.3e}")
    for f in ("FLUX", "M2"):
        v = [r[f]["E"] for r in rows]; print(f, f"floor mean {np.mean(v):.3e} median {np.median(v):.3e} max {np.max(v):.3e}")


if __name__ == "__main__":
    main()
