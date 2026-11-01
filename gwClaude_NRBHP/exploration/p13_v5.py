"""EXPLORATION pass 13 (V5 = V4 + the pass-12 principal shapes as fixed basis functions, 32 coefficients)
    ln beta  += (pa0 + pa1 chi1 + pa2 chi1^2 + pa3 nu) phi_beta(s)
    ln alpha += (pb0 + pb1 chi1 + pb2 chi1^2) phi_alpha(s)
phi_beta, phi_alpha: natural cubic splines through the first principal shapes of the pass-12 corrections
(exploration/p12/p12_shapes.json), on the ppBHPT clock s = t - t_peak,pp.

EXPLORATION pass 11 (V4 = V3 + chi1^2 in rho and in the alpha level, 25 coefficients; from p8_m2_insp.py)

EXPLORATION pass 8 (V3) -- M2 / V1 plus a richer INSPIRAL regression (from p7_m2_spinwin.py).

Pass 6 band analysis: the global M2 loses its gap to the per-case floor mostly in the inspiral (3.5e-4 vs 0.8e-4);
the floor's inspiral parameters want  ln(b_pp/X1^(6/5)) += b2 nu^2 + b3 nu chi1^2  and  P += s3 chi1 + s4 chi1^2.
FLUX branch here (13): a_pp = X1^(6/5)(1 + c0 nu + c1 nu^2), a_F = A0 nu + A1 nu^2,
    b_pp = X1^(6/5)(1 + b nu + b2 nu^2 + b3 nu chi1^2),  P = P0 + P1 nu + nu (s1 chi1 + s2 chi1^2) + s3 chi1 + s4 chi1^2,
    alpha = a_pp (1 + a_F F),  beta = (b_pp + P E)(1 + kappa_SO chi1 x^(3/2)).
Merger-ringdown branch: as V1 (t1, t2 linear in chi1; rho, alpha level linear in nu, chi1).  23 coefficients.

--- original pass-7 docstring follows ---
EXPLORATION pass 7 -- M2 with a spin-dependent blend window (and optional chi1^2 terms).

Same model as p6_m2_global.py, theta extended:
  V1 (19):  t1 = t1a + t1b chi1,  t2 = t2a + t2b chi1                  (blend window, ppBHPT clock from its peak)
  V2 (21):  V1 + rho += r3 chi1^2,  ln alpha level += l3 chi1^2
theta = FLUX 9 | t1a t1b t2a t2b | r0 r1 r2 [r3] | l0 l1 l2 [l3]
Both start from the pass-6 T96 solution and are refitted on T96 (Powell), then scored on T96 and H270 at full
resolution.  Writes exploration/p7/p7_<variant>.json (keys q_chi1 with 6 / 4 decimals).
"""
from __future__ import annotations
import json, sys, time, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO):
    sys.path.insert(0, str(p))
import p6_m2_global as P6                     # noqa: E402  (get, err machinery, anchors)
import fit_pn_anchored_spin as S              # noqa: E402
import fit_flux_anc_spin_with_orbit as FO     # noqa: E402
import p3_rd as P3                            # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p13"; OUT.mkdir(exist_ok=True)
VAR = "V5"
FLUXN = [f"flux_{n}" for n in FO.NAMES] + ["flux_b2", "flux_b3", "flux_s3", "flux_s4"]
NAMES = FLUXN + ["t1a", "t1b", "t2a", "t2b", "r0", "r1", "r2"] + (["r3"] if VAR in ("V2", "V4", "V5") else []) + \
        ["l0", "l1", "l2"] + (["l3"] if VAR in ("V2", "V4", "V5") else []) + (["pa0", "pa1", "pa2", "pa3", "pb0", "pb1", "pb2"] if VAR == "V5" else [])


from scipy.interpolate import CubicSpline as _CS
_SH = json.loads((HERE / "p12" / "p12_shapes.json").read_text())
_PB = _CS(_SH["knots_beta"], _SH["phi_beta"], bc_type="natural")
_PA = _CS(_SH["knots_alpha"], _SH["phi_alpha"], bc_type="natural")
_KB, _KA = (_SH["knots_beta"][0], _SH["knots_beta"][-1]), (_SH["knots_alpha"][0], _SH["knots_alpha"][-1])


def unpack(th):
    d = dict(zip(NAMES, th)); d.setdefault("r3", 0.0); d.setdefault("l3", 0.0); return d


def flux13(f, q, chi1, case, key):
    c0, c1, A0, A1, b, P0, P1, s1, s2, b2, b3, s3, s4 = f
    pre = "fit_" if key == "fit" else ""
    nu = q / (1 + q) ** 2; bs = (q / (1 + q)) ** 1.2
    a_pp = bs * (1 + c0 * nu + c1 * nu ** 2)
    b_pp = bs * (1 + b * nu + b2 * nu ** 2 + b3 * nu * chi1 ** 2)
    a_f = A0 * nu + A1 * nu ** 2
    P = P0 + P1 * nu + nu * (s1 * chi1 + s2 * chi1 ** 2) + s3 * chi1 + s4 * chi1 ** 2
    los = case[pre + "losses"]
    x = S.PC.x_scale(q) * np.minimum(case[pre + "x"], S.XCLIP)
    alpha = a_pp * (1.0 + a_f * los["flux_hat"])
    beta = (b_pp + P * los["e_oft"]) * (1.0 + FO.kappa_so(q) * chi1 * x ** 1.5)
    return alpha, beta


def ab(th, k, key):
    q, chi1 = k; c = P6.get(k); case = c["case"]; pre = "fit_" if key == "fit" else ""
    t = case[pre + "t_bhpt"]
    d = unpack(th)
    aF, bF = flux13(np.asarray(th[:13]), q, chi1, case, key)
    s1 = d["t1a"] + d["t1b"] * chi1; s2 = d["t2a"] + d["t2b"] * chi1
    if not (-150 <= s1 <= 30 and s1 + 3 <= s2 <= 150) or np.any(aF <= 0) or np.any(bF <= 0):
        return None, None
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    br, kap = c["anc"]
    s = t - c["tpk"]
    W = P3.smooth5((s - s1) / (s2 - s1))
    lbm = np.log(br) + d["r0"] + d["r1"] * chi1 + d["r2"] * nu + d["r3"] * chi1 ** 2
    lam = (1.2 * np.log(X1) + d["l0"] + d["l1"] * chi1 + d["l2"] * nu + d["l3"] * chi1 ** 2
           + kap * br * np.clip(s - P6.S_RING, 0, None))
    la = (1 - W) * np.log(aF) + W * lam
    lb = (1 - W) * np.log(bF) + W * lbm
    if VAR == "V5":
        A = d["pa0"] + d["pa1"] * chi1 + d["pa2"] * chi1 ** 2 + d["pa3"] * nu
        B = d["pb0"] + d["pb1"] * chi1 + d["pb2"] * chi1 ** 2
        lb = lb + A * _PB(np.clip(s, *_KB))
        la = la + B * _PA(np.clip(s, *_KA))
    return np.exp(la), np.exp(lb)


P6.ab = ab          # route P6.err through this model


def key(q, c):
    return f"{q:.6f}_{c:+.4f}"


def main():
    pool = mp.get_context("fork").Pool(96)
    v4 = json.loads((HERE / "p11" / "p11_V4.json").read_text())["theta"]
    x0 = v4 + [-0.044, 0.215, -0.179, -0.374, 0.066, -0.2, 0.228]      # amplitudes from the pass-12 projections (on V3)
    P6.OUT = OUT
    th, note = P6.run_fit(pool, x0, P6.T96, 75, VAR)
    e96 = np.array(pool.map(P6._full, [(list(th), k) for k in P6.T96]))
    import plot_spin_paper_figs as PF
    H = PF.heldout_points()
    eh = np.array(pool.map(P6._full, [(list(th), k) for k in H]))
    out = dict(variant=VAR, names=NAMES, theta=list(map(float, th)), note=note,
               T96={key(*k): float(e) for k, e in zip(P6.T96, e96)}, H270={key(*k): float(e) for k, e in zip(H, eh)})
    (OUT / f"p13_{VAR}.json").write_text(json.dumps(out, indent=1))
    print(f"{VAR}: T96 mean {e96.mean():.4e} median {np.median(e96):.4e} max {e96.max():.4e} | "
          f"H270 mean {eh.mean():.4e} median {np.median(eh):.4e} max {eh.max():.4e}", flush=True)
    for ch in S.CHIS:
        v = [e for (q, c), e in zip(P6.T96, e96) if c == ch]
        print(f"   chi1 {ch:+.1f}: median {np.median(v):.3e} max {np.max(v):.3e}", flush=True)
    print("   q medians: " + " ".join(f"{np.median([e for (qq, c), e in zip(P6.T96, e96) if qq == q]):.2e}" for q in S.TRAIN_Q), flush=True)
    print("   theta: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)), flush=True)


if __name__ == "__main__":
    main()
