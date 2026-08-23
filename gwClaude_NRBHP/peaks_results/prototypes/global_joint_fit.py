"""Global joint fit for wf_nu_hybrid (PURE [3,8]).

Optimize the master POLYNOMIAL COEFFICIENTS directly against all [3,8] waveforms at
once (vs per-q fit -> regress, which fits degeneracy-scattered per-q params). Fast
evaluator: phi0 is analytic (optimal overall phase = angle of the complex overlap),
so only t0_nr is a per-q 1D search. Regularization penalizes each param's NONLINEAR
excursion at q=2 relative to its [3,8] span -> gentle extrapolation -> q=2 near the
[3,8] trend and alpha/beta shapes continuous below q=3.

No q<3 NR data is used; q=2 appears only as a model self-evaluation in the penalty.
"""
import sys, json, warnings, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_wf_nu_hybrid_q_dep as H

N = H.PARAM_NAMES
IT0, IPH = N.index("t0_nr"), N.index("phi0")
GLOBAL = [i for i in range(11) if i not in (IT0, IPH)]        # 9 physical params
ANCH = [H.PP_ANCHORS[N[i]] for i in range(11)]
NU_Q2 = H.get_nu(2.0)

# ---------------- pack / unpack ----------------
def layout(degree):
    return [(i, degree if ANCH[i] is not None else degree + 1) for i in GLOBAL]

def unpack(theta, degree):
    coeffs, k = {}, 0
    for i, nc in layout(degree):
        blk = theta[k:k+nc]; k += nc
        coeffs[N[i]] = np.concatenate([[ANCH[i]], blk]) if ANCH[i] is not None else np.array(blk)
    return coeffs

def pack(coeffs, degree):
    out = []
    for i, nc in layout(degree):
        c = coeffs[N[i]]
        out.extend(c[1:1+nc] if ANCH[i] is not None else c[:nc])
    return np.array(out, float)

# ---------------- fast evaluator: analytic phi0, tau shape precomputed ----------------
def model_shape(params, t_bhpt, h_bhpt, losses):
    """alpha(t) and tau_shape(t)=beta_cum-anchor (t0-independent). Returns None if non-monotone."""
    p0, w, ai, aL, aE, aJ, bi, br, _t0, _ph, bL = params
    pl = losses["p_loss"]; eh = losses["e_hat"]; jh = losses["j_hat"]
    S = H.creative.sigmoid((pl - p0) / w)
    dp = pl - p0
    e0 = np.interp(p0, pl, eh); j0 = np.interp(p0, pl, jh)
    dE = eh - e0; dJ = jh - j0
    denom = pl[0] - p0
    if abs(denom) < 1e-8: denom = -1.0
    dp_hat = dp / denom
    alpha = ai + (1 - S) * aL * dp_hat + S * (aE * dE + aJ * dJ)
    beta = (1 - S) * (bi + bL * dp) + S * br
    bc = H.creative.cumulative_trapezoid(beta, t_bhpt)
    anchor = np.interp(H.T_ANCHOR, t_bhpt, bc)
    tau_shape = bc - anchor
    dmin = np.min(np.diff(tau_shape))
    return alpha, tau_shape, dmin

def err_at_t0(t0, alpha, tau_shape, t_bhpt, h_bhpt, t_nr, h_nr):
    tau = t0 + tau_shape
    tmin = max(t_nr[0], tau[0]); tmax = min(t_nr[-1], tau[-1])
    if tmax - tmin < 0: return 50.0
    m = (t_nr >= tmin) & (t_nr <= tmax)
    cov = m.sum() / len(t_nr)
    if cov < H.MIN_COVERAGE: return 50.0
    tc = t_nr[m]; href = h_nr[m]
    gr = np.interp(tc, tau, h_bhpt.real); gi = np.interp(tc, tau, h_bhpt.imag)
    ac = np.interp(tc, tau, alpha)
    g = ac * (gr + 1j * gi)
    n1 = np.sum(np.abs(href) ** 2); n2 = np.sum(np.abs(g) ** 2)
    Z = np.abs(np.sum(href * np.conj(g)))          # analytic phi0 optimum
    return float((n1 + n2 - 2 * Z) / (2 * n1))

def fast_mismatch(params, case, t0_seed, key):
    """min over t0 of analytic-phi0 mismatch. key selects 'fit' (coarse) or full arrays."""
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"],
                               case["fit_t_nr"], case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"],
                               case["t_nr"], case["h_nr"], case["losses"])
    alpha, tau_shape, dmin = model_shape(params, tb, hb, los)
    if dmin <= 0:                                   # non-monotone: smooth penalty
        return 50.0 + 1e3 * (-dmin), t0_seed
    f = lambda t0: err_at_t0(t0, alpha, tau_shape, tb, hb, tn, hn)
    r = minimize_scalar(f, bounds=(t0_seed - 50.0, t0_seed + 50.0),
                        method="bounded", options={"xatol": 1e-3, "maxiter": 50})
    return float(r.fun), float(r.x)

def clip9(p):
    p = p.copy()
    p[0] = np.clip(p[0], -3.0, 0.05); p[1] = np.clip(p[1], H.creative.W_MIN, 0.45)
    p[2] = np.clip(p[2], 0.05, 2.5); p[6] = np.clip(p[6], 0.2, 1.6)
    p[7] = np.clip(p[7], 0.2, 1.6); p[10] = np.clip(p[10], -0.05, 0.05)
    return p

def params_at(q, coeffs):
    return clip9(np.array([H.eval_poly(q, coeffs[N[i]]) if i in GLOBAL else 0.0
                           for i in range(11)]))

# ---------------- objective ----------------
def build_objective(train_q, cases, degree, lam, spans, t0_init):
    nuis = dict(t0_init)
    def objective(theta):
        coeffs = unpack(theta, degree)
        errs = []
        for q in train_q:
            p = params_at(q, coeffs)
            e, t0 = fast_mismatch(p, cases[q], nuis[q], "fit")
            nuis[q] = t0; errs.append(e)
        errs = np.array(errs)
        # penalize nonlinear excursion of each param at q=2 (relative to [3,8] span)
        pen = 0.0
        for i in GLOBAL:
            c = coeffs[N[i]]
            nonlin = sum(c[k] * NU_Q2 ** k for k in range(2, len(c)))
            pen += (nonlin / spans[i]) ** 2
        reg = lam * pen
        objective.last = (float(errs.mean()), float(errs.max()), float(reg))
        return errs.mean() + reg
    objective.last = (np.nan,) * 3
    return objective

def full_mismatch(q, coeffs, case, t0_seed=None):
    p = params_at(q, coeffs)
    if t0_seed is None:
        t0_seed = H.eval_poly(q, coeffs[N[IT0]]) if len(coeffs[N[IT0]]) else -50.0
    e, t0 = fast_mismatch(p, case, t0_seed, "fit")
    e, t0 = fast_mismatch(p, case, t0, "full")
    return e

# ---------------- driver ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1e-8)
    ap.add_argument("--ntrain", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--src", type=int, default=6)
    ap.add_argument("--nr", type=int, default=10)
    # even-nu sampling is the shipped choice: dense near the q=3 boundary that governs
    # q<3 extrapolation (replaced even-q, which failed the gate at q=2.25).
    ap.add_argument("--sample", type=str, default="nu", choices=["q", "logq", "nu"])
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    cache = json.loads((ROOT/"wf_nu_hybrid_q_dep_results"/"per_q_cache.json").read_text())
    q_all = np.array(sorted(float(k) for k in cache.keys()))
    if args.sample == "q":                 # even in q (dense in nu at high q)
        targets = np.linspace(q_all[0], q_all[-1], args.ntrain)
    elif args.sample == "logq":            # even in log q (weights low q)
        targets = np.exp(np.linspace(np.log(q_all[0]), np.log(q_all[-1]), args.ntrain))
    else:                                  # even in nu (densest near q=3 boundary)
        nu_t = np.linspace(H.get_nu(q_all[-1]), H.get_nu(q_all[0]), args.ntrain)
        # invert nu->q on the branch q>=1: q = (1-2nu+sqrt(1-4nu))/(2nu)
        targets = (1 - 2*nu_t + np.sqrt(np.clip(1 - 4*nu_t, 0, None))) / (2*nu_t)
    train_q = sorted({float(q_all[np.argmin(np.abs(q_all - t))]) for t in targets})
    print(f"degree={args.degree} lam={args.lam} ntrain={len(train_q)} src={args.src} nr={args.nr} sample={args.sample}", flush=True)
    print(f"train q: {[round(q,2) for q in train_q]}", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = {q: H.load_case(q, args.src, args.nr) for q in train_q}
        rows = [{"q": float(k), "params": v["params"]} for k, v in cache.items()]
        seed_coeffs = H.fit_master(rows, args.degree)
        # per-param [3,8] span for reg normalization
        qs = np.array([r["q"] for r in rows]); pa = np.array([r["params"] for r in rows])
        spans = {i: max(pa[:, i].std(), 1e-3) for i in range(11)}
        theta0 = pack(seed_coeffs, args.degree)
        t0_init = {q: H.eval_poly(q, seed_coeffs[N[IT0]]) for q in train_q}

        obj = build_objective(train_q, cases, args.degree, args.lam, spans, t0_init)
        obj(theta0); m0 = obj.last
        print(f"seed: mean={m0[0]:.3e} max={m0[1]:.3e} reg={m0[2]:.3e}", flush=True)

        t0 = time.time()
        res = minimize(obj, theta0, method="Powell",
                       options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        print(f"opt {time.time()-t0:.0f}s: mean={obj.last[0]:.3e} max={obj.last[1]:.3e} reg={obj.last[2]:.3e}", flush=True)
        coeffs = unpack(res.x, args.degree)
        coeffs[N[IT0]] = seed_coeffs[N[IT0]]; coeffs[N[IPH]] = seed_coeffs[N[IPH]]

        print("\n=== validation (analytic-phi0, full-res) ===", flush=True)
        ir = []
        for q in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
            c = cases.get(q) or H.load_case(q, args.src, args.nr)
            e = full_mismatch(q, coeffs, c); eb = full_mismatch(q, seed_coeffs, c); ir.append(e)
            print(f"  q={q}: global={e:.3e}  master={eb:.3e}", flush=True)
        print(f"  in-range median: global={np.median(ir):.3e}", flush=True)
        for q in [2.5, 2.25, 2.0, 1.75]:
            c = H.load_case(q, args.src, args.nr)
            e = full_mismatch(q, coeffs, c); eb = full_mismatch(q, seed_coeffs, c)
            g = "  <--q2 GATE" if q == 2.0 else ""
            fl = "" if e < 1e-2 else " (>1e-2)"
            print(f"  q={q}: global={e:.3e}  master={eb:.3e}{g}{fl}", flush=True)

        out = {N[i]: coeffs[N[i]].tolist() for i in range(11)}
        out["_meta"] = {"degree": args.degree, "lam": args.lam, "train_q": train_q,
                        "in_range_median": float(np.median(ir))}
        fn = ROOT/"wf_nu_hybrid_q_dep_results"/f"global_coeffs{args.tag}.json"
        fn.write_text(json.dumps(out, indent=2)); print(f"\nsaved {fn.name}", flush=True)

if __name__ == "__main__":
    main()
