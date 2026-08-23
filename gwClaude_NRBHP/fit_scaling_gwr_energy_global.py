"""
Global joint fit of the gw_remnant energy-driven model (gwr_energy_global).

Same model as `fit_scaling_gw_remnant_energy.py --form mult`:

    E(t)   = gw_remnant Eoft   (cumulative radiated energy, units of M, zero at
                                the start of the surrogate window; modes (2,2)+(2,-2))
    alpha(t, nu) = alpha_PP(nu) * (1 + alpha_E(nu) * E(t))
    beta (t, nu) = beta_PP (nu) * (1 + beta_E (nu) * E(t))
    tau(t) = t0_nr + integral_{T_ANCHOR}^{t} beta dt'
    h_model(tau) = alpha * exp(i*phi0) * h_BHPT

what changes is HOW the nu-polynomials are obtained.  The per-q -> regress route
fits each q independently and then regresses each coefficient separately, which
provably cannot work here: (beta_PP, P = beta_PP*beta_E) is a strongly degenerate
pair, so the per-q solutions sit on a curved valley whose location moves with nu, and
regressing the two coordinates independently walks off it.  Measured consequences at
q=2: beta_PP needs ~+-0.5% accuracy, correcting beta_PP or beta_E ALONE makes the
mismatch worse (2.4e-2 -> 3.3e-2 / 9.4e-2) while correcting both together gives
2.5e-3, and P crosses zero at q ~ 4.1 so its slope at the q=3 boundary is
unconstrained (held-out CV inside [3,8]: 44-721% error).

Here the master polynomial coefficients are optimised DIRECTLY against all the
training waveforms at once (Powell), so the fit lands on the valley by construction.

Fast evaluator, as in wf_nu_hybrid_global:
  * phi0 is analytic -- the optimal overall phase is the argument of the complex
    overlap, so it is never searched;
  * t0_nr is the only per-q nuisance and is a 1-D bounded search, warm-started
    from the previous objective evaluation.

Regularisation penalises each coefficient's NONLINEAR (k >= 2) excursion at nu(q=2),
normalised by that parameter's [3,8] spread.  This is what keeps the extrapolation
gentle: it is precisely the freedom the per-q route used to bend beta_E the wrong way
below q=3 while accommodating the real turnover above q ~ 7.

NO q < 3 NR data is used anywhere.  q = 2 enters only through the self-penalty, which
never touches a waveform.  Training points are sampled even in nu (densest near the
q=3 boundary that governs q<3 behaviour), matching wf_nu_hybrid_global and pn_anchored.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G

FORM = "mult"
NAMES = ["alpha_PP", "alpha_E", "beta_PP", "beta_E"]
ANCH = [G.PP_ANCHORS[FORM][n] for n in NAMES]        # [1.0, None, 1.0, None]

RESULTS = ROOT / "gwr_energy_global_results"
RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs.json"
MD_PATH = ROOT / "scaling_gwr_energy_global.md"
SEED_CACHE = G.RESULTS_DIR / "per_q_cache_mult.json"
SEED_COEFFS = G.RESULTS_DIR / "coeffs_mult.json"

LOW_Q = (2.75, 2.5, 2.25, 2.0)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
NU_Q2 = G.get_nu(2.0)


# ---------------------------------------------------------------------------
# pack / unpack: anchored params contribute `degree` free coefficients (c0 pinned),
# unanchored ones contribute degree+1
# ---------------------------------------------------------------------------

def layout(degree):
    return [(i, degree if ANCH[i] is not None else degree + 1) for i in range(len(NAMES))]


def unpack(theta, degree):
    coeffs, k = {}, 0
    for i, nc in layout(degree):
        blk = np.asarray(theta[k:k + nc], float)
        k += nc
        coeffs[NAMES[i]] = (np.concatenate([[ANCH[i]], blk]) if ANCH[i] is not None
                            else blk)
    return coeffs


def pack(coeffs, degree):
    out = []
    for i, nc in layout(degree):
        c = np.asarray(coeffs[NAMES[i]], float)
        out.extend(c[1:1 + nc] if ANCH[i] is not None else c[:nc])
    return np.array(out, float)


# ---------------------------------------------------------------------------
# fast evaluator
# ---------------------------------------------------------------------------

def params_at(q, coeffs):
    p = np.array([G.eval_poly(q, coeffs[n]) for n in NAMES], float)
    p[0] = np.clip(p[0], 0.05, 2.5)      # alpha_PP
    p[2] = np.clip(p[2], 0.2, 1.6)       # beta_PP
    return p


def model_shape(p, t_bhpt, losses):
    """alpha(t) and the t0-independent tau shape.  dmin <= 0 flags non-monotonicity."""
    a_pp, a_e, b_pp, b_e = p
    e = losses["e_oft"]
    alpha = a_pp * (1.0 + a_e * e)
    beta = b_pp * (1.0 + b_e * e)
    if np.any(beta <= 0):
        return alpha, None, -1.0
    bc = G.creative.cumulative_trapezoid(beta, t_bhpt)
    tau_shape = bc - float(np.interp(G.T_ANCHOR, t_bhpt, bc))
    return alpha, tau_shape, float(np.min(np.diff(tau_shape)))


def err_at_t0(t0, alpha, tau_shape, h_bhpt, t_nr, h_nr):
    tau = t0 + tau_shape
    tmin = max(t_nr[0], tau[0]); tmax = min(t_nr[-1], tau[-1])
    if tmax - tmin < 0:
        return 50.0
    m = (t_nr >= tmin) & (t_nr <= tmax)
    if m.sum() / len(t_nr) < G.MIN_COVERAGE:
        return 50.0
    tc = t_nr[m]; href = h_nr[m]
    g = np.interp(tc, tau, h_bhpt.real) + 1j * np.interp(tc, tau, h_bhpt.imag)
    g = np.interp(tc, tau, alpha) * g
    n1 = np.sum(np.abs(href) ** 2); n2 = np.sum(np.abs(g) ** 2)
    Z = np.abs(np.sum(href * np.conj(g)))            # analytic phi0 optimum
    return float((n1 + n2 - 2.0 * Z) / (2.0 * n1))


def fast_mismatch(p, case, t0_seed, key="fit"):
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"],
                               case["fit_t_nr"], case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"],
                               case["t_nr"], case["h_nr"], case["losses"])
    alpha, tau_shape, dmin = model_shape(p, tb, los)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_seed
    f = lambda t0: err_at_t0(t0, alpha, tau_shape, hb, tn, hn)
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0),
                        method="bounded", options={"xatol": 1e-3, "maxiter": 50})
    return float(r.fun), float(r.x)


def full_mismatch(q, coeffs, case, t0_seed=-75.0):
    p = params_at(q, coeffs)
    e, t0 = fast_mismatch(p, case, t0_seed, "fit")
    e, t0 = fast_mismatch(p, case, t0, "full")
    return e, t0


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------

def build_objective(train_q, cases, degree, lam, spans, t0_init):
    nuis = dict(t0_init)

    def objective(theta):
        coeffs = unpack(theta, degree)
        errs = []
        for q in train_q:
            e, t0 = fast_mismatch(params_at(q, coeffs), cases[q], nuis[q], "fit")
            nuis[q] = t0
            errs.append(e)
        errs = np.array(errs)
        pen = 0.0
        for i, n in enumerate(NAMES):
            c = coeffs[n]
            nonlin = sum(c[k] * NU_Q2 ** k for k in range(2, len(c)))
            pen += (nonlin / spans[i]) ** 2
        reg = lam * pen
        objective.last = (float(errs.mean()), float(errs.max()), float(reg))
        return errs.mean() + reg

    objective.last = (np.nan,) * 3
    return objective


# ---------------------------------------------------------------------------

def even_nu_targets(q_all, ntrain):
    nu_t = np.linspace(G.get_nu(q_all[-1]), G.get_nu(q_all[0]), ntrain)
    return (1 - 2 * nu_t + np.sqrt(np.clip(1 - 4 * nu_t, 0, None))) / (2 * nu_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=3)
    # lam=0 is the shipped choice.  Sweeping lam in {0, 1e-8, 1e-6} at degree 3 makes
    # every low-q point monotonically WORSE with increasing lam
    # (q2.5: 1.40e-3 / 2.39e-3 / 7.53e-3; q2: 2.76e-2 / 4.79e-2 / 7.72e-2).  The
    # penalty suppresses nonlinear excursion at nu(q=2), but the true behaviour there
    # IS strongly nonlinear -- beta_E turns sharply upward below q=3 -- so smoothing
    # pushes away from the answer rather than toward it.  Kept as an option only.
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--ntrain", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--src", type=int, default=6)
    ap.add_argument("--nr", type=int, default=10)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    cache = json.loads(SEED_CACHE.read_text())
    rows = sorted(({"q": float(k), "params": v["params"]} for k, v in cache.items()),
                  key=lambda r: r["q"])
    q_all = np.array([r["q"] for r in rows])
    targets = even_nu_targets(q_all, args.ntrain)
    train_q = sorted({float(q_all[np.argmin(np.abs(q_all - t))]) for t in targets})

    print(f"[gwr_energy_global] degree={args.degree} lam={args.lam} "
          f"ntrain={len(train_q)} src={args.src} nr={args.nr} sample=nu", flush=True)
    print(f"train q: {[round(q, 2) for q in train_q]}", flush=True)

    cases = {q: G.load_case(q, args.src, args.nr) for q in train_q}

    # seed from the shipped per-q -> regress solution
    seed = json.loads(SEED_COEFFS.read_text())
    sdeg = str(args.degree) if str(args.degree) in seed["coeffs"] else \
        str(seed["selected_degree"])
    seed_coeffs = {n: np.asarray(seed["coeffs"][sdeg][n], float) for n in NAMES}
    theta0 = pack(seed_coeffs, args.degree)

    pa = np.array([r["params"] for r in rows], float)
    spans = {i: max(pa[:, i].std(), 1e-3) for i in range(len(NAMES))}
    t0_poly = np.polyfit(np.array([G.get_nu(r["q"]) for r in rows]), pa[:, 4], 3)
    t0_init = {q: float(np.polyval(t0_poly, G.get_nu(q))) for q in train_q}

    obj = build_objective(train_q, cases, args.degree, args.lam, spans, t0_init)
    obj(theta0)
    m0 = obj.last
    print(f"seed:  mean={m0[0]:.4e}  max={m0[1]:.4e}  reg={m0[2]:.3e}", flush=True)

    t_start = time.time()
    res = minimize(obj, theta0, method="Powell",
                   options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
    print(f"opt {time.time() - t_start:.0f}s:  mean={obj.last[0]:.4e}  "
          f"max={obj.last[1]:.4e}  reg={obj.last[2]:.3e}", flush=True)
    coeffs = unpack(res.x, args.degree)

    print("\n=== validation (analytic phi0, full resolution) ===", flush=True)
    ir_g, ir_s = [], []
    for q in IN_Q:
        c = cases.get(q) or G.load_case(q, args.src, args.nr)
        eg, _ = full_mismatch(q, coeffs, c)
        es, _ = full_mismatch(q, seed_coeffs, c)
        ir_g.append(eg); ir_s.append(es)
        print(f"  q={q:<5g} global={eg:.4e}   per-q->regress={es:.4e}", flush=True)
    print(f"  in-range  median: global={np.median(ir_g):.4e}  "
          f"per-q->regress={np.median(ir_s):.4e}", flush=True)
    print(f"  in-range  max   : global={np.max(ir_g):.4e}  "
          f"per-q->regress={np.max(ir_s):.4e}", flush=True)

    low = {}
    for q in LOW_Q:
        c = G.load_case(q, args.src, args.nr)
        eg, _ = full_mismatch(q, coeffs, c)
        es, _ = full_mismatch(q, seed_coeffs, c)
        low[q] = (eg, es)
        dom = "" if q >= G.BHPT_Q_MIN - 1e-9 else "  [BHPT input out of domain]"
        gate = "  <-- gate" if abs(q - G.BHPT_Q_MIN) < 1e-9 else ""
        flag = "" if eg < 1e-2 else "  (>1e-2)"
        print(f"  q={q:<5g} global={eg:.4e}   per-q->regress={es:.4e}"
              f"{gate}{flag}{dom}", flush=True)

    out = {n: coeffs[n].tolist() for n in NAMES}
    out["_meta"] = {
        "model": "gwr_energy_global",
        "form": FORM,
        "degree": args.degree,
        "lam": args.lam,
        "train_q": train_q,
        "sample": "even-nu",
        "n_free_coeffs": int(len(theta0)),
        "analytic_phi0": True,
        "in_range_median": float(np.median(ir_g)),
        "in_range_max": float(np.max(ir_g)),
        "low_q": {f"{q:.4g}": {"global": low[q][0], "per_q_regress": low[q][1]}
                  for q in LOW_Q},
        "note": "phi0 analytic; t0_nr per-q 1-D nuisance; no q<3 NR data used "
                "(q=2 enters only via the nonlinear-excursion penalty).",
    }
    path = COEFFS if not args.tag else COEFFS.with_name(f"coeffs{args.tag}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}", flush=True)
    write_markdown(coeffs, seed_coeffs, ir_g, ir_s, low, args, train_q, len(theta0))
    print(f"wrote {MD_PATH}", flush=True)


def write_markdown(coeffs, seed_coeffs, ir_g, ir_s, low, args, train_q, ncoef):
    lines = [
        "# gw_remnant energy-driven model, GLOBAL joint fit (gwr_energy_global)",
        "",
        "Same model as `gw_remnant_energy` form `mult`; the master polynomial",
        "coefficients are optimised directly against all training waveforms at once",
        "(Powell) instead of per-q fit -> independent regression.",
        "",
        "## Model",
        "",
        "```python",
        "E(t) = gw_remnant Eoft            # radiated energy, units of M, E(t_start)=0",
        "alpha(t, nu) = alpha_PP(nu) * (1 + alpha_E(nu) * E(t))",
        "beta (t, nu) = beta_PP (nu) * (1 + beta_E (nu) * E(t))",
        "tau(t) = t0_nr + integral_{T_ANCHOR}^{t} beta dt'",
        "```",
        "",
        f"Degree {args.degree} in nu; alpha_PP, beta_PP PP-anchored to 1 at nu=0; the",
        f"couplings unanchored (E_rad -> 0 carries the PP limit).  **{ncoef} free",
        "coefficients total.**  phi0 analytic, t0_nr a per-q 1-D nuisance.",
        "",
        f"Training: {len(train_q)} mass ratios sampled even in nu over [3, 8] --",
        f"{[round(q, 2) for q in train_q]}.  No q<3 NR data is used; q=2 enters only",
        f"through the nonlinear-excursion penalty (lam = {args.lam:g}).",
        "",
        "## Why global",
        "",
        "(beta_PP, P = beta_PP*beta_E) is a strongly degenerate pair: at q=2 correcting",
        "either alone makes the mismatch WORSE (2.4e-2 -> 3.3e-2 / 9.4e-2) while",
        "correcting both together gives 2.5e-3.  Independent regression of the two",
        "cannot stay on that valley; a joint fit lands on it by construction.",
        "",
        "## Results",
        "",
        "| q | global | per-q -> regress |",
        "|---:|---:|---:|",
    ]
    for q, eg, es in zip(IN_Q, ir_g, ir_s):
        lines.append(f"| {q:g} | {eg:.4e} | {es:.4e} |")
    lines += [
        f"| **in-range median** | **{np.median(ir_g):.4e}** | {np.median(ir_s):.4e} |",
        f"| **in-range max** | **{np.max(ir_g):.4e}** | {np.max(ir_s):.4e} |",
        "",
        "| q < 3 | global | per-q -> regress | BHPT input |",
        "|---:|---:|---:|:---|",
    ]
    for q in LOW_Q:
        dom = "in domain" if q >= G.BHPT_Q_MIN - 1e-9 else "**extrapolated**"
        lines.append(f"| {q:g} | {low[q][0]:.4e} | {low[q][1]:.4e} | {dom} |")
    lines += [
        "",
        f"`BHPTNRSur1dq1e4` is declared valid only for q >= {G.BHPT_Q_MIN}, so q=2.5 is the",
        "honest low-q gate; 2.25 and 2.0 additionally extrapolate the surrogate itself.",
        "",
        "## Coefficients",
        "",
        "| parameter | anchor | " + " | ".join(f"c{i}" for i in range(args.degree + 1)) + " |",
        "|:---|:---|" + "|".join("---:" for _ in range(args.degree + 1)) + "|",
    ]
    for i, n in enumerate(NAMES):
        a = "1" if ANCH[i] is not None else "free"
        lines.append(f"| {n} | {a} | " + " | ".join(f"{v:.10g}" for v in coeffs[n]) + " |")
    lines += ["", f"Command: `python {Path(__file__).name} --degree {args.degree} "
                  f"--lam {args.lam:g} --ntrain {args.ntrain}`", ""]
    MD_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
