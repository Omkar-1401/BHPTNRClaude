"""
Hybrid-alpha wf-nu q-dependent calibration (wf_nu_hybrid_q_dep).

Extends wf_nu_q_dep (10 params) by adding a normalized inspiral drift to alpha:

    dp_hat = (p_loss - p0) / (p_loss[0] - p0)   # ∈ [0,1] during inspiral
    alpha  = alpha_i + (1-S)*alpha_L*dp_hat + S*(alpha_E*dE + alpha_J*dJ)

dp_hat is normalized by the total inspiral dp range, so alpha_L represents a
bounded drift amplitude independent of where p0 sits — prevents compounding
at q=2 extrapolation (old unnormalized form had ~25× larger effective drift at
q=2 vs q=3 due to p0 extrapolating close to merger).
wf_nu_q_dep is the special case alpha_L=0.  One additional parameter.

Parameters (11):
    [p0, w, alpha_i, alpha_L, alpha_E, alpha_J, beta_i, beta_r, t0_nr, phi0, beta_L]

Indices:
    0:p0  1:w  2:alpha_i  3:alpha_L  4:alpha_E  5:alpha_J
    6:beta_i  7:beta_r  8:t0_nr  9:phi0  10:beta_L

Same waveform-flux loss coordinates, same PP-anchored nu regression as wf_nu_q_dep.
PP anchor: alpha_L → 0 at nu→0 (no drift correction at test-particle limit).
Per-q optimizer warm-started from wf_nu_q_dep per-q cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTNRSurrogate" / "surrogates"))

warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_wf_nu_q_dep as wfnu

# Reuse waveform loading / loss coordinates / regression helpers from wf_nu
load_case          = wfnu.load_case
get_nu             = wfnu.get_nu
fit_poly_anchored  = wfnu.fit_poly_anchored
eval_poly          = wfnu.eval_poly

PARAM_NAMES = [
    "p0", "w", "alpha_i", "alpha_L", "alpha_E", "alpha_J",
    "beta_i", "beta_r", "t0_nr", "phi0", "beta_L",
]

RESULTS_DIR = ROOT / "wf_nu_hybrid_q_dep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH      = RESULTS_DIR / "per_q_cache.json"
WFNU_CACHE_PATH = ROOT / "wf_nu_q_dep_results" / "per_q_cache.json"
MD_PATH         = ROOT / "scaling_wf_nu_hybrid_q_dep.md"

MODE           = creative.MODE
NR_T_START     = creative.NR_T_START
NR_T_END       = creative.NR_T_END
T_ANCHOR       = creative.T_ANCHOR
T_REF          = creative.T_REF
MIN_COVERAGE   = qdep.MIN_COVERAGE
MASTER_OUTLIER_CUT = qdep.MASTER_OUTLIER_CUT
WAVEFORM_CACHE_DIR = qdep.WAVEFORM_CACHE_DIR

PP_ANCHORS: dict[str, float | None] = {
    "p0":      None,
    "w":       None,
    "alpha_i": 1.0,
    "alpha_L": 0.0,   # no drift at test-particle limit
    "alpha_E": 0.0,
    "alpha_J": 0.0,
    "beta_i":  1.0,
    "beta_r":  1.0,
    "t0_nr":   None,
    "phi0":    None,
    "beta_L":  0.0,
}


# ---------------------------------------------------------------------------
# Hybrid model evaluation
# ---------------------------------------------------------------------------

def evaluate_model_hybrid(
    params, t_bhpt, h_bhpt, t_nr, h_nr, losses, min_coverage=MIN_COVERAGE
):
    p0, w, alpha_i, alpha_L, alpha_E, alpha_J, beta_i, beta_r, t0_nr, phi0, beta_L = \
        [float(v) for v in params]

    p_loss = losses["p_loss"]
    e_hat  = losses["e_hat"]
    j_hat  = losses["j_hat"]

    S  = creative.sigmoid((p_loss - p0) / w)
    dp = p_loss - p0
    e0 = float(np.interp(p0, p_loss, e_hat))
    j0 = float(np.interp(p0, p_loss, j_hat))
    dE = e_hat - e0
    dJ = j_hat - j0

    # Normalize dp so alpha_L represents bounded drift amplitude ∈ [0,1]
    # regardless of where p0 sits — prevents compounding at q=2 extrapolation.
    denom  = p_loss[0] - p0          # < 0 during inspiral (start < switch)
    if abs(denom) < 1e-8:
        denom = -1.0
    dp_hat = dp / denom              # ∈ [0, 1] during inspiral
    alpha = alpha_i + (1.0 - S) * alpha_L * dp_hat + S * (alpha_E * dE + alpha_J * dJ)
    beta  = (1.0 - S) * (beta_i + beta_L * dp) + S * beta_r

    beta_cum   = creative.cumulative_trapezoid(beta, t_bhpt)
    anchor_val = float(np.interp(T_ANCHOR, t_bhpt, beta_cum))
    tau = t0_nr + beta_cum - anchor_val

    if not np.all(np.diff(tau) > 0):
        return {"error": 50.0}

    t_min = max(t_nr[0], tau[0])
    t_max = min(t_nr[-1], tau[-1])
    if t_max - t_min < 0:
        return {"error": 50.0}

    nr_mask  = (t_nr >= t_min) & (t_nr <= t_max)
    coverage = nr_mask.sum() / len(t_nr)
    if coverage < min_coverage:
        return {"error": 50.0}

    t_common = t_nr[nr_mask]
    h_ref    = h_nr[nr_mask]

    h_bhpt_r = np.interp(t_common, tau, h_bhpt.real)
    h_bhpt_i = np.interp(t_common, tau, h_bhpt.imag)
    alpha_c  = np.interp(t_common, tau, alpha)

    h_model = alpha_c * np.exp(1j * phi0) * (h_bhpt_r + 1j * h_bhpt_i)

    n1 = np.sum(np.abs(h_ref) ** 2)
    n2 = np.sum(np.abs(h_model) ** 2)
    sd = np.real(np.sum(h_ref * h_model.conjugate()))

    return {
        "error":    float(((n1 + n2) - 2.0 * sd) / (2.0 * n1)),
        "coverage": float(coverage),
        "tau":      tau,
        "alpha":    alpha,
        "beta":     beta,
        "e_hat":    np.interp(t_common, tau, e_hat),
        "j_hat":    np.interp(t_common, tau, j_hat),
        "h_ref":    h_ref,
        "h_model":  h_model,
        "common":   nr_mask,
    }


# ---------------------------------------------------------------------------
# Per-q optimisation
# ---------------------------------------------------------------------------

def _wfnu_to_hybrid(p10, alpha_L=0.0):
    """Map 10-param wf_nu vector to 11-param hybrid by inserting alpha_L."""
    # wf_nu:  [p0, w, alpha_i, alpha_E, alpha_J, beta_i, beta_r, t0_nr, phi0, beta_L]
    # hybrid: [p0, w, alpha_i, alpha_L, alpha_E, alpha_J, beta_i, beta_r, t0_nr, phi0, beta_L]
    p = list(p10)
    return np.array([p[0], p[1], p[2], alpha_L, p[3], p[4], p[5], p[6], p[7], p[8], p[9]])


def _load_wfnu_cache():
    if not WFNU_CACHE_PATH.exists():
        return {}
    return json.loads(WFNU_CACHE_PATH.read_text())


def optimize_case_hybrid(q, source_stride=3, nr_stride=5, top_n=4, maxiter=6000):
    case = load_case(q, source_stride, nr_stride)
    meta = case["meta"]

    def fit_err(params):
        return evaluate_model_hybrid(
            params,
            case["fit_t_bhpt"], case["fit_h_bhpt"],
            case["fit_t_nr"],   case["fit_h_nr"],
            case["fit_losses"],
        )["error"]

    seeds = []

    # Warm starts from wf_nu per-q cache
    wfnu_cache = _load_wfnu_cache()
    key = f"{q:.10f}"
    if key in wfnu_cache:
        wfnu_params = np.array(wfnu_cache[key]["params"])
        seeds.append(_wfnu_to_hybrid(wfnu_params, alpha_L=0.0))
        for al in [-0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05]:
            seeds.append(_wfnu_to_hybrid(wfnu_params, alpha_L=al))

    # Random seeds
    rng   = np.random.default_rng(int(q * 1000) % (2 ** 31))
    scale = (q / (1.0 + q)) / (5.0 / 6.0)
    for _ in range(60):
        p0   = float(rng.uniform(-1.5, -0.1))
        w    = float(np.clip(rng.uniform(0.05, 0.4), creative.W_MIN, 0.4))
        ai   = float(scale * rng.uniform(0.6, 1.1))
        aL   = float(rng.uniform(-0.05, 0.05))
        aE   = float(rng.uniform(-0.5,  0.5))
        aJ   = float(rng.uniform(-0.5,  0.5))
        bi   = float(scale * rng.uniform(0.6, 1.1))
        br   = float(scale * rng.uniform(0.7, 1.2))
        t0   = float(meta["t_nr_merger"] - bi * (meta["t_bhpt_merger"] - T_ANCHOR))
        phi0 = float(rng.uniform(-np.pi, np.pi))
        bL   = float(rng.uniform(-0.01, 0.01))
        seeds.append(np.array([p0, w, ai, aL, aE, aJ, bi, br, t0, phi0, bL]))

    ranked = []
    for s in seeds:
        e = fit_err(s)
        if e < 10.0:
            ranked.append((e, s.copy()))
    ranked.sort(key=lambda x: x[0])

    if not ranked:
        print(f"  q={q:.4g}  hybrid: NO valid seeds", flush=True)
        return {"q": q, "error": 50.0, "params": [0.0] * 11, "coverage": float("nan")}

    best_err    = ranked[0][0]
    best_params = ranked[0][1].copy()

    for _, start in ranked[:top_n]:
        res = minimize(
            fit_err, start, method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-13},
        )
        if res.fun < best_err:
            best_err    = float(res.fun)
            best_params = res.x.copy()

    full = evaluate_model_hybrid(
        best_params,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"],   case["h_nr"],
        case["losses"],
    )

    print(
        f"  q={q:.4g}  hybrid: fit={best_err:.4g}  full={full.get('error', 50.):.4g}"
        f"  alpha_L={best_params[3]:.5g}",
        flush=True,
    )
    return {
        "q":        q,
        "error":    float(full.get("error", 50.0)),
        "params":   best_params.tolist(),
        "coverage": float(full.get("coverage", float("nan"))),
    }


def optimize_case_worker(args):
    q, source_stride, nr_stride, top_n, maxiter = args
    return optimize_case_hybrid(q, source_stride, nr_stride, top_n, maxiter)


# ---------------------------------------------------------------------------
# Polynomial fitting (nu basis, PP-anchored)
# ---------------------------------------------------------------------------

def fit_master(rows, degree):
    q_arr  = np.array([r["q"] for r in rows], dtype=float)
    nu_arr = np.array([get_nu(q) for q in q_arr])
    params_arr = np.array([r["params"] for r in rows], dtype=float)

    coeffs: dict[str, np.ndarray] = {}
    for i, name in enumerate(PARAM_NAMES):
        vals = params_arr[:, i].copy()
        if name == "phi0":
            vals = np.unwrap(vals)
        coeffs[name] = fit_poly_anchored(nu_arr, vals, degree, PP_ANCHORS[name])
    return coeffs


def constrained_master_params(q, coeffs):
    params = np.array([eval_poly(q, coeffs[name]) for name in PARAM_NAMES], dtype=float)
    params[0]  = float(np.clip(params[0],  -3.0, 0.05))           # p0
    params[1]  = float(np.clip(params[1],  creative.W_MIN, 0.45)) # w
    params[2]  = float(np.clip(params[2],  0.05, 2.5))            # alpha_i
    params[6]  = float(np.clip(params[6],  0.2,  1.6))            # beta_i
    params[7]  = float(np.clip(params[7],  0.2,  1.6))            # beta_r
    params[10] = float(np.clip(params[10], -0.05, 0.05))          # beta_L
    return params


# ---------------------------------------------------------------------------
# Nuisance polish and master evaluation
# ---------------------------------------------------------------------------

def polish_nuisance(params, case):
    def obj(x):
        trial = params.copy()
        trial[8] = x[0]   # t0_nr
        trial[9] = x[1]   # phi0
        return evaluate_model_hybrid(
            trial,
            case["fit_t_bhpt"], case["fit_h_bhpt"],
            case["fit_t_nr"],   case["fit_h_nr"],
            case["fit_losses"],
        )["error"]

    result = minimize(
        obj, [params[8], params[9]], method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-7, "fatol": 1e-9},
    )
    out = params.copy()
    out[8] = float(result.x[0])
    out[9] = float(result.x[1])
    return out, float(result.fun)


def evaluate_master(rows, coeffs, source_stride, nr_stride, do_polish):
    out = []
    for row in rows:
        q      = row["q"]
        params = constrained_master_params(q, coeffs)
        case   = load_case(q, source_stride, nr_stride)
        fit_err = float("nan")
        if do_polish:
            params, fit_err = polish_nuisance(params, case)
        ev = evaluate_model_hybrid(
            params,
            case["t_bhpt"], case["h_bhpt"],
            case["t_nr"],   case["h_nr"],
            case["losses"],
        )
        out.append({
            "q":        q,
            "params":   [float(x) for x in params],
            "fit_error": fit_err,
            "error":    float(ev.get("error", 50.0)),
            "coverage": float(ev.get("coverage", float("nan"))),
        })
    return out


def _eval_master_worker(args):
    q, source_stride, nr_stride, do_polish, coeffs2, coeffs3, degree = args
    coeffs = coeffs3 if degree == 3 else coeffs2
    return evaluate_master([{"q": q}], coeffs, source_stride, nr_stride, do_polish)[0]


# ---------------------------------------------------------------------------
# Statistics and degree selection
# ---------------------------------------------------------------------------

def summarize(values):
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    return {
        "min": float(np.min(a)), "median": float(np.median(a)),
        "mean": float(np.mean(a)), "max": float(np.max(a)),
    }


def should_use_cubic(s2, s3):
    med_gain = (s2["median"] - s3["median"]) / max(s2["median"], 1e-15)
    max_gain = (s2["max"]    - s3["max"])    / max(s2["max"],    1e-15)
    return bool(med_gain > 0.03 or max_gain > 0.03)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def q_key(q):
    return f"{q:.10f}"


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def coeff_table_lines(coeffs, degree):
    header = "| parameter | anchor | " + " | ".join(f"c{i}" for i in range(degree + 1)) + " |"
    sep    = "|:---|:---|" + "|".join("---:" for _ in range(degree + 1)) + "|"
    lines  = [header, sep]
    for name in PARAM_NAMES:
        anc     = PP_ANCHORS[name]
        anc_str = f"{anc:.4g}" if anc is not None else "free"
        row     = " | ".join(f"{v:.12g}" for v in coeffs[name])
        lines.append(f"| {name} | {anc_str} | {row} |")
    return lines


def write_markdown(rows, fit_rows, master2, master3, coeffs2, coeffs3, selected_degree, args_ns):
    selected        = master3 if selected_degree == 3 else master2
    selected_coeffs = coeffs3 if selected_degree == 3 else coeffs2

    valid_rows = [r for r in rows if np.isfinite(r.get("coverage", float("nan"))) and r["error"] < 10.0]
    s_perq = summarize([r["error"] for r in valid_rows])
    s_fit  = summarize([r["error"] for r in fit_rows])
    s2     = summarize([r["error"] for r in master2])
    s3     = summarize([r["error"] for r in master3])
    s_sel  = summarize([r["error"] for r in selected])
    q_at_max = max(selected, key=lambda r: r["error"])["q"]

    nu_q3 = get_nu(3.0)
    nu_q8 = get_nu(8.0)
    nu_q2 = get_nu(2.0)

    anchored_params = [n for n, a in PP_ANCHORS.items() if a is not None]
    free_params     = [n for n, a in PP_ANCHORS.items() if a is None]

    lines = [
        "# Hybrid-alpha wf-nu q-dependent scaling fit (wf_nu_hybrid_q_dep)",
        "",
        "Extends wf_nu_q_dep by adding a linear inspiral drift to alpha (alpha_L).",
        "11 parameters; wf_nu_q_dep is the special case alpha_L=0.",
        "",
        "## Model equations",
        "",
        "```python",
        "S  = 1 / (1 + exp(-(p_loss - p0) / w))",
        "dE = Ehat - Ehat(p0);  dJ = Jhat - Jhat(p0);  dp = p_loss - p0",
        "dp_hat = (p_loss - p0) / (p_loss[0] - p0)   # in [0,1] during inspiral",
        "alpha = alpha_i + (1-S)*alpha_L*dp_hat + S*(alpha_E*dE + alpha_J*dJ)",
        "beta  = (1-S)*(beta_i + beta_L*dp) + S*beta_r",
        "tau(t) = t0_nr + integral_{t_anchor}^{t} beta(t') dt'",
        "h_model(tau(t)) = alpha(t) * exp(i*phi0) * h_BHPT(t)",
        "```",
        "",
        "## Regression",
        "",
        "```python",
        "nu(q) = q / (1 + q)**2",
        "",
        f"# Anchored (PP limit nu→0): {', '.join(anchored_params)}",
        f"# Unanchored:               {', '.join(free_params)}",
        "```",
        "",
        f"Training grid: {len(rows)} q values in "
        f"[{min(r['q'] for r in rows):.4g}, {max(r['q'] for r in rows):.4g}].",
        f"nu training range: [{nu_q8:.4f}, {nu_q3:.4f}].  "
        f"Extrapolation to q=2: nu={nu_q2:.4f} (+{(nu_q2 - nu_q3)/nu_q3 * 100:.1f}% above boundary).",
        f"Rows used in regression: {len(fit_rows)}.",
        f"Selected polynomial degree: **{selected_degree}**",
        "",
        "## Selected Master Coefficients",
        "",
        *coeff_table_lines(selected_coeffs, selected_degree),
        "",
        "## Error Summary",
        "",
        "| model | min | median | mean | max | q at max |",
        "|:---|---:|---:|---:|---:|---:|",
        f"| per-q independent | {s_perq['min']:.6g} | {s_perq['median']:.6g}"
        f" | {s_perq['mean']:.6g} | {s_perq['max']:.6g} | |",
        f"| per-q used in regression | {s_fit['min']:.6g} | {s_fit['median']:.6g}"
        f" | {s_fit['mean']:.6g} | {s_fit['max']:.6g} | |",
        f"| quadratic master | {s2['min']:.6g} | {s2['median']:.6g}"
        f" | {s2['mean']:.6g} | {s2['max']:.6g} | |",
        f"| cubic master | {s3['min']:.6g} | {s3['median']:.6g}"
        f" | {s3['mean']:.6g} | {s3['max']:.6g} | |",
        f"| selected master | {s_sel['min']:.6g} | {s_sel['median']:.6g}"
        f" | {s_sel['mean']:.6g} | {s_sel['max']:.6g} | {q_at_max:.6g} |",
        "",
        "## Per-q Results",
        "",
        "| q | nu | per-q E | master E | coverage | p0 | w | alpha_L | alpha_i | beta_i | beta_L |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    sel_by_q = {f"{r['q']:.10f}": r for r in selected}
    fit_keys = {f"{r['q']:.10f}" for r in fit_rows}
    for row in rows:
        key = f"{row['q']:.10f}"
        p   = row["params"]
        mr  = sel_by_q.get(key, {"error": float("nan"), "coverage": float("nan")})
        nu_q = get_nu(float(row["q"]))
        reg  = "✓" if key in fit_keys else ""
        # p indices: 0=p0,1=w,2=alpha_i,3=alpha_L,4=alpha_E,5=alpha_J,6=beta_i,7=beta_r,8=t0_nr,9=phi0,10=beta_L
        lines.append(
            f"| {row['q']:.6g} | {nu_q:.4f} | {row['error']:.4g}"
            f" | {mr['error']:.4g} | {mr['coverage']:.4f} | {reg}"
            f" | {p[0]:.4g} | {p[1]:.4g} | {p[3]:.6g} | {p[2]:.4g} | {p[6]:.4g} | {p[10]:.6g} |"
        )
    lines.append("")
    MD_PATH.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Cached q discovery
# ---------------------------------------------------------------------------

def get_cached_q_values():
    import re
    pat = re.compile(r"waveforms_q(.+)\.npz")
    qs  = []
    for p in WAVEFORM_CACHE_DIR.glob("waveforms_q*.npz"):
        m = pat.match(p.name)
        if m:
            qs.append(float(m.group(1)))
    return sorted(qs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-min",              type=float, default=3.0)
    parser.add_argument("--q-max",              type=float, default=8.0)
    parser.add_argument("--source-stride",      type=int,   default=3)
    parser.add_argument("--nr-stride",          type=int,   default=5)
    parser.add_argument("--top-n",              type=int,   default=4)
    parser.add_argument("--maxiter",            type=int,   default=6000)
    parser.add_argument("--master-outlier-cut", type=float, default=MASTER_OUTLIER_CUT)
    parser.add_argument("--workers",            type=int,   default=4)
    parser.add_argument("--force",              action="store_true")
    args = parser.parse_args()

    all_q = [q for q in get_cached_q_values()
             if args.q_min - 1e-9 <= q <= args.q_max + 1e-9]
    if not all_q:
        raise RuntimeError(
            "No waveforms found in .cache/q_dep/.  "
            "Run fit_scaling_PN_opt_creative_q_dep.py first."
        )
    print(f"Training grid: {len(all_q)} q values in [{all_q[0]:.4g}, {all_q[-1]:.4g}]",
          flush=True)

    # Phase 1: per-q optimisation (warm-started from wf_nu cache)
    cache = {} if args.force else load_cache()
    todo  = [q for q in all_q if q_key(q) not in cache]
    if todo:
        print(f"Optimising {len(todo)} q values ({args.workers} workers) ...", flush=True)
        opt_args = [
            (q, args.source_stride, args.nr_stride, args.top_n, args.maxiter)
            for q in todo
        ]
        with Pool(args.workers) as pool:
            results = pool.map(optimize_case_worker, opt_args)
        for row in results:
            cache[q_key(row["q"])] = row
        save_cache(cache)
    else:
        print("All per-q results already cached.", flush=True)

    rows = sorted(
        [v for v in cache.values() if all_q[0] - 1e-9 <= v["q"] <= all_q[-1] + 1e-9],
        key=lambda r: r["q"],
    )

    # Phase 2: polynomial fitting (nu basis, PP-anchored)
    fit_rows = [
        r for r in rows
        if np.isfinite(r.get("coverage", float("nan"))) and r["error"] < args.master_outlier_cut
    ]
    if len(fit_rows) < 8:
        raise RuntimeError(f"Too few valid rows ({len(fit_rows)}) for polynomial regression.")
    print(f"Fitting polynomials on {len(fit_rows)} rows (nu basis, PP-anchored) ...", flush=True)

    coeffs2 = fit_master(fit_rows, degree=2)
    coeffs3 = fit_master(fit_rows, degree=3)

    # Phase 3: parallel master evaluation
    print("Evaluating master model ...", flush=True)
    eval_args = [
        (r["q"], args.source_stride, args.nr_stride, True, coeffs2, coeffs3, deg)
        for r in rows
        for deg in (2, 3)
    ]
    with Pool(args.workers) as pool:
        all_results = pool.map(_eval_master_worker, eval_args)

    master2 = all_results[0::2]
    master3 = all_results[1::2]

    s2 = summarize([r["error"] for r in master2])
    s3 = summarize([r["error"] for r in master3])
    selected_degree = 3 if should_use_cubic(s2, s3) else 2

    print(f"quadratic: median={s2['median']:.6g}  max={s2['max']:.6g}")
    print(f"cubic:     median={s3['median']:.6g}  max={s3['max']:.6g}")
    print(f"selected_degree={selected_degree}")

    write_markdown(rows, fit_rows, master2, master3, coeffs2, coeffs3, selected_degree, args)
    print(f"Wrote {MD_PATH}", flush=True)


if __name__ == "__main__":
    main()
