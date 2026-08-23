"""
Instantaneous-flux wf-nu q-dependent calibration (wf_nu_fluxes).

Gated wf_nu, plus two INSTANTANEOUS-flux terms on alpha:

    F_e = flux_e / max(flux_e)      # peaks ~at merger
    F_j = flux_j / max(flux_j)      # peaks slightly earlier (flux_e ~ omega*flux_j)
    S   = 1 / (1 + exp(-(p_loss - p0)/w))
    dE  = e_hat - e_hat(p0);  dJ = j_hat - j_hat(p0);  dp = p_loss - p0

    alpha = alpha_i + S*(alpha_E*dE + alpha_J*dJ) + alpha_F*F_e + alpha_G*F_j
    beta  = (1-S)*(beta_i + beta_L*dp) + S*beta_r
    tau(t) = t0_nr + integral_{T_ANCHOR}^{t} beta dt'
    h_model(tau) = alpha * exp(i*phi0) * h_BHPT

12 params: [p0, w, alpha_i, alpha_E, alpha_J, alpha_F, alpha_G,
            beta_i, beta_r, t0_nr, phi0, beta_L]

Motivation: the cumulative dE, dJ are monotonic (e_hat ~ j_hat) so a gated flux
alpha cannot represent the non-monotonic merger structure of |h_NR|/|h_BHPT|.
The instantaneous fluxes ARE non-monotonic (peak at merger) and, since
flux_e ~ omega(t)*flux_j, F_e and F_j have distinct time profiles — two
independent non-monotonic coordinates.  Adding them sharpens alpha through the
merger (per-q spot check: 15-43% better than gated wf_nu at q=3,5,8).

Everything that works in gated wf_nu is preserved: the switch stays (so p0,w
remain anchored by alpha's flux term), beta unchanged (safe time-map).  The new
couplings are pointwise on alpha, so imperfect regression cannot compound.

Same waveform-flux loss coordinates + PP-anchored nu regression as wf_nu_q_dep.
Warm-started from wf_nu_q_dep cache (alpha_F=alpha_G=0 => exactly gated wf_nu).
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

get_nu            = wfnu.get_nu
fit_poly_anchored = wfnu.fit_poly_anchored
eval_poly         = wfnu.eval_poly

PARAM_NAMES = [
    "p0", "w", "alpha_i", "alpha_E", "alpha_J", "alpha_F", "alpha_G",
    "beta_i", "beta_r", "t0_nr", "phi0", "beta_L",
]

RESULTS_DIR = ROOT / "wf_nu_fluxes_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH      = RESULTS_DIR / "per_q_cache.json"
WFNU_CACHE_PATH = ROOT / "wf_nu_q_dep_results" / "per_q_cache.json"
MD_PATH         = ROOT / "scaling_wf_nu_fluxes.md"

MODE           = creative.MODE
T_ANCHOR       = creative.T_ANCHOR
MIN_COVERAGE   = qdep.MIN_COVERAGE
MASTER_OUTLIER_CUT = qdep.MASTER_OUTLIER_CUT
WAVEFORM_CACHE_DIR = qdep.WAVEFORM_CACHE_DIR

PP_ANCHORS: dict[str, float | None] = {
    "p0":      None, "w":       None,
    "alpha_i": 1.0,  "alpha_E": 0.0, "alpha_J": 0.0, "alpha_F": 0.0, "alpha_G": 0.0,
    "beta_i":  1.0,  "beta_r":  1.0,
    "t0_nr":   None, "phi0":    None, "beta_L": 0.0,
}


# ---------------------------------------------------------------------------
# Case loading: wf_nu load_case + stored scalar flux-peak normalization
# ---------------------------------------------------------------------------

def load_case(q, source_stride=3, nr_stride=5):
    """wfnu.load_case, then inject scalar flux peaks so the fit-grid and
    full-grid use the identical normalization (a strided subsample can miss
    the exact peak sample)."""
    case = wfnu.load_case(q, source_stride, nr_stride)
    fe_peak = float(np.max(case["losses"]["flux_e"]))
    fj_peak = float(np.max(case["losses"]["flux_j"]))
    for d in (case["losses"], case["fit_losses"]):
        d["fe_peak"] = fe_peak
        d["fj_peak"] = fj_peak
    return case


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model_fluxes(
    params, t_bhpt, h_bhpt, t_nr, h_nr, losses, min_coverage=MIN_COVERAGE
):
    (p0, w, alpha_i, alpha_E, alpha_J, alpha_F, alpha_G,
     beta_i, beta_r, t0_nr, phi0, beta_L) = [float(v) for v in params]

    p_loss = losses["p_loss"]
    e_hat  = losses["e_hat"]
    j_hat  = losses["j_hat"]

    S  = creative.sigmoid((p_loss - p0) / w)
    dp = p_loss - p0
    e0 = float(np.interp(p0, p_loss, e_hat))
    j0 = float(np.interp(p0, p_loss, j_hat))
    dE = e_hat - e0
    dJ = j_hat - j0
    F_e = losses["flux_e"] / losses["fe_peak"]
    F_j = losses["flux_j"] / losses["fj_peak"]

    alpha = alpha_i + S * (alpha_E * dE + alpha_J * dJ) + alpha_F * F_e + alpha_G * F_j
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
        "tau":      tau, "alpha": alpha, "beta": beta,
        "e_hat":    np.interp(t_common, tau, e_hat),
        "j_hat":    np.interp(t_common, tau, j_hat),
        "h_ref":    h_ref, "h_model": h_model, "common": nr_mask,
    }


# ---------------------------------------------------------------------------
# Per-q optimisation
# ---------------------------------------------------------------------------

def _warm_from_wfnu(p10):
    """wf_nu [p0,w,ai,aE,aJ,bi,br,t0,phi0,bL] -> fluxes (insert alpha_F=alpha_G=0)."""
    p = list(p10)
    return np.array([p[0], p[1], p[2], p[3], p[4], 0.0, 0.0, p[5], p[6], p[7], p[8], p[9]])


def _load_wfnu_cache():
    if not WFNU_CACHE_PATH.exists():
        return {}
    return json.loads(WFNU_CACHE_PATH.read_text())


def optimize_case_fluxes(q, source_stride=3, nr_stride=5, top_n=4, maxiter=9000):
    case = load_case(q, source_stride, nr_stride)
    meta = case["meta"]

    def fit_err(params):
        return evaluate_model_fluxes(
            params,
            case["fit_t_bhpt"], case["fit_h_bhpt"],
            case["fit_t_nr"],   case["fit_h_nr"],
            case["fit_losses"],
        )["error"]

    seeds = []
    wfnu_cache = _load_wfnu_cache()
    for qk in (q, 3.0, 5.0):
        key = f"{qk:.10f}"
        if key in wfnu_cache:
            base = _warm_from_wfnu(wfnu_cache[key]["params"])
            seeds.append(base.copy())
            for fF, fG in [(-0.05, 0.0), (0.05, 0.0), (0.0, -0.05), (0.0, 0.05),
                           (-0.2, 0.1), (0.1, -0.05)]:
                s = base.copy(); s[5] = fF; s[6] = fG; seeds.append(s)

    rng   = np.random.default_rng(int(q * 677) % (2 ** 31))
    scale = (q / (1.0 + q)) / (5.0 / 6.0)
    for _ in range(40):
        bi = float(scale * rng.uniform(0.6, 1.05))
        seeds.append(np.array([
            float(rng.uniform(-1.3, -0.3)),                              # p0
            float(np.clip(rng.uniform(0.05, 0.3), creative.W_MIN, 0.4)),  # w
            float(scale * rng.uniform(0.6, 1.05)),                       # alpha_i
            float(rng.uniform(-0.5, 0.5)),                               # alpha_E
            float(rng.uniform(-0.5, 0.5)),                               # alpha_J
            float(rng.uniform(-0.25, 0.25)),                             # alpha_F
            float(rng.uniform(-0.25, 0.25)),                             # alpha_G
            bi,                                                          # beta_i
            float(scale * rng.uniform(0.7, 1.15)),                       # beta_r
            float(meta["t_nr_merger"] - bi * (meta["t_bhpt_merger"] - T_ANCHOR)),  # t0_nr
            float(rng.uniform(-np.pi, np.pi)),                           # phi0
            float(rng.uniform(-0.01, 0.01)),                             # beta_L
        ]))

    ranked = []
    for s in seeds:
        e = fit_err(s)
        if e < 10.0:
            ranked.append((e, s.copy()))
    ranked.sort(key=lambda x: x[0])
    if not ranked:
        print(f"  q={q:.4g}  fluxes: NO valid seeds", flush=True)
        return {"q": q, "error": 50.0, "params": [0.0] * 12, "coverage": float("nan")}

    best_err    = ranked[0][0]
    best_params = ranked[0][1].copy()
    for _, start in ranked[:top_n]:
        res = minimize(fit_err, start, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-13})
        if res.fun < best_err:
            best_err    = float(res.fun)
            best_params = res.x.copy()

    full = evaluate_model_fluxes(
        best_params, case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"], case["losses"])
    print(f"  q={q:.4g}  fluxes: fit={best_err:.4g}  full={full.get('error', 50.):.4g}"
          f"  aF={best_params[5]:+.4f} aG={best_params[6]:+.4f}", flush=True)
    return {"q": q, "error": float(full.get("error", 50.0)),
            "params": best_params.tolist(), "coverage": float(full.get("coverage", float("nan")))}


def optimize_case_worker(args):
    q, source_stride, nr_stride, top_n, maxiter = args
    return optimize_case_fluxes(q, source_stride, nr_stride, top_n, maxiter)


# ---------------------------------------------------------------------------
# Regression / master
# ---------------------------------------------------------------------------

def fit_master(rows, degree):
    nu_arr = np.array([get_nu(r["q"]) for r in rows])
    params_arr = np.array([r["params"] for r in rows], dtype=float)
    coeffs = {}
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
    params[7]  = float(np.clip(params[7],  0.2,  1.6))            # beta_i
    params[8]  = float(np.clip(params[8],  0.2,  1.6))            # beta_r
    params[11] = float(np.clip(params[11], -0.05, 0.05))          # beta_L
    return params


def polish_nuisance(params, case):
    def obj(x):
        trial = params.copy()
        trial[9] = x[0]; trial[10] = x[1]
        return evaluate_model_fluxes(
            trial, case["fit_t_bhpt"], case["fit_h_bhpt"],
            case["fit_t_nr"], case["fit_h_nr"], case["fit_losses"])["error"]
    result = minimize(obj, [params[9], params[10]], method="Nelder-Mead",
                      options={"maxiter": 400, "xatol": 1e-7, "fatol": 1e-9})
    out = params.copy()
    out[9] = float(result.x[0]); out[10] = float(result.x[1])
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
        ev = evaluate_model_fluxes(
            params, case["t_bhpt"], case["h_bhpt"],
            case["t_nr"], case["h_nr"], case["losses"])
        out.append({"q": q, "params": [float(x) for x in params], "fit_error": fit_err,
                    "error": float(ev.get("error", 50.0)),
                    "coverage": float(ev.get("coverage", float("nan")))})
    return out


def _eval_master_worker(args):
    q, source_stride, nr_stride, do_polish, coeffs2, coeffs3, degree = args
    coeffs = coeffs3 if degree == 3 else coeffs2
    return evaluate_master([{"q": q}], coeffs, source_stride, nr_stride, do_polish)[0]


def summarize(values):
    a = np.array(values, dtype=float); a = a[np.isfinite(a)]
    return {"min": float(np.min(a)), "median": float(np.median(a)),
            "mean": float(np.mean(a)), "max": float(np.max(a))}


def should_use_cubic(s2, s3):
    med_gain = (s2["median"] - s3["median"]) / max(s2["median"], 1e-15)
    max_gain = (s2["max"]    - s3["max"])    / max(s2["max"],    1e-15)
    return bool(med_gain > 0.03 or max_gain > 0.03)


def q_key(q): return f"{q:.10f}"
def load_cache():
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
def save_cache(cache): CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_cached_q_values():
    import re
    pat = re.compile(r"waveforms_q(.+)\.npz")
    qs = []
    for p in WAVEFORM_CACHE_DIR.glob("waveforms_q*.npz"):
        m = pat.match(p.name)
        if m:
            qs.append(float(m.group(1)))
    return sorted(qs)


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
    s2 = summarize([r["error"] for r in master2])
    s3 = summarize([r["error"] for r in master3])
    s_sel = summarize([r["error"] for r in selected])
    q_at_max = max(selected, key=lambda r: r["error"])["q"]
    nu_q3 = get_nu(3.0); nu_q8 = get_nu(8.0); nu_q2 = get_nu(2.0)
    anchored = [n for n, a in PP_ANCHORS.items() if a is not None]
    free     = [n for n, a in PP_ANCHORS.items() if a is None]

    lines = [
        "# Instantaneous-flux wf-nu q-dependent scaling fit (wf_nu_fluxes)",
        "",
        "Gated wf_nu + two instantaneous-flux terms on alpha (alpha_F*F_e + alpha_G*F_j).",
        "12 params.  F_e=flux_e/max, F_j=flux_j/max (peak-normalized, peak ~at merger).",
        "",
        "## Model equations",
        "",
        "```python",
        "S  = 1 / (1 + exp(-(p_loss - p0) / w))",
        "dE = e_hat - e_hat(p0);  dJ = j_hat - j_hat(p0);  dp = p_loss - p0",
        "F_e = flux_e / max(flux_e);  F_j = flux_j / max(flux_j)",
        "alpha = alpha_i + S*(alpha_E*dE + alpha_J*dJ) + alpha_F*F_e + alpha_G*F_j",
        "beta  = (1-S)*(beta_i + beta_L*dp) + S*beta_r",
        "tau(t) = t0_nr + integral_{T_ANCHOR}^{t} beta dt'",
        "h_model(tau) = alpha * exp(i*phi0) * h_BHPT",
        "```",
        "",
        "## Regression",
        "",
        "```python",
        "nu(q) = q / (1 + q)**2",
        f"# Anchored (PP limit nu->0): {', '.join(anchored)}",
        f"# Unanchored:               {', '.join(free)}",
        "```",
        "",
        f"Training grid: {len(rows)} q values in "
        f"[{min(r['q'] for r in rows):.4g}, {max(r['q'] for r in rows):.4g}].",
        f"nu training range: [{nu_q8:.4f}, {nu_q3:.4f}].  Extrapolation to q=2: nu={nu_q2:.4f}.",
        f"Rows used in regression: {len(fit_rows)}.  Selected degree: **{selected_degree}**",
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
        "| q | nu | per-q E | master E | coverage | p0 | w | alpha_F | alpha_G | beta_i | beta_r |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sel_by_q = {f"{r['q']:.10f}": r for r in selected}
    fit_keys = {f"{r['q']:.10f}" for r in fit_rows}
    for row in rows:
        key = f"{row['q']:.10f}"; p = row["params"]
        mr  = sel_by_q.get(key, {"error": float("nan"), "coverage": float("nan")})
        reg = "✓" if key in fit_keys else ""
        # p: 0p0 1w 2ai 3aE 4aJ 5aF 6aG 7bi 8br 9t0 10phi0 11bL
        lines.append(
            f"| {row['q']:.6g} | {get_nu(float(row['q'])):.4f} | {row['error']:.4g}"
            f" | {mr['error']:.4g} | {mr['coverage']:.4f} | {reg}"
            f" | {p[0]:.4g} | {p[1]:.4g} | {p[5]:.6g} | {p[6]:.6g} | {p[7]:.4g} | {p[8]:.4g} |")
    lines.append("")
    MD_PATH.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-min",              type=float, default=3.0)
    parser.add_argument("--q-max",              type=float, default=8.0)
    parser.add_argument("--source-stride",      type=int,   default=3)
    parser.add_argument("--nr-stride",          type=int,   default=5)
    parser.add_argument("--top-n",              type=int,   default=4)
    parser.add_argument("--maxiter",            type=int,   default=9000)
    parser.add_argument("--master-outlier-cut", type=float, default=MASTER_OUTLIER_CUT)
    parser.add_argument("--workers",            type=int,   default=4)
    parser.add_argument("--force",              action="store_true")
    args = parser.parse_args()

    all_q = [q for q in get_cached_q_values() if args.q_min - 1e-9 <= q <= args.q_max + 1e-9]
    if not all_q:
        raise RuntimeError("No waveforms found in .cache/q_dep/.")
    print(f"Training grid: {len(all_q)} q values in [{all_q[0]:.4g}, {all_q[-1]:.4g}]", flush=True)

    cache = {} if args.force else load_cache()
    todo  = [q for q in all_q if q_key(q) not in cache]
    if todo:
        print(f"Optimising {len(todo)} q values ({args.workers} workers) ...", flush=True)
        opt_args = [(q, args.source_stride, args.nr_stride, args.top_n, args.maxiter) for q in todo]
        with Pool(args.workers) as pool:
            results = pool.map(optimize_case_worker, opt_args)
        for row in results:
            cache[q_key(row["q"])] = row
        save_cache(cache)
    else:
        print("All per-q results already cached.", flush=True)

    rows = sorted([v for v in cache.values()
                   if all_q[0] - 1e-9 <= v["q"] <= all_q[-1] + 1e-9], key=lambda r: r["q"])

    fit_rows = [r for r in rows
                if np.isfinite(r.get("coverage", float("nan"))) and r["error"] < args.master_outlier_cut]
    if len(fit_rows) < 8:
        raise RuntimeError(f"Too few valid rows ({len(fit_rows)}).")
    print(f"Fitting polynomials on {len(fit_rows)} rows ...", flush=True)

    coeffs2 = fit_master(fit_rows, degree=2)
    coeffs3 = fit_master(fit_rows, degree=3)

    print("Evaluating master model ...", flush=True)
    eval_args = [(r["q"], args.source_stride, args.nr_stride, True, coeffs2, coeffs3, deg)
                 for r in rows for deg in (2, 3)]
    with Pool(args.workers) as pool:
        all_results = pool.map(_eval_master_worker, eval_args)
    master2 = all_results[0::2]; master3 = all_results[1::2]

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
