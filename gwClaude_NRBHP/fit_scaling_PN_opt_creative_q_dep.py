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

import fit_scaling_PN_opt_creative as creative  # noqa: E402

PARAM_NAMES = ["p0", "w", "alpha_i", "alpha_E", "alpha_J",
               "beta_i", "beta_r", "t0_nr", "phi0", "beta_L"]

RESULTS_DIR = ROOT / "PN_opt_creative_q_dep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "per_q_cache.json"
WAVEFORM_CACHE_DIR = ROOT / ".cache" / "q_dep"
WAVEFORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MD_PATH = ROOT / "scaling_PN_opt_creative_q_dep.md"

MODE = creative.MODE
NR_T_START = creative.NR_T_START
NR_T_END = creative.NR_T_END
T_ANCHOR = creative.T_ANCHOR
MIN_COVERAGE = 0.88
MASTER_OUTLIER_CUT = 2.0e-3


# ---------------------------------------------------------------------------
# Waveform caching
# ---------------------------------------------------------------------------

def waveform_cache_path(q: float) -> Path:
    return WAVEFORM_CACHE_DIR / f"waveforms_q{q:.10f}.npz"


def generate_and_cache_waveform(q: float) -> None:
    path = waveform_cache_path(q)
    if path.exists():
        return
    import gwsurrogate
    import BHPTNRSur1dq1e4 as bhptsur  # noqa: F401
    import warnings as _w

    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    t_bhpt, h_bhpt_dict = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=[MODE], neg_modes=False
    )
    h_bhpt = h_bhpt_dict[MODE]
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        t_nr, h_nr_dict, _ = nrsur(q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    nr_mask = (t_nr >= NR_T_START) & (t_nr <= NR_T_END)
    t_nr = t_nr[nr_mask]
    h_nr = h_nr_dict[MODE][nr_mask]

    nu = q / (1.0 + q) ** 2
    t_bhpt_merger = float(t_bhpt[np.argmax(np.abs(h_bhpt))])
    t_nr_merger = float(t_nr[np.argmax(np.abs(h_nr))])

    np.savez(
        path,
        t_bhpt=t_bhpt,
        h_bhpt_re=h_bhpt.real,
        h_bhpt_im=h_bhpt.imag,
        t_nr=t_nr,
        h_nr_re=h_nr.real,
        h_nr_im=h_nr.imag,
        nu=np.float64(nu),
        t_bhpt_merger=np.float64(t_bhpt_merger),
        t_nr_merger=np.float64(t_nr_merger),
    )
    print(f"cached waveforms q={q:.6g}", flush=True)


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def subsample_losses(losses: dict, idx: np.ndarray) -> dict:
    out = {}
    for key, val in losses.items():
        out[key] = val[idx] if isinstance(val, np.ndarray) else val
    return out


def load_case(q: float, source_stride: int = 3, nr_stride: int = 5) -> dict:
    d = np.load(waveform_cache_path(q))
    t_bhpt = d["t_bhpt"]
    h_bhpt = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    t_nr = d["t_nr"]
    h_nr = d["h_nr_re"] + 1j * d["h_nr_im"]
    meta = {
        "nu": float(d["nu"]),
        "t_bhpt_merger": float(d["t_bhpt_merger"]),
        "t_nr_merger": float(d["t_nr_merger"]),
    }

    losses_full = creative.pn_loss_coordinates(t_bhpt, h_bhpt, meta)

    # subsampled arrays for the optimizer
    src_idx = np.unique(np.r_[np.arange(0, len(t_bhpt), source_stride), len(t_bhpt) - 1])
    nr_idx = np.unique(np.r_[np.arange(0, len(t_nr), nr_stride), len(t_nr) - 1])

    t_bhpt_fit = t_bhpt[src_idx]
    h_bhpt_fit = h_bhpt[src_idx]
    losses_fit = subsample_losses(losses_full, src_idx)

    return {
        "q": q,
        "t_bhpt": t_bhpt,
        "h_bhpt": h_bhpt,
        "t_nr": t_nr,
        "h_nr": h_nr,
        "meta": meta,
        "losses": losses_full,
        "fit_t_bhpt": t_bhpt_fit,
        "fit_h_bhpt": h_bhpt_fit,
        "fit_losses": losses_fit,
        "fit_t_nr": t_nr[nr_idx],
        "fit_h_nr": h_nr[nr_idx],
    }


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def anchored_start(params: np.ndarray, meta: dict) -> np.ndarray:
    """Recompute t0_nr so tau(T_ANCHOR) ≈ maps BHPT merger to NR merger."""
    out = params.copy()
    p = creative.unpack(out)
    # tau(t_bhpt_merger) ≈ t0_nr + beta_i * (t_bhpt_merger - T_ANCHOR)
    # set that equal to t_nr_merger
    out[7] = meta["t_nr_merger"] - p["beta_i"] * (meta["t_bhpt_merger"] - T_ANCHOR)
    return out


def q_scaled_seed(q: float, meta: dict) -> np.ndarray:
    """Scale amplitude/time-stretch params by mass-ratio factor relative to q=5."""
    scale = (q / (1.0 + q)) / (5.0 / 6.0)
    seed = creative.ACCEPTED_CREATIVE_PARAMS.copy()
    seed[2] *= scale   # alpha_i
    seed[5] *= scale   # beta_i
    seed[6] *= scale   # beta_r
    return anchored_start(seed, meta)


def phase_seed(params: np.ndarray, case: dict) -> np.ndarray:
    trial = params.copy()
    trial[8] = 0.0
    ev = creative.evaluate_model(
        trial,
        case["fit_t_bhpt"], case["fit_h_bhpt"],
        case["fit_t_nr"], case["fit_h_nr"],
        case["fit_losses"], min_coverage=0.82,
    )
    if ev["error"] >= 10.0:
        return params.copy()
    phi_est = float(np.angle(np.sum(ev["h_ref"] * ev["h_model"].conjugate())))
    out = params.copy()
    out[8] = phi_est
    return out


def candidate_starts(case: dict) -> list[np.ndarray]:
    meta = case["meta"]
    q = case["q"]

    base_anchored = anchored_start(creative.ACCEPTED_CREATIVE_PARAMS, meta)
    scaled = q_scaled_seed(q, meta)

    candidates = [base_anchored, scaled]

    # perturbations of scaled seed
    for p0_shift in (-0.08, 0.0, 0.08):
        for w_scale in (0.7, 1.0, 1.4):
            for bl_scale in (0.5, 1.0, 1.5):
                c = scaled.copy()
                c[0] += p0_shift
                c[1] = float(np.clip(c[1] * w_scale, creative.W_MIN, 0.45))
                c[9] *= bl_scale
                candidates.append(anchored_start(c, meta))

    valid = [c for c in candidates if creative.validate_basic(c)]
    seeded = [phase_seed(c, case) for c in valid]
    return seeded


# ---------------------------------------------------------------------------
# Per-q optimizer
# ---------------------------------------------------------------------------

def _fit_error(params, case):
    return creative.evaluate_model(
        params,
        case["fit_t_bhpt"], case["fit_h_bhpt"],
        case["fit_t_nr"], case["fit_h_nr"],
        case["fit_losses"], min_coverage=MIN_COVERAGE,
    )["error"]


def optimize_case(case: dict, top_n: int, maxiter: int) -> dict:
    q = case["q"]
    starts = candidate_starts(case)

    ranked = []
    for s in starts:
        err = _fit_error(s, case)
        if err < 10.0:
            ranked.append((err, s))
    ranked.sort(key=lambda x: x[0])

    if not ranked:
        return {"q": q, "params": creative.ACCEPTED_CREATIVE_PARAMS.tolist(),
                "fit_error": 50.0, "error": 50.0, "coverage": float("nan")}

    best_params = ranked[0][1].copy()
    best_fit_err = ranked[0][0]

    for _rank, (start_err, start) in enumerate(ranked[:top_n]):
        result = minimize(
            lambda p: _fit_error(p, case),
            start,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-12},
        )
        if result.fun < best_fit_err:
            best_fit_err = float(result.fun)
            best_params = result.x.copy()

    full = creative.evaluate_model(
        best_params,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"],
        case["losses"], min_coverage=MIN_COVERAGE,
    )
    print(f"q={q:.6g} fit={best_fit_err:.6g} full={full['error']:.6g}", flush=True)
    return {
        "q": q,
        "params": [float(x) for x in best_params],
        "fit_error": float(best_fit_err),
        "error": float(full["error"]),
        "coverage": float(full.get("coverage", float("nan"))),
    }


def optimize_case_worker(args: tuple) -> dict:
    """Top-level worker for Pool.map (must be picklable)."""
    q, source_stride, nr_stride, top_n, maxiter = args
    case = load_case(q, source_stride, nr_stride)
    return optimize_case(case, top_n, maxiter)


# ---------------------------------------------------------------------------
# Polynomial fitting in 1/q
# ---------------------------------------------------------------------------

def design(q_values: np.ndarray, degree: int) -> np.ndarray:
    y = 1.0 / q_values
    return np.vstack([y ** k for k in range(degree + 1)]).T


def fit_poly(q_values: np.ndarray, values: np.ndarray, degree: int) -> np.ndarray:
    return np.linalg.lstsq(design(q_values, degree), values, rcond=None)[0]


def eval_poly(q: float, coeffs: np.ndarray) -> float:
    y = 1.0 / q
    return float(sum(coeffs[i] * y ** i for i in range(len(coeffs))))


def fit_master(rows: list[dict], degree: int) -> dict[str, np.ndarray]:
    q_arr = np.array([r["q"] for r in rows], dtype=float)
    params = np.array([r["params"] for r in rows], dtype=float)
    coeffs: dict[str, np.ndarray] = {}
    for i, name in enumerate(PARAM_NAMES):
        vals = params[:, i].copy()
        if name == "phi0":
            vals = np.unwrap(vals)
        coeffs[name] = fit_poly(q_arr, vals, degree)
    return coeffs


def constrained_master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    params = np.array([eval_poly(q, coeffs[name]) for name in PARAM_NAMES], dtype=float)
    params[0] = float(np.clip(params[0], -3.0, 0.05))    # p0
    params[1] = float(np.clip(params[1], creative.W_MIN, 0.45))  # w
    params[2] = float(np.clip(params[2], 0.05, 2.5))     # alpha_i
    params[5] = float(np.clip(params[5], 0.2, 1.6))      # beta_i
    params[6] = float(np.clip(params[6], 0.2, 1.6))      # beta_r
    params[9] = float(np.clip(params[9], -0.05, 0.05))   # beta_L
    return params


# ---------------------------------------------------------------------------
# Master model evaluation (polish t0_nr and phi0 as nuisance)
# ---------------------------------------------------------------------------

def polish_nuisance(params: np.ndarray, case: dict) -> tuple[np.ndarray, float]:
    def obj(x: np.ndarray) -> float:
        trial = params.copy()
        trial[7] = x[0]   # t0_nr
        trial[8] = x[1]   # phi0
        return creative.evaluate_model(
            trial,
            case["fit_t_bhpt"], case["fit_h_bhpt"],
            case["fit_t_nr"], case["fit_h_nr"],
            case["fit_losses"], min_coverage=MIN_COVERAGE,
        )["error"]

    result = minimize(
        obj, [params[7], params[8]],
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-7, "fatol": 1e-9},
    )
    out = params.copy()
    out[7] = float(result.x[0])
    out[8] = float(result.x[1])
    return out, float(result.fun)


def evaluate_master(
    rows: list[dict],
    coeffs: dict[str, np.ndarray],
    source_stride: int,
    nr_stride: int,
    do_polish: bool,
) -> list[dict]:
    out = []
    for row in rows:
        q = row["q"]
        params = constrained_master_params(q, coeffs)
        case = load_case(q, source_stride, nr_stride)
        fit_err = float("nan")
        if do_polish:
            params, fit_err = polish_nuisance(params, case)
        ev = creative.evaluate_model(
            params,
            case["t_bhpt"], case["h_bhpt"],
            case["t_nr"], case["h_nr"],
            case["losses"], min_coverage=MIN_COVERAGE,
        )
        out.append({
            "q": q,
            "params": [float(x) for x in params],
            "fit_error": fit_err,
            "error": float(ev["error"]),
            "coverage": float(ev.get("coverage", float("nan"))),
        })
    return out


def _eval_master_worker(args: tuple) -> dict:
    """Module-level worker for parallel master evaluation (picklable)."""
    q, source_stride, nr_stride, do_polish, coeffs2, coeffs3, degree = args
    coeffs = coeffs3 if degree == 3 else coeffs2
    result = evaluate_master([{"q": q}], coeffs, source_stride, nr_stride, do_polish)
    return result[0]


# ---------------------------------------------------------------------------
# Degree selection and markdown
# ---------------------------------------------------------------------------

def summarize(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    return {"min": float(np.min(a)), "median": float(np.median(a)),
            "mean": float(np.mean(a)), "max": float(np.max(a))}


def should_use_cubic(s2: dict, s3: dict) -> bool:
    med_gain = (s2["median"] - s3["median"]) / max(s2["median"], 1e-15)
    max_gain = (s2["max"] - s3["max"]) / max(s2["max"], 1e-15)
    return bool(med_gain > 0.03 or max_gain > 0.03)


def coeff_table_lines(coeffs: dict[str, np.ndarray], degree: int) -> list[str]:
    header = "| parameter | " + " | ".join(f"c{i}" for i in range(degree + 1)) + " |"
    sep = "|:---|" + "|".join("---:" for _ in range(degree + 1)) + "|"
    lines = [header, sep]
    for name in PARAM_NAMES:
        row = " | ".join(f"{v:.12g}" for v in coeffs[name])
        lines.append(f"| {name} | {row} |")
    return lines


def write_markdown(
    rows: list[dict],
    fit_rows: list[dict],
    master2: list[dict],
    master3: list[dict],
    coeffs2: dict[str, np.ndarray],
    coeffs3: dict[str, np.ndarray],
    selected_degree: int,
    args: argparse.Namespace,
) -> None:
    selected = master3 if selected_degree == 3 else master2
    selected_coeffs = coeffs3 if selected_degree == 3 else coeffs2

    valid_rows = [r for r in rows if np.isfinite(r.get("coverage", float("nan"))) and r["error"] < 10.0]
    s_perq = summarize([r["error"] for r in valid_rows])
    s_fit = summarize([r["error"] for r in fit_rows])
    s2 = summarize([r["error"] for r in master2])
    s3 = summarize([r["error"] for r in master3])
    s_sel = summarize([r["error"] for r in selected])
    q_at_max = max(selected, key=lambda r: r["error"])["q"]

    lines = [
        "# Creative PN-loss q-dependent scaling fit",
        "",
        "This fit generalises the creative 10-parameter logistic-switch model over",
        "mass ratios 3 ≤ q ≤ 8. The per-q model architecture is identical to the",
        "q=5 creative fit in `gwClaude_NRBHP`:",
        "",
        "```python",
        "S = 1 / (1 + exp(-(p_loss - p0) / w))",
        "dE = Ehat - Ehat(p0);  dJ = Jhat - Jhat(p0);  dp = p_loss - p0",
        "alpha = alpha_i + S * (alpha_E * dE + alpha_J * dJ)",
        "beta  = (1 - S) * (beta_i + beta_L * dp) + S * beta_r",
        "tau(t) = t0_nr + integral_{t_anchor}^{t} beta(t') dt'",
        "h_model(tau(t)) = alpha * exp(i*phi0) * h_BHPT(t)",
        "```",
        "",
        "The q-dependence of all 10 parameters is represented as polynomials in `y = 1/q`:",
        "",
        "```python",
        "C(q) = c0 + c1/q + c2/q**2 [+ c3/q**3]",
        "```",
        "",
        f"Calibration grid: `{len(rows)}` q values from"
        f" `{min(r['q'] for r in rows):.4g}` to `{max(r['q'] for r in rows):.4g}`.",
        f"Rows used for regression: `{len(fit_rows)}` (outliers with"
        f" mathcalE ≥ {args.master_outlier_cut:g} excluded).",
        f"Fit settings: source stride `{args.source_stride}`, NR stride `{args.nr_stride}`,"
        f" top starts `{args.top_n}`, maxiter `{args.maxiter}`, workers `{args.workers}`.",
        "",
        "Nuisance parameters `t0_nr` and `phi0` are re-optimised when evaluating",
        "the master model so that the error statistics reflect only the q-polynomial",
        "error, not arbitrary time/phase misalignment.",
        "",
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
        f"| per-q independent (valid) | {s_perq['min']:.6g} | {s_perq['median']:.6g}"
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
        "| q | per-q mathcalE | master mathcalE | coverage | in regression |"
        " p0 | w | beta_i | alpha_i | beta_L |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]

    sel_by_q = {f"{r['q']:.10f}": r for r in selected}
    fit_keys = {f"{r['q']:.10f}" for r in fit_rows}
    for row in rows:
        key = f"{row['q']:.10f}"
        p = row["params"]
        mr = sel_by_q.get(key, {"error": float("nan"), "coverage": float("nan")})
        lines.append(
            f"| {row['q']:.6g} | {row['error']:.6g} | {mr['error']:.6g}"
            f" | {mr['coverage']:.6g} | {key in fit_keys}"
            f" | {p[0]:.6g} | {p[1]:.6g} | {p[5]:.6g} | {p[2]:.6g} | {p[9]:.8g} |"
        )
    lines.append("")
    MD_PATH.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def q_key(q: float) -> str:
    return f"{q:.10f}"


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-q", type=int, default=40)
    parser.add_argument("--q-min", type=float, default=3.0)
    parser.add_argument("--q-max", type=float, default=8.0)
    parser.add_argument("--source-stride", type=int, default=3)
    parser.add_argument("--nr-stride", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--master-outlier-cut", type=float, default=MASTER_OUTLIER_CUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    q_values = np.linspace(args.q_min, args.q_max, args.n_q)

    # ---- Phase 1: parallel waveform caching --------------------------------
    need_cache = [q for q in q_values if not waveform_cache_path(q).exists()]
    if need_cache:
        print(f"Generating waveforms for {len(need_cache)} q values ...", flush=True)
        with Pool(args.workers) as pool:
            pool.map(generate_and_cache_waveform, need_cache)
    else:
        print("All waveforms already cached.", flush=True)

    # ---- Phase 2: parallel per-q optimization ------------------------------
    cache = {} if args.force else load_cache()
    todo = [q for q in q_values if q_key(q) not in cache]
    if todo:
        print(f"Optimising {len(todo)} q values with {args.workers} workers ...", flush=True)
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

    rows = [cache[q_key(float(q))] for q in sorted(q_values)]

    # ---- Phase 3: polynomial fitting ---------------------------------------
    fit_rows = [
        r for r in rows
        if np.isfinite(r.get("coverage", float("nan"))) and r["error"] < args.master_outlier_cut
    ]
    if len(fit_rows) < 8:
        raise RuntimeError("Too few valid rows for polynomial regression.")
    print(f"Fitting polynomials on {len(fit_rows)} rows ...", flush=True)

    coeffs2 = fit_master(fit_rows, degree=2)
    coeffs3 = fit_master(fit_rows, degree=3)

    # ---- Phase 4: parallel master model evaluation -------------------------
    print("Evaluating master model ...", flush=True)
    eval_args = [
        (r["q"], args.source_stride, args.nr_stride, True, coeffs2, coeffs3, deg)
        for r in rows
        for deg in (2, 3)
    ]
    with Pool(args.workers) as pool:
        all_results = pool.map(_eval_master_worker, eval_args)

    # interleaved: deg2, deg3, deg2, deg3, ...
    master2 = all_results[0::2]
    master3 = all_results[1::2]

    s2 = summarize([r["error"] for r in master2])
    s3 = summarize([r["error"] for r in master3])
    selected_degree = 3 if should_use_cubic(s2, s3) else 2

    print(f"quadratic: median={s2['median']:.6g} max={s2['max']:.6g}")
    print(f"cubic:     median={s3['median']:.6g} max={s3['max']:.6g}")
    print(f"selected_degree={selected_degree}")

    write_markdown(rows, fit_rows, master2, master3, coeffs2, coeffs3, selected_degree, args)
    print(f"Wrote {MD_PATH}", flush=True)


if __name__ == "__main__":
    main()
