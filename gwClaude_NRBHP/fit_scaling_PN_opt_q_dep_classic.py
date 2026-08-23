"""
Constant alpha-beta q-dependent scaling fit  (identifier: q_dep_classic).

Replicates the Islam et al. 2204.01972 (BHPTNRSurrogate) model class:
two physical parameters per q — a constant amplitude scaling (alpha) and a
constant time-stretch (beta) — represented as PP-anchored quartic polynomials
in 1/q.  Alignment nuisances (t0_nr, phi0) are re-optimised at evaluation.

Model:
    tau(t)       = t0_nr + beta * (t - T_ANCHOR)     T_ANCHOR = -100 M
    h_model(tau) = alpha * exp(i*phi0) * h_BHPT(t)

alpha(q) = c0 + c1/q + c2/q^2 + ... + c_d/q^d   [c0 = 1, PP anchor]
beta(q)  = c0 + c1/q + c2/q^2 + ... + c_d/q^d   [c0 = 1, PP anchor]

Reuses the waveform cache from .cache/q_dep/ (shared with PN_opt_creative_q_dep).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTNRSurrogate" / "surrogates"))

warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative_q_dep as qdep  # waveform cache + generation only

PHYS_NAMES = ["alpha", "beta"]

NR_T_START   = qdep.NR_T_START   # -5000.1 M
NR_T_END     = qdep.NR_T_END     #   100.0 M
T_ANCHOR     = qdep.T_ANCHOR     #  -100.0 M in BHPT time
MIN_COVERAGE = 0.88
MASTER_OUTLIER_CUT = 5.0e-3

RESULTS_DIR       = ROOT / "PN_opt_q_dep_classic_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH        = RESULTS_DIR / "per_q_cache.json"
WAVEFORM_CACHE_DIR = qdep.WAVEFORM_CACHE_DIR   # .cache/q_dep/
MD_PATH           = ROOT / "scaling_PN_opt_q_dep_classic.md"


# ---------------------------------------------------------------------------
# Waveform loading  (reuses qdep npz cache, no PN-loss coordinates needed)
# ---------------------------------------------------------------------------

def waveform_cache_path(q: float) -> Path:
    return qdep.waveform_cache_path(q)


def load_case(q: float, nr_stride: int = 5) -> dict:
    d      = np.load(waveform_cache_path(q))
    t_bhpt = d["t_bhpt"]
    h_bhpt = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    t_nr   = d["t_nr"]
    h_nr   = d["h_nr_re"]   + 1j * d["h_nr_im"]
    meta   = {
        "t_bhpt_merger": float(d["t_bhpt_merger"]),
        "t_nr_merger":   float(d["t_nr_merger"]),
    }
    nr_idx = np.unique(np.r_[np.arange(0, len(t_nr), nr_stride), len(t_nr) - 1])
    return {
        "q":        q,
        "t_bhpt":   t_bhpt,
        "h_bhpt":   h_bhpt,
        "t_nr":     t_nr,
        "h_nr":     h_nr,
        "meta":     meta,
        "fit_t_nr": t_nr[nr_idx],
        "fit_h_nr": h_nr[nr_idx],
    }


# ---------------------------------------------------------------------------
# Core model and mismatch
# ---------------------------------------------------------------------------

def mathcalE_error(h1: np.ndarray, h2: np.ndarray) -> float:
    n1 = float(np.sum(np.abs(h1) ** 2))
    n2 = float(np.sum(np.abs(h2) ** 2))
    sd = float(np.real(np.sum(h1 * h2.conjugate())))
    return (n1 + n2 - 2.0 * sd) / (2.0 * n1)


def interp_complex(t_src: np.ndarray, h_src: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    return (np.interp(t_eval, t_src, h_src.real)
            + 1j * np.interp(t_eval, t_src, h_src.imag))


def evaluate_model(
    alpha: float, beta: float, t0_nr: float, phi0: float,
    t_bhpt: np.ndarray, h_bhpt: np.ndarray,
    t_nr: np.ndarray, h_nr: np.ndarray,
    min_coverage: float = MIN_COVERAGE,
) -> dict:
    if alpha <= 0.0 or beta <= 0.0:
        return {"error": 50.0}
    tau = t0_nr + beta * (t_bhpt - T_ANCHOR)
    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    if np.count_nonzero(use) < 2000:
        return {"error": 50.0}
    common   = (t_nr >= tau[use][0]) & (t_nr <= tau[use][-1])
    coverage = float(np.count_nonzero(common)) / len(t_nr)
    if coverage < min_coverage:
        return {"error": 50.0 + (min_coverage - coverage)}
    h_scaled = alpha * np.exp(1j * phi0) * h_bhpt
    h_model  = interp_complex(tau[use], h_scaled[use], t_nr[common])
    return {
        "error":    mathcalE_error(h_nr[common], h_model),
        "coverage": coverage,
        "common":   common,
        "h_ref":    h_nr[common],
        "h_model":  h_model,
        "tau":      tau[use],
    }


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def _t0_seed(beta: float, meta: dict) -> float:
    return float(meta["t_nr_merger"] - beta * (meta["t_bhpt_merger"] - T_ANCHOR))


def _phi0_estimate(
    alpha: float, beta: float, t0_nr: float,
    t_bhpt: np.ndarray, h_bhpt: np.ndarray,
    fit_t_nr: np.ndarray, fit_h_nr: np.ndarray,
) -> float:
    tau = t0_nr + beta * (t_bhpt - T_ANCHOR)
    use = (tau >= fit_t_nr[0]) & (tau <= fit_t_nr[-1])
    if np.count_nonzero(use) < 200:
        return 0.0
    common = (fit_t_nr >= tau[use][0]) & (fit_t_nr <= tau[use][-1])
    h_nopha = interp_complex(tau[use], alpha * h_bhpt[use], fit_t_nr[common])
    return float(np.angle(np.sum(fit_h_nr[common] * h_nopha.conjugate())))


def candidate_starts(case: dict) -> list[np.ndarray]:
    meta   = case["meta"]
    seeds  = []
    alphas = [0.66, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
    betas  = [0.66, 0.71, 0.76, 0.81, 0.86]
    for alpha in alphas:
        for beta in betas:
            t0  = _t0_seed(beta, meta)
            phi = _phi0_estimate(alpha, beta, t0,
                                 case["t_bhpt"], case["h_bhpt"],
                                 case["fit_t_nr"], case["fit_h_nr"])
            seeds.append(np.array([alpha, beta, t0, phi]))
    return seeds


# ---------------------------------------------------------------------------
# Per-q optimizer
# ---------------------------------------------------------------------------

def _fit_error(params: np.ndarray, case: dict) -> float:
    return evaluate_model(
        params[0], params[1], params[2], params[3],
        case["t_bhpt"], case["h_bhpt"],
        case["fit_t_nr"], case["fit_h_nr"],
    )["error"]


def optimize_case(case: dict, top_n: int, maxiter: int) -> dict:
    q = case["q"]
    starts = candidate_starts(case)

    ranked: list[tuple[float, np.ndarray]] = []
    for s in starts:
        err = _fit_error(s, case)
        if err < 10.0:
            ranked.append((err, s))
    ranked.sort(key=lambda x: x[0])
    if not ranked:
        t0 = _t0_seed(0.80, case["meta"])
        ranked = [(10.0, np.array([0.80, 0.80, t0, 0.0]))]

    best_params = ranked[0][1].copy()
    best_err    = ranked[0][0]

    for _, start in ranked[:top_n]:
        result = minimize(
            lambda p: _fit_error(p, case),
            start,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-12},
        )
        if result.fun < best_err:
            best_err    = float(result.fun)
            best_params = result.x.copy()

    full = evaluate_model(
        best_params[0], best_params[1], best_params[2], best_params[3],
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"],
    )
    print(f"q={q:.6g}  fit={best_err:.6g}  full={full['error']:.6g}  "
          f"alpha={best_params[0]:.5g}  beta={best_params[1]:.5g}", flush=True)
    return {
        "q":         q,
        "alpha":     float(best_params[0]),
        "beta":      float(best_params[1]),
        "t0_nr":     float(best_params[2]),
        "phi0":      float(best_params[3]),
        "fit_error": float(best_err),
        "error":     float(full["error"]),
        "coverage":  float(full.get("coverage", float("nan"))),
    }


def optimize_case_worker(args: tuple) -> dict:
    q, nr_stride, top_n, maxiter = args
    case = load_case(q, nr_stride)
    return optimize_case(case, top_n, maxiter)


# ---------------------------------------------------------------------------
# Polynomial fitting — PP-anchored (c0 = 1 fixed for alpha and beta)
# ---------------------------------------------------------------------------

def fit_poly_anchored(
    q_arr: np.ndarray, vals: np.ndarray, degree: int, anchor: float = 1.0,
) -> np.ndarray:
    """Fit f(q) = anchor + c1/q + c2/q^2 + ... + c_d/q^d.
    Returns [anchor, c1, ..., c_d] — c0 is fixed, c1..c_d are fitted."""
    y = 1.0 / q_arr
    A = np.vstack([y ** k for k in range(1, degree + 1)]).T
    free_coeffs = np.linalg.lstsq(A, vals - anchor, rcond=None)[0]
    return np.concatenate([[anchor], free_coeffs])


def eval_poly(q: float, coeffs: np.ndarray) -> float:
    """Evaluate c0 + c1/q + c2/q^2 + ..."""
    y = 1.0 / q
    return float(sum(c * y ** k for k, c in enumerate(coeffs)))


def fit_master(rows: list[dict], degree: int) -> dict[str, np.ndarray]:
    q_arr = np.array([r["q"]    for r in rows], dtype=float)
    a_arr = np.array([r["alpha"] for r in rows], dtype=float)
    b_arr = np.array([r["beta"]  for r in rows], dtype=float)
    return {
        "alpha": fit_poly_anchored(q_arr, a_arr, degree),
        "beta":  fit_poly_anchored(q_arr, b_arr, degree),
    }


# ---------------------------------------------------------------------------
# Nuisance polish and master evaluation
# ---------------------------------------------------------------------------

def polish_nuisance(alpha: float, beta: float, case: dict) -> tuple[float, float, float]:
    """Fix alpha and beta from polynomial; minimise over t0_nr and phi0 only."""
    t0_seed  = _t0_seed(beta, case["meta"])
    phi_seed = _phi0_estimate(alpha, beta, t0_seed,
                               case["t_bhpt"], case["h_bhpt"],
                               case["fit_t_nr"], case["fit_h_nr"])

    def obj(x: np.ndarray) -> float:
        return evaluate_model(
            alpha, beta, x[0], x[1],
            case["t_bhpt"], case["h_bhpt"],
            case["fit_t_nr"], case["fit_h_nr"],
        )["error"]

    result = minimize(
        obj, [t0_seed, phi_seed],
        method="Nelder-Mead",
        options={"maxiter": 600, "xatol": 1e-7, "fatol": 1e-9},
    )
    return float(result.x[0]), float(result.x[1]), float(result.fun)


def evaluate_master_at_q(q: float, coeffs: dict[str, np.ndarray], nr_stride: int) -> dict:
    alpha = float(np.clip(eval_poly(q, coeffs["alpha"]), 0.05, 2.5))
    beta  = float(np.clip(eval_poly(q, coeffs["beta"]),  0.20, 1.6))
    case  = load_case(q, nr_stride)
    t0_nr, phi0, _ = polish_nuisance(alpha, beta, case)
    full = evaluate_model(
        alpha, beta, t0_nr, phi0,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"],
    )
    return {
        "q":        q,
        "alpha":    alpha,
        "beta":     beta,
        "t0_nr":    t0_nr,
        "phi0":     phi0,
        "error":    float(full["error"]),
        "coverage": float(full.get("coverage", float("nan"))),
    }


def _master_worker(args: tuple) -> dict:
    q, nr_stride, coeffs = args
    return evaluate_master_at_q(q, coeffs, nr_stride)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def summarize(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a) & (a < 50.0)]
    return {"min": float(np.min(a)), "median": float(np.median(a)),
            "mean": float(np.mean(a)), "max": float(np.max(a))}


def should_use_quartic(s3: dict, s4: dict) -> bool:
    med_gain = (s3["median"] - s4["median"]) / max(s3["median"], 1e-15)
    max_gain = (s3["max"]    - s4["max"])    / max(s3["max"],    1e-15)
    return bool(med_gain > 0.03 or max_gain > 0.03)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def _coeff_table_lines(coeffs: dict[str, np.ndarray]) -> list[str]:
    degree = len(coeffs["alpha"]) - 1
    header = "| parameter | " + " | ".join(f"c{k}" for k in range(degree + 1)) + " |"
    sep    = "|:---|" + "".join("---:|" for _ in range(degree + 1))
    lines  = [header, sep]
    for name in PHYS_NAMES:
        row = " | ".join(f"{v:.12g}" for v in coeffs[name])
        lines.append(f"| {name} | {row} |")
    return lines


def write_markdown(
    rows: list[dict],
    fit_rows: list[dict],
    master3: list[dict],
    master4: list[dict],
    coeffs3: dict,
    coeffs4: dict,
    selected_degree: int,
    args: argparse.Namespace,
) -> None:
    selected        = master4 if selected_degree == 4 else master3
    selected_coeffs = coeffs4 if selected_degree == 4 else coeffs3

    s_perq = summarize([r["error"] for r in rows])
    s_fit  = summarize([r["error"] for r in fit_rows])
    s3     = summarize([r["error"] for r in master3])
    s4     = summarize([r["error"] for r in master4])
    s_sel  = summarize([r["error"] for r in selected])
    q_max  = max(selected, key=lambda r: r["error"])["q"]

    lines = [
        "# Constant alpha-beta q-dependent scaling fit (q_dep_classic)",
        "",
        "Replicates the Islam et al. (arXiv:2204.01972) BHPTNRSurrogate model class:",
        "constant amplitude scaling (alpha) and time-stretch (beta) per q, fit as",
        "PP-anchored polynomials in 1/q against NRHybSur3dq8.",
        "",
        "## Model equations",
        "",
        "```python",
        "tau(t)       = t0_nr + beta * (t - T_ANCHOR)    # T_ANCHOR = -100 M in BHPT time",
        "h_model(tau) = alpha * exp(i*phi0) * h_BHPT(t)",
        "",
        "alpha(q) = c0 + c1/q + c2/q^2 + ... + c_d/q^d   # c0 = 1 (PP anchor)",
        "beta(q)  = c0 + c1/q + c2/q^2 + ... + c_d/q^d   # c0 = 1 (PP anchor)",
        "```",
        "",
        "t0_nr and phi0 are alignment nuisances, re-optimised at every evaluation.",
        "",
        f"Calibration grid: {len(rows)} q values from "
        f"{min(r['q'] for r in rows):.4g} to {max(r['q'] for r in rows):.4g}.",
        f"Rows used for regression: {len(fit_rows)} "
        f"(outliers with mathcalE >= {args.master_outlier_cut:g} excluded).",
        f"Fit settings: NR stride {args.nr_stride}, top starts {args.top_n}, "
        f"maxiter {args.maxiter}, workers {args.workers}.",
        "",
        f"Selected polynomial degree: **{selected_degree}**",
        "",
        "## Selected Master Coefficients",
        "",
        *_coeff_table_lines(selected_coeffs),
        "",
        "### Islam et al. (2204.01972) reference coefficients (calibrated to SXS NR)",
        "",
        "| parameter | c0 | c1 | c2 | c3 | c4 |",
        "|:---|---:|---:|---:|---:|---:|",
        "| alpha | 1.0 | -1.3301744362 | 2.7201499323 | -5.9043496190 | 5.5489244350 |",
        "| beta  | 1.0 | -1.2384720748 | 1.5967739745 | -1.7765605884 | 1.0577827906 |",
        "",
        "## Error Summary",
        "",
        "| model | min | median | mean | max | q at max |",
        "|:---|---:|---:|---:|---:|---:|",
        f"| per-q independent | {s_perq['min']:.6g} | {s_perq['median']:.6g}"
        f" | {s_perq['mean']:.6g} | {s_perq['max']:.6g} | |",
        f"| per-q used in regression | {s_fit['min']:.6g} | {s_fit['median']:.6g}"
        f" | {s_fit['mean']:.6g} | {s_fit['max']:.6g} | |",
        f"| cubic master | {s3['min']:.6g} | {s3['median']:.6g}"
        f" | {s3['mean']:.6g} | {s3['max']:.6g} | |",
        f"| quartic master | {s4['min']:.6g} | {s4['median']:.6g}"
        f" | {s4['mean']:.6g} | {s4['max']:.6g} | |",
        f"| selected master | {s_sel['min']:.6g} | {s_sel['median']:.6g}"
        f" | {s_sel['mean']:.6g} | {s_sel['max']:.6g} | {q_max:.6g} |",
        "",
        "## Per-q Results",
        "",
        "| q | per-q alpha | per-q beta | per-q mathcalE | master mathcalE | coverage |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    sel_by_q = {f"{r['q']:.10f}": r for r in selected}
    for row in rows:
        key = f"{row['q']:.10f}"
        mr  = sel_by_q.get(key, {"error": float("nan"), "coverage": float("nan")})
        lines.append(
            f"| {row['q']:.6g} | {row['alpha']:.6g} | {row['beta']:.6g}"
            f" | {row['error']:.6g} | {mr['error']:.6g} | {mr['coverage']:.6g} |"
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
    parser = argparse.ArgumentParser(description="Fit constant alpha-beta q_dep_classic model")
    parser.add_argument("--n-q",               type=int,   default=40)
    parser.add_argument("--q-min",             type=float, default=3.0)
    parser.add_argument("--q-max",             type=float, default=8.0)
    parser.add_argument("--nr-stride",         type=int,   default=5)
    parser.add_argument("--top-n",             type=int,   default=3)
    parser.add_argument("--maxiter",           type=int,   default=3000)
    parser.add_argument("--master-outlier-cut", type=float, default=MASTER_OUTLIER_CUT)
    parser.add_argument("--workers",           type=int,   default=4)
    parser.add_argument("--degree",            type=int,   default=None,
                        help="Force polynomial degree (3 or 4). Default: auto-select.")
    parser.add_argument("--force",             action="store_true",
                        help="Re-run per-q optimisation even if cached.")
    args = parser.parse_args()

    q_values = np.linspace(args.q_min, args.q_max, args.n_q)

    # ---- Phase 1: waveform cache (reuse qdep generator) --------------------
    need_cache = [q for q in q_values if not waveform_cache_path(q).exists()]
    if need_cache:
        print(f"Generating waveform cache for {len(need_cache)} q values ...", flush=True)
        with Pool(args.workers) as pool:
            pool.map(qdep.generate_and_cache_waveform, need_cache)
    else:
        print("All waveforms already cached.", flush=True)

    # ---- Phase 2: per-q optimisation ---------------------------------------
    cache = {} if args.force else load_cache()
    todo  = [q for q in q_values if q_key(q) not in cache]
    if todo:
        print(f"Optimising {len(todo)} q values with {args.workers} workers ...", flush=True)
        opt_args = [(q, args.nr_stride, args.top_n, args.maxiter) for q in todo]
        with Pool(args.workers) as pool:
            results = pool.map(optimize_case_worker, opt_args)
        for r in results:
            cache[q_key(r["q"])] = r
        save_cache(cache)
    else:
        print("All per-q results already cached.", flush=True)

    rows = [cache[q_key(float(q))] for q in sorted(q_values)]

    # ---- Phase 3: polynomial regression ------------------------------------
    fit_rows = [r for r in rows if r["error"] < args.master_outlier_cut]
    if len(fit_rows) < 8:
        raise RuntimeError("Too few valid rows for polynomial regression.")
    print(f"Fitting polynomials on {len(fit_rows)} rows ...", flush=True)

    coeffs3 = fit_master(fit_rows, degree=3)
    coeffs4 = fit_master(fit_rows, degree=4)

    # ---- Phase 4: master model evaluation (parallel) -----------------------
    print("Evaluating master model ...", flush=True)
    eval3_args = [(r["q"], args.nr_stride, coeffs3) for r in rows]
    eval4_args = [(r["q"], args.nr_stride, coeffs4) for r in rows]
    with Pool(args.workers) as pool:
        master3 = pool.map(_master_worker, eval3_args)
        master4 = pool.map(_master_worker, eval4_args)

    s3 = summarize([r["error"] for r in master3])
    s4 = summarize([r["error"] for r in master4])
    if args.degree is not None:
        selected_degree = args.degree
    else:
        selected_degree = 4 if should_use_quartic(s3, s4) else 3

    print(f"cubic:   median={s3['median']:.6g}  max={s3['max']:.6g}")
    print(f"quartic: median={s4['median']:.6g}  max={s4['max']:.6g}")
    print(f"selected degree: {selected_degree}")

    write_markdown(rows, fit_rows, master3, master4, coeffs3, coeffs4, selected_degree, args)
    print(f"Wrote {MD_PATH}", flush=True)


if __name__ == "__main__":
    main()
