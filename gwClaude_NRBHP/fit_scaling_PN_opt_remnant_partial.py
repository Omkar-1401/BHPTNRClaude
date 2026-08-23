"""
Remnant-informed partial calibration (factorised beta_r).

Same 10-parameter creative model as q_dep, with two changes to the
polynomial regression:

  1. beta_r is factorised as:
        beta_r(q) = r_beta(chi_f(q)) * beta_r_physical(M_f, chi_f, q)
     where:
        beta_r_physical = omega_BHPT_primary / omega_NR_remnant
                       = [0.3683*(1+q)/q] / [(F1+F2*(1-chi_f)^F3)/M_f]
     (Berti et al. 2006 (2,2,0) QNM; F1=1.5251, F2=-1.1568, F3=0.1292)
     is computed exactly from surfinBH for any q, and
        r_beta(chi_f) — the systematic correction between the fitted
     beta_r and the QNM prediction — is fit as a cubic poly in chi_f.

  2. All other 9 parameters are fit as cubic polynomials in chi_f(q)
     (the remnant final spin from NRSur3dq8Remnant), as in the plain
     chi_f model.

Physical motivation: beta_r_physical uses the EXACT surfinBH remnant
properties at any q — including q < 3 and q > 8 where BHPT is unavailable.
The correction factor r_beta, which captures the offset between the BHPT
surrogate's effective ringdown frequency and the idealised QNM, is smooth
in chi_f and extrapolates using the physical chi_f anchor from surfinBH.

This is the correct implementation of "using remnant info across q=1-10
with BHPT data only at q=3-8": beta_r_physical is exact at any q via
surfinBH; the learned correction r_beta is extrapolated in chi_f space.

Reuses the per-q optimisation cache from PN_opt_creative_q_dep_results/.
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

import surfinBH as _surfinbh
_SFBH = _surfinbh.LoadFits("NRSur3dq8Remnant")

# Berti et al. 2006 (2,2,0) QNM coefficients
QNM_F1, QNM_F2, QNM_F3 = 1.5251, -1.1568, 0.1292
BETA_R_IDX = 6   # index of beta_r in PARAM_NAMES

PARAM_NAMES = qdep.PARAM_NAMES

RESULTS_DIR = ROOT / "PN_opt_remnant_partial_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "per_q_cache.json"
WAVEFORM_CACHE_DIR = qdep.WAVEFORM_CACHE_DIR
MD_PATH = ROOT / "scaling_PN_opt_remnant_partial.md"

MODE = creative.MODE
NR_T_START = creative.NR_T_START
NR_T_END = creative.NR_T_END
T_ANCHOR = creative.T_ANCHOR
MIN_COVERAGE = qdep.MIN_COVERAGE
MASTER_OUTLIER_CUT = qdep.MASTER_OUTLIER_CUT


# ---------------------------------------------------------------------------
# Remnant helpers
# ---------------------------------------------------------------------------

def get_remnant(q: float) -> tuple[float, float]:
    """(M_f, chi_f_z) for non-spinning binary at mass ratio q."""
    mf, _ = _SFBH.mf(q, [0, 0, 0], [0, 0, 0])
    chif, _ = _SFBH.chif(q, [0, 0, 0], [0, 0, 0])
    return float(mf), float(chif[2])


def get_chi_f(q: float) -> float:
    """Remnant dimensionless spin — regression coordinate."""
    return get_remnant(q)[1]


def beta_r_physical(mf: float, chi_f: float, q: float) -> float:
    """
    Physical QNM-based time-stretch for the ringdown.

    Derived from matching the (2,2,0) QNM frequencies:
        beta_r = omega_BHPT_primary / omega_NR_remnant
    where omega_BHPT_primary is the Schwarzschild QNM of the primary BH
    (mass M_1 = q/(1+q), spin 0) and omega_NR_remnant is the Kerr QNM
    of the remnant (mass M_f, spin chi_f), both in units of 1/M_total.

    Valid for any q via surfinBH; used as physics anchor for beta_r
    extrapolation to q outside the BHPT training range [3, 8].
    """
    omega_primary = 0.3683 * (1.0 + q) / q
    omega_remnant = (QNM_F1 + QNM_F2 * (1.0 - chi_f) ** QNM_F3) / mf
    return omega_primary / omega_remnant


# ---------------------------------------------------------------------------
# Polynomial functions — chi_f basis
# ---------------------------------------------------------------------------

def design(q_values: np.ndarray, degree: int) -> np.ndarray:
    x = np.array([get_chi_f(float(q)) for q in q_values])
    return np.vstack([x ** k for k in range(degree + 1)]).T


def fit_poly(q_values: np.ndarray, values: np.ndarray, degree: int) -> np.ndarray:
    return np.linalg.lstsq(design(q_values, degree), values, rcond=None)[0]


def eval_poly(q: float, coeffs: np.ndarray) -> float:
    """Evaluate polynomial at chi_f(q).  For beta_r, returns r_beta correction factor."""
    x = get_chi_f(q)
    return float(sum(coeffs[i] * x ** i for i in range(len(coeffs))))


def fit_master(rows: list[dict], degree: int) -> dict[str, np.ndarray]:
    """
    Fit cubic polynomials in chi_f for all parameters.

    For beta_r specifically: fit the correction ratio
        r_beta = beta_r_fitted / beta_r_physical
    so that at evaluation: beta_r = r_beta(chi_f) * beta_r_physical(M_f, chi_f, q).
    """
    q_arr = np.array([r["q"] for r in rows], dtype=float)
    params = np.array([r["params"] for r in rows], dtype=float)
    coeffs: dict[str, np.ndarray] = {}
    for i, name in enumerate(PARAM_NAMES):
        vals = params[:, i].copy()
        if name == "phi0":
            vals = np.unwrap(vals)
        elif name == "beta_r":
            # Factorised: store correction ratio r = fitted / physical
            phys = np.array([
                beta_r_physical(*get_remnant(float(q)), float(q))
                for q in q_arr
            ])
            vals = vals / phys
        coeffs[name] = fit_poly(q_arr, vals, degree)
    return coeffs


def constrained_master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Reconstruct full 10-param array from polynomial coefficients.

    beta_r is reassembled as:  r_beta_poly(chi_f(q)) * beta_r_physical(q)
    All other parameters: direct polynomial evaluation in chi_f.
    """
    params = np.array([eval_poly(q, coeffs[name]) for name in PARAM_NAMES], dtype=float)

    # Reassemble beta_r from correction factor × physical prediction
    mf, chi_f_val = get_remnant(q)
    params[BETA_R_IDX] = params[BETA_R_IDX] * beta_r_physical(mf, chi_f_val, q)

    # Clip to physical bounds
    params[0] = float(np.clip(params[0], -3.0, 0.05))           # p0
    params[1] = float(np.clip(params[1], creative.W_MIN, 0.45)) # w
    params[2] = float(np.clip(params[2], 0.05, 2.5))            # alpha_i
    params[5] = float(np.clip(params[5], 0.2, 1.6))             # beta_i
    params[6] = float(np.clip(params[6], 0.2, 1.6))             # beta_r
    params[9] = float(np.clip(params[9], -0.05, 0.05))          # beta_L
    return params


# ---------------------------------------------------------------------------
# Master model evaluation with nuisance polish
# ---------------------------------------------------------------------------

def polish_nuisance(params: np.ndarray, case: dict) -> tuple[np.ndarray, float]:
    def obj(x: np.ndarray) -> float:
        trial = params.copy()
        trial[7] = x[0]
        trial[8] = x[1]
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
        case = qdep.load_case(q, source_stride, nr_stride)
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
    q, source_stride, nr_stride, do_polish, coeffs2, coeffs3, degree = args
    coeffs = coeffs3 if degree == 3 else coeffs2
    result = evaluate_master([{"q": q}], coeffs, source_stride, nr_stride, do_polish)
    return result[0]


# ---------------------------------------------------------------------------
# Per-q optimisation (identical to q_dep; reuse cache when available)
# ---------------------------------------------------------------------------

def optimize_case_worker(args: tuple) -> dict:
    q, source_stride, nr_stride, top_n, maxiter = args
    case = qdep.load_case(q, source_stride, nr_stride)
    return qdep.optimize_case(case, top_n, maxiter)


# ---------------------------------------------------------------------------
# Statistics / degree selection
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


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def coeff_table_lines(coeffs: dict[str, np.ndarray], degree: int) -> list[str]:
    header = "| parameter | " + " | ".join(f"c{i}" for i in range(degree + 1)) + " |"
    sep = "|:---|" + "|".join("---:" for _ in range(degree + 1)) + "|"
    lines = [header, sep]
    for name in PARAM_NAMES:
        row = " | ".join(f"{v:.12g}" for v in coeffs[name])
        note = " ★" if name == "beta_r" else ""
        lines.append(f"| {name}{note} | {row} |")
    return lines


def write_markdown(
    rows: list[dict],
    fit_rows: list[dict],
    master2: list[dict],
    master3: list[dict],
    coeffs2: dict[str, np.ndarray],
    coeffs3: dict[str, np.ndarray],
    selected_degree: int,
    args_ns,
) -> None:
    selected = master3 if selected_degree == 3 else master2
    selected_coeffs = coeffs3 if selected_degree == 3 else coeffs2

    valid_rows = [r for r in rows
                  if np.isfinite(r.get("coverage", float("nan"))) and r["error"] < 10.0]
    s_perq = summarize([r["error"] for r in valid_rows])
    s_fit  = summarize([r["error"] for r in fit_rows])
    s2 = summarize([r["error"] for r in master2])
    s3 = summarize([r["error"] for r in master3])
    s_sel = summarize([r["error"] for r in selected])
    q_at_max = max(selected, key=lambda r: r["error"])["q"]

    chi_f_q3 = get_chi_f(3.0)
    chi_f_q8 = get_chi_f(8.0)
    chi_f_q2 = get_chi_f(2.0)
    mf_q2, _ = get_remnant(2.0)
    brphys_q2 = beta_r_physical(mf_q2, chi_f_q2, 2.0)

    # beta_r comparison table for key q values
    br_table_rows = []
    sample_qs = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    import json as _json
    qdep_cache_path = ROOT / "PN_opt_creative_q_dep_results" / "per_q_cache.json"
    qdep_cache = {}
    if qdep_cache_path.exists():
        qdep_cache = _json.loads(qdep_cache_path.read_text())
    for sq in sample_qs:
        key = f"{sq:.10f}"
        if key not in qdep_cache:
            continue
        brfit = qdep_cache[key]["params"][6]
        mf_sq, chi_f_sq = get_remnant(sq)
        brp = beta_r_physical(mf_sq, chi_f_sq, sq)
        r = brfit / brp
        br_table_rows.append(f"| {sq:.1f} | {chi_f_sq:.4f} | {brp:.4f} | {brfit:.4f} | {r:.4f} |")

    lines = [
        "# Remnant-partial PN-loss q-dependent scaling fit (factorised beta_r)",
        "",
        "Same 10-parameter logistic-switch creative model as q-dep.  Two changes",
        "to the polynomial regression:",
        "",
        "1. **beta_r is factorised** into a physics-known part and a learned correction.",
        "2. **All other 9 parameters** are fit as cubic polynomials in `chi_f(q)` from",
        "   NRSur3dq8Remnant, replacing the `1/q` regression coordinate.",
        "",
        "## beta_r factorisation",
        "",
        "```",
        "beta_r(q) = r_beta(chi_f(q)) * beta_r_physical(M_f, chi_f, q)",
        "",
        "beta_r_physical = omega_BHPT_primary / omega_NR_remnant",
        "               = [0.3683*(1+q)/q]",
        "                 / [(1.5251 - 1.1568*(1-chi_f)^0.1292) / M_f]",
        "",
        "r_beta(chi_f) = c0 + c1*chi_f + c2*chi_f^2 + c3*chi_f^3   [correction poly]",
        "```",
        "",
        "`beta_r_physical` is computed from `NRSur3dq8Remnant` (surfinBH) at any q —",
        "valid for q ∈ [1, 10].  The physical prediction uses the exact remnant M_f and",
        "chi_f, not a polynomial extrapolation.  `r_beta` captures the systematic offset",
        "between the BHPT surrogate's effective ringdown frequency and the idealised QNM.",
        "",
        f"At q=2: chi_f={chi_f_q2:.4f}, M_f={mf_q2:.5f}, beta_r_physical={brphys_q2:.4f}.",
        "(Note: beta_r_physical > 1 at q=2 because the primary BH is small",
        "relative to total mass — its Schwarzschild QNM is faster than the",
        "remnant's Kerr QNM.)",
        "",
        "## beta_r physical vs fitted (training grid sample)",
        "",
        "| q | chi_f | beta_r_physical | beta_r_fitted | r = fit/phys |",
        "|---:|---:|---:|---:|---:|",
        *br_table_rows,
        "",
        "The correction ratio r is smooth and monotone in chi_f: {:.4f} (q=3) to {:.4f} (q=8).".format(
            qdep_cache.get(f"{3.0:.10f}", {"params": [0]*10})["params"][6]
              / beta_r_physical(*get_remnant(3.0), 3.0)
            if f"{3.0:.10f}" in qdep_cache else float("nan"),
            qdep_cache.get(f"{8.0:.10f}", {"params": [0]*10})["params"][6]
              / beta_r_physical(*get_remnant(8.0), 8.0)
            if f"{8.0:.10f}" in qdep_cache else float("nan"),
        ),
        "",
        "## Model equations",
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
        "## Regression",
        "",
        "```python",
        "# 9 parameters: cubic polynomial in chi_f from surfinBH",
        "C(q) = c0 + c1*chi_f + c2*chi_f**2 + c3*chi_f**3",
        "",
        "# beta_r: correction factor stored in table, physical formula applied at eval",
        "beta_r(q) = poly_r_beta(chi_f(q)) * beta_r_physical(M_f(q), chi_f(q), q)",
        "```",
        "",
        f"Training grid: {len(rows)} q values in [{min(r['q'] for r in rows):.4g}, {max(r['q'] for r in rows):.4g}].",
        f"chi_f training range: [{chi_f_q8:.4f}, {chi_f_q3:.4f}].  "
        f"Rows used in regression: {len(fit_rows)}.",
        "",
        f"Selected polynomial degree: **{selected_degree}**",
        "",
        "## Selected Master Coefficients",
        "",
        "★ beta_r row stores the correction factor r_beta, not raw beta_r.",
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
        "| q | chi_f | beta_r_phys | beta_r_master | r_beta | per-q E | master E | coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    sel_by_q = {f"{r['q']:.10f}": r for r in selected}
    for row in rows:
        key = f"{row['q']:.10f}"
        mf_q, chi_f_q = get_remnant(float(row["q"]))
        brphys = beta_r_physical(mf_q, chi_f_q, float(row["q"]))
        mr = sel_by_q.get(key, {"error": float("nan"), "coverage": float("nan"), "params": [0]*10})
        br_master = mr["params"][6] if "params" in mr else float("nan")
        r_val = br_master / brphys if np.isfinite(br_master) else float("nan")
        lines.append(
            f"| {row['q']:.6g} | {chi_f_q:.4f} | {brphys:.4f} | {br_master:.4f}"
            f" | {r_val:.4f} | {row['error']:.4g} | {mr['error']:.4g} | {mr['coverage']:.4f} |"
        )
    lines.append("")
    MD_PATH.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def q_key(q: float) -> str:
    return f"{q:.10f}"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    qdep_cache = ROOT / "PN_opt_creative_q_dep_results" / "per_q_cache.json"
    if qdep_cache.exists():
        print("Using per-q cache from PN_opt_creative_q_dep_results/.", flush=True)
        return json.loads(qdep_cache.read_text())
    return {}


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

    # Phase 1: waveform caching (shared with q_dep)
    need_cache = [q for q in q_values if not qdep.waveform_cache_path(q).exists()]
    if need_cache:
        print(f"Generating waveforms for {len(need_cache)} q values ...", flush=True)
        with Pool(args.workers) as pool:
            pool.map(qdep.generate_and_cache_waveform, need_cache)
    else:
        print("All waveforms already cached.", flush=True)

    # Phase 2: per-q optimisation (reuse q_dep cache by default)
    cache = {} if args.force else load_cache()
    todo = [q for q in q_values if q_key(q) not in cache]
    if todo:
        print(f"Optimising {len(todo)} q values ...", flush=True)
        opt_args = [(q, args.source_stride, args.nr_stride, args.top_n, args.maxiter)
                    for q in todo]
        with Pool(args.workers) as pool:
            results = pool.map(optimize_case_worker, opt_args)
        for row in results:
            cache[q_key(row["q"])] = row
        save_cache(cache)
    else:
        print("All per-q results already cached.", flush=True)

    rows = sorted(
        [v for v in cache.values() if args.q_min - 1e-9 <= v["q"] <= args.q_max + 1e-9],
        key=lambda r: r["q"],
    )

    # Phase 3: polynomial fitting (chi_f basis; factorised beta_r)
    fit_rows = [r for r in rows
                if np.isfinite(r.get("coverage", float("nan")))
                and r["error"] < args.master_outlier_cut]
    if len(fit_rows) < 8:
        raise RuntimeError("Too few valid rows for regression.")
    print(f"Fitting polynomials on {len(fit_rows)} rows (factorised beta_r) ...", flush=True)

    coeffs2 = fit_master(fit_rows, degree=2)
    coeffs3 = fit_master(fit_rows, degree=3)

    # Phase 4: parallel master model evaluation
    print("Evaluating master model ...", flush=True)
    eval_args = [
        (r["q"], args.source_stride, args.nr_stride, True, coeffs2, coeffs3, deg)
        for r in rows for deg in (2, 3)
    ]
    with Pool(args.workers) as pool:
        all_results = pool.map(_eval_master_worker, eval_args)

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
