"""
Waveform-flux nu q-dependent calibration (wf_nu_q_dep).

Same 10-parameter logistic-switch architecture as the creative/remnant_partial
models.  Two changes relative to those models:

1. Loss coordinates (Ehat, Jhat, p_loss) are computed directly from the BHPT
   waveform strain using exact GW flux formulas:

       dE/dt  = |dh_{22}/dt|² / 16π            (energy flux)
       dJz/dt = -2 Im[h_{22}* dh_{22}/dt] / 16π  (z-angular-momentum flux)

   These are valid for any waveform, including through merger and ringdown —
   unlike the 2PN flux expressions (dEdt/dJdt) used in PN_opt models, which
   diverge near merger.  Only the (2,2) mode is available from
   BHPTNRSur1dq1e4, but the normalization at T_REF cancels the leading-mode
   approximation error: Ehat and Jhat are identical (up to sub-percent
   corrections) whether computed from the (2,2) mode alone or from all modes.

2. Polynomial regression coordinate: nu = q/(1+q)² (symmetric mass ratio),
   replacing chi_f from surfinBH (which has degeneracy under spin permutation).
   The polynomial is PP-anchored at nu→0 (test-particle limit) where BHPT is
   exact and no calibration is needed:

       alpha_i → 1,  alpha_E → 0,  alpha_J → 0
       beta_i  → 1,  beta_r  → 1,  beta_L  → 0

   p0, w, t0_nr, phi0 are unanchored (their test-particle limit depends on
   arbitrary phase/time conventions, not on physical calibration).

   nu generalises cleanly to spinning systems as (nu, chi_eff, chi_a) without
   the chi_f degeneracy.

Training grid: all q values cached in .cache/q_dep/ (64 values, q ∈ [3, 8]).
New per-q cache at wf_nu_q_dep_results/per_q_cache.json — CANNOT reuse qdep or
remnant_partial caches since those were fitted with PN loss coordinates.
Waveform cache: shared with qdep (.cache/q_dep/).
"""
from __future__ import annotations

import argparse
import json
import re
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

PARAM_NAMES = qdep.PARAM_NAMES  # ["p0","w","alpha_i","alpha_E","alpha_J",
                                 #  "beta_i","beta_r","t0_nr","phi0","beta_L"]

RESULTS_DIR = ROOT / "wf_nu_q_dep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "per_q_cache.json"
WAVEFORM_CACHE_DIR = qdep.WAVEFORM_CACHE_DIR
MD_PATH = ROOT / "scaling_wf_nu_q_dep.md"

MODE = creative.MODE
NR_T_START = creative.NR_T_START
NR_T_END = creative.NR_T_END
T_ANCHOR = creative.T_ANCHOR
T_REF = creative.T_REF
MIN_COVERAGE = qdep.MIN_COVERAGE
MASTER_OUTLIER_CUT = qdep.MASTER_OUTLIER_CUT

# PP anchors at nu→0 (test-particle limit): no calibration needed → identity map
PP_ANCHORS: dict[str, float | None] = {
    "p0":      None,
    "w":       None,
    "alpha_i": 1.0,
    "alpha_E": 0.0,
    "alpha_J": 0.0,
    "beta_i":  1.0,
    "beta_r":  1.0,
    "t0_nr":   None,
    "phi0":    None,
    "beta_L":  0.0,
}


# ---------------------------------------------------------------------------
# Waveform-derived loss coordinates
# ---------------------------------------------------------------------------

def wf_loss_coordinates(
    t_bhpt: np.ndarray, h_bhpt: np.ndarray, meta: dict
) -> dict:
    """
    Compute loss coordinates directly from BHPT waveform strain.

    For the (2,2) mode:
        dE/dt  = |ḣ_{22}|² / 16π
        dJz/dt = -2 Im[h_{22}* ḣ_{22}] / 16π   (radiated z-ang-mom, positive)

    Derivation: During inspiral h_{22} = A e^{-iΦ} so ḣ_{22} = (Ȧ - iΦ̇A)e^{-iΦ},
    giving Im[h_{22}* ḣ_{22}] = -Φ̇ A² < 0 (orbital frequency > 0).  Hence
    dJz/dt = -2*(-Φ̇A²)/16π > 0 (angular momentum is radiated away).

    Single-mode normalization is exact for Ehat, Jhat, p_loss: the factor-of-2
    from including h_{2,-2} = h_{22}* cancels in numerator and denominator.
    """
    h_dot = np.gradient(h_bhpt, t_bhpt)

    flux_e = np.abs(h_dot) ** 2 / (16.0 * np.pi)
    flux_j = -2.0 * np.imag(np.conj(h_bhpt) * h_dot) / (16.0 * np.pi)

    e_cum = creative.cumulative_trapezoid(flux_e, t_bhpt)
    j_cum = creative.cumulative_trapezoid(flux_j, t_bhpt)

    merger_index = int(np.argmax(np.abs(h_bhpt)))
    delta_e = e_cum - e_cum[merger_index]
    delta_j = j_cum - j_cum[merger_index]

    e_ref = abs(float(np.interp(T_REF, t_bhpt, delta_e)))
    j_ref = abs(float(np.interp(T_REF, t_bhpt, delta_j)))

    if e_ref < 1e-15 or j_ref < 1e-15:
        raise ValueError("Near-zero waveform flux reference at T_REF.")

    e_hat = delta_e / e_ref
    j_hat = delta_j / j_ref
    p_loss = 0.5 * (e_hat + j_hat)

    return {
        "flux_e": flux_e,
        "flux_j": flux_j,
        "delta_e": delta_e,
        "delta_j": delta_j,
        "e_hat": e_hat,
        "j_hat": j_hat,
        "p_loss": p_loss,
        "e_ref": e_ref,
        "j_ref": j_ref,
    }


# ---------------------------------------------------------------------------
# Case loading (local version using wf_loss_coordinates)
# ---------------------------------------------------------------------------

def load_case(q: float, source_stride: int = 3, nr_stride: int = 5) -> dict:
    """Load waveform cache and compute waveform-derived loss coordinates."""
    d = np.load(qdep.waveform_cache_path(q))
    t_bhpt = d["t_bhpt"]
    h_bhpt = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    t_nr = d["t_nr"]
    h_nr = d["h_nr_re"] + 1j * d["h_nr_im"]
    meta = {
        "nu": float(d["nu"]),
        "t_bhpt_merger": float(d["t_bhpt_merger"]),
        "t_nr_merger": float(d["t_nr_merger"]),
    }

    losses_full = wf_loss_coordinates(t_bhpt, h_bhpt, meta)

    src_idx = np.unique(np.r_[np.arange(0, len(t_bhpt), source_stride), len(t_bhpt) - 1])
    nr_idx = np.unique(np.r_[np.arange(0, len(t_nr), nr_stride), len(t_nr) - 1])

    return {
        "q": q,
        "t_bhpt": t_bhpt,
        "h_bhpt": h_bhpt,
        "t_nr": t_nr,
        "h_nr": h_nr,
        "meta": meta,
        "losses": losses_full,
        "fit_t_bhpt": t_bhpt[src_idx],
        "fit_h_bhpt": h_bhpt[src_idx],
        "fit_losses": qdep.subsample_losses(losses_full, src_idx),
        "fit_t_nr": t_nr[nr_idx],
        "fit_h_nr": h_nr[nr_idx],
    }


# ---------------------------------------------------------------------------
# Regression coordinate: nu = q/(1+q)²
# ---------------------------------------------------------------------------

def get_nu(q: float) -> float:
    return q / (1.0 + q) ** 2


def fit_poly_anchored(
    nu_arr: np.ndarray, values: np.ndarray, degree: int, anchor: float | None
) -> np.ndarray:
    """
    Fit polynomial in nu of given degree.

    If anchor is given: fix c0=anchor, fit only c1..c_degree
    (PP-constrained regression).  Otherwise fit all d+1 coefficients.
    """
    if anchor is not None:
        shifted = values - anchor
        X = np.vstack([nu_arr ** k for k in range(1, degree + 1)]).T
        slope = np.linalg.lstsq(X, shifted, rcond=None)[0]
        return np.concatenate([[anchor], slope])
    X = np.vstack([nu_arr ** k for k in range(degree + 1)]).T
    return np.linalg.lstsq(X, values, rcond=None)[0]


def eval_poly(q: float, coeffs: np.ndarray) -> float:
    """Evaluate c0 + c1*nu + c2*nu² + ... at nu(q)."""
    nu = get_nu(q)
    return float(sum(coeffs[i] * nu ** i for i in range(len(coeffs))))


def fit_master(rows: list[dict], degree: int) -> dict[str, np.ndarray]:
    """Fit PP-anchored polynomials in nu for all 10 parameters."""
    q_arr = np.array([r["q"] for r in rows], dtype=float)
    nu_arr = np.array([get_nu(float(q)) for q in q_arr])
    params_arr = np.array([r["params"] for r in rows], dtype=float)

    coeffs: dict[str, np.ndarray] = {}
    for i, name in enumerate(PARAM_NAMES):
        vals = params_arr[:, i].copy()
        if name == "phi0":
            vals = np.unwrap(vals)
        coeffs[name] = fit_poly_anchored(nu_arr, vals, degree, PP_ANCHORS[name])
    return coeffs


def constrained_master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    params = np.array([eval_poly(q, coeffs[name]) for name in PARAM_NAMES], dtype=float)
    params[0] = float(np.clip(params[0], -3.0, 0.05))            # p0
    params[1] = float(np.clip(params[1], creative.W_MIN, 0.45))  # w
    params[2] = float(np.clip(params[2], 0.05, 2.5))             # alpha_i
    params[5] = float(np.clip(params[5], 0.2, 1.6))              # beta_i
    params[6] = float(np.clip(params[6], 0.2, 1.6))              # beta_r
    params[9] = float(np.clip(params[9], -0.05, 0.05))           # beta_L
    return params


# ---------------------------------------------------------------------------
# Nuisance polish and master evaluation
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
    q, source_stride, nr_stride, do_polish, coeffs2, coeffs3, degree = args
    coeffs = coeffs3 if degree == 3 else coeffs2
    return evaluate_master([{"q": q}], coeffs, source_stride, nr_stride, do_polish)[0]


# ---------------------------------------------------------------------------
# Per-q optimisation worker
# ---------------------------------------------------------------------------

def optimize_case_worker(args: tuple) -> dict:
    """Load case with wf_loss_coordinates, then run the shared optimizer."""
    q, source_stride, nr_stride, top_n, maxiter = args
    case = load_case(q, source_stride, nr_stride)
    return qdep.optimize_case(case, top_n, maxiter)


# ---------------------------------------------------------------------------
# Statistics and degree selection
# ---------------------------------------------------------------------------

def summarize(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    return {
        "min": float(np.min(a)), "median": float(np.median(a)),
        "mean": float(np.mean(a)), "max": float(np.max(a)),
    }


def should_use_cubic(s2: dict, s3: dict) -> bool:
    med_gain = (s2["median"] - s3["median"]) / max(s2["median"], 1e-15)
    max_gain = (s2["max"] - s3["max"]) / max(s2["max"], 1e-15)
    return bool(med_gain > 0.03 or max_gain > 0.03)


# ---------------------------------------------------------------------------
# Cached q discovery
# ---------------------------------------------------------------------------

def get_cached_q_values() -> list[float]:
    """Return all q values for which waveform cache files exist."""
    pat = re.compile(r"waveforms_q(.+)\.npz")
    qs = []
    for p in WAVEFORM_CACHE_DIR.glob("waveforms_q*.npz"):
        m = pat.match(p.name)
        if m:
            qs.append(float(m.group(1)))
    return sorted(qs)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def coeff_table_lines(coeffs: dict[str, np.ndarray], degree: int) -> list[str]:
    header = "| parameter | anchor | " + " | ".join(f"c{i}" for i in range(degree + 1)) + " |"
    sep = "|:---|:---|" + "|".join("---:" for _ in range(degree + 1)) + "|"
    lines = [header, sep]
    for name in PARAM_NAMES:
        anc = PP_ANCHORS[name]
        anc_str = f"{anc:.4g}" if anc is not None else "free"
        row = " | ".join(f"{v:.12g}" for v in coeffs[name])
        lines.append(f"| {name} | {anc_str} | {row} |")
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

    nu_q3 = get_nu(3.0)
    nu_q8 = get_nu(8.0)
    nu_q2 = get_nu(2.0)

    anchored_params = [n for n, a in PP_ANCHORS.items() if a is not None]
    free_params = [n for n, a in PP_ANCHORS.items() if a is None]

    lines = [
        "# Waveform-flux nu q-dependent scaling fit (wf_nu_q_dep)",
        "",
        "Same 10-parameter logistic-switch creative model as q_dep / remnant_partial.",
        "Two changes:",
        "",
        "1. **Loss coordinates** (Ehat, Jhat, p_loss) computed from BHPT waveform",
        "   strain directly:",
        "   ```",
        "   dE/dt  = |ḣ_{22}|² / 16π",
        "   dJz/dt = -2 Im[h_{22}* ḣ_{22}] / 16π",
        "   ```",
        "   Valid through merger and ringdown — unlike 2PN flux expressions.",
        "",
        "2. **Regression coordinate**: nu = q/(1+q)² (symmetric mass ratio).",
        "   PP-anchored at nu=0 (test-particle limit):",
        "   ```",
        "   alpha_i, beta_i, beta_r → 1 (identity map, no calibration needed)",
        "   alpha_E, alpha_J, beta_L → 0 (no PN-flux correction)",
        "   ```",
        f"   PP anchor at nu=0 is {1/get_nu(3.0):.0f}× extrapolated from the q=3 boundary",
        "   (nu=0.1875) and constrains the polynomial globally.",
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
        "nu(q) = q / (1 + q)**2",
        "",
        "# PP-anchored params: c0 = anchor, fit only c1..c_d",
        f"# Anchored:   {', '.join(anchored_params)}",
        f"# Unanchored: {', '.join(free_params)}",
        "C(q) = anchor + c1*nu + c2*nu**2 [+ c3*nu**3]",
        "",
        "# Unanchored params:",
        "C(q) = c0 + c1*nu + c2*nu**2 [+ c3*nu**3]",
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
        "| q | nu | per-q E | master E | coverage | p0 | w | beta_i | alpha_i | beta_L |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    sel_by_q = {f"{r['q']:.10f}": r for r in selected}
    fit_keys = {f"{r['q']:.10f}" for r in fit_rows}
    for row in rows:
        key = f"{row['q']:.10f}"
        p = row["params"]
        mr = sel_by_q.get(key, {"error": float("nan"), "coverage": float("nan")})
        nu_q = get_nu(float(row["q"]))
        reg = "✓" if key in fit_keys else ""
        lines.append(
            f"| {row['q']:.6g} | {nu_q:.4f} | {row['error']:.4g}"
            f" | {mr['error']:.4g} | {mr['coverage']:.4f} | {reg}"
            f" | {p[0]:.4g} | {p[1]:.4g} | {p[5]:.4g} | {p[2]:.4g} | {p[9]:.6g} |"
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
    parser.add_argument("--q-min", type=float, default=3.0)
    parser.add_argument("--q-max", type=float, default=8.0)
    parser.add_argument("--source-stride", type=int, default=3)
    parser.add_argument("--nr-stride", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=6000)
    parser.add_argument("--master-outlier-cut", type=float, default=MASTER_OUTLIER_CUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Use all q values with cached waveforms within the requested range
    all_q = [q for q in get_cached_q_values()
             if args.q_min - 1e-9 <= q <= args.q_max + 1e-9]
    if not all_q:
        raise RuntimeError(
            "No waveforms found in .cache/q_dep/.  "
            "Run fit_scaling_PN_opt_creative_q_dep.py first to populate the cache."
        )
    print(f"Training grid: {len(all_q)} q values in [{all_q[0]:.4g}, {all_q[-1]:.4g}]",
          flush=True)

    # Phase 1: per-q optimisation with wf_loss_coordinates
    cache = {} if args.force else load_cache()
    todo = [q for q in all_q if q_key(q) not in cache]
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
        if np.isfinite(r.get("coverage", float("nan")))
        and r["error"] < args.master_outlier_cut
    ]
    if len(fit_rows) < 8:
        raise RuntimeError(
            f"Too few valid rows ({len(fit_rows)}) for polynomial regression."
        )
    print(f"Fitting polynomials on {len(fit_rows)} rows (nu basis, PP-anchored) ...",
          flush=True)

    coeffs2 = fit_master(fit_rows, degree=2)
    coeffs3 = fit_master(fit_rows, degree=3)

    # Phase 3: parallel master model evaluation
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
