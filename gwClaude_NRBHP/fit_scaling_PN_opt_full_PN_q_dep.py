"""
Full-PN q-dependent calibration model (no logistic switch).

Direct coupling of amplitude and time-stretch to PN-loss coordinates:

    alpha(t) = alpha_i + alpha_E * Ehat(t) + alpha_J * Jhat(t)
    beta(t)  = beta_i  + beta_E  * Ehat(t) + beta_J  * Jhat(t) + beta_L * p_loss(t)

    tau(t) = t0_nr + integral_{T_ANCHOR}^{t} beta(t') dt'
    h_model(tau(t)) = alpha(t) * exp(1j * phi0) * h_BHPT(t)

At merger (Ehat=0, Jhat=0, p_loss≈0): alpha=alpha_i, beta=beta_i.
9 parameters (vs 10 in logistic-switch model; p0, w, beta_r dropped; beta_E, beta_J added).

During inspiral Ehat≈Jhat, so opposite-sign alpha_E/alpha_J cancel and alpha stays
near alpha_i; the same cancellation in beta absorbs the inspiral plateau, while
beta_L provides the slow linear drift. After merger Ehat and Jhat diverge, giving
each coupling its own independent handle on the ringdown.

Q-regression coordinate: chi_f(q) from NRSur3dq8Remnant (bounded, monotone in [3,8]).
Training grid: 64-q [3,8] (all waveforms already cached in .cache/q_dep/).
Per-q cache: PN_opt_full_PN_q_dep_results/per_q_cache.json (independent of q_dep cache).
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

# param order: alpha_i alpha_E alpha_J  beta_i beta_E beta_J beta_L  t0_nr phi0
PARAM_NAMES = ["alpha_i", "alpha_E", "alpha_J",
               "beta_i",  "beta_E",  "beta_J",  "beta_L",
               "t0_nr",   "phi0"]

RESULTS_DIR = ROOT / "PN_opt_full_PN_q_dep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "per_q_cache.json"
WAVEFORM_CACHE_DIR = qdep.WAVEFORM_CACHE_DIR
MD_PATH = ROOT / "scaling_PN_opt_full_PN_q_dep.md"

MODE = creative.MODE
NR_T_START = creative.NR_T_START
NR_T_END = creative.NR_T_END
T_ANCHOR = creative.T_ANCHOR
MIN_COVERAGE = qdep.MIN_COVERAGE
MASTER_OUTLIER_CUT = qdep.MASTER_OUTLIER_CUT


# ---------------------------------------------------------------------------
# Remnant helpers (chi_f regression coordinate)
# ---------------------------------------------------------------------------

def get_remnant(q: float) -> tuple[float, float]:
    mf, _ = _SFBH.mf(q, [0, 0, 0], [0, 0, 0])
    chif, _ = _SFBH.chif(q, [0, 0, 0], [0, 0, 0])
    return float(mf), float(chif[2])


def get_chi_f(q: float) -> float:
    return get_remnant(q)[1]


# ---------------------------------------------------------------------------
# Full-PN model (no switch)
# ---------------------------------------------------------------------------

def unpack(params: np.ndarray) -> dict[str, float]:
    alpha_i, alpha_E, alpha_J, beta_i, beta_E, beta_J, beta_L, t0_nr, phi0 = [
        float(v) for v in params
    ]
    return dict(alpha_i=alpha_i, alpha_E=alpha_E, alpha_J=alpha_J,
                beta_i=beta_i, beta_E=beta_E, beta_J=beta_J, beta_L=beta_L,
                t0_nr=t0_nr, phi0=phi0)


def model_arrays(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    losses: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = unpack(params)
    Ehat   = losses["e_hat"]
    Jhat   = losses["j_hat"]
    p_loss = losses["p_loss"]

    alpha = p["alpha_i"] + p["alpha_E"] * Ehat + p["alpha_J"] * Jhat
    beta  = p["beta_i"]  + p["beta_E"]  * Ehat + p["beta_J"]  * Jhat + p["beta_L"] * p_loss

    beta_integral = creative.cumulative_trapezoid(beta, t_bhpt)
    anchor = float(np.interp(T_ANCHOR, t_bhpt, beta_integral))
    tau = p["t0_nr"] + beta_integral - anchor
    h_scaled = alpha * np.exp(1j * p["phi0"]) * h_bhpt
    return tau, h_scaled, alpha, beta


def validate_basic(params: np.ndarray) -> bool:
    p = unpack(params)
    if not (0.05 <= p["alpha_i"] <= 2.5):
        return False
    if not (-2.0 <= p["alpha_E"] <= 2.0):
        return False
    if not (-2.0 <= p["alpha_J"] <= 2.0):
        return False
    if not (0.2 <= p["beta_i"] <= 1.8):
        return False
    if not (-0.6 <= p["beta_E"] <= 0.6):
        return False
    if not (-0.6 <= p["beta_J"] <= 0.6):
        return False
    if not (-0.05 <= p["beta_L"] <= 0.05):
        return False
    if not (-800.0 <= p["t0_nr"] <= 30.0):
        return False
    return True


def evaluate_model(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    losses: dict,
    min_coverage: float = 0.94,
) -> dict:
    if not validate_basic(params):
        return {"error": 50.0}
    tau, h_scaled, alpha, beta = model_arrays(params, t_bhpt, h_bhpt, losses)
    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    if np.count_nonzero(use) < 2000:
        return {"error": 50.0}
    if np.any(np.diff(tau[use]) <= 0.0):
        return {"error": 50.0}
    if np.any(alpha[use] <= 0.0) or np.any(beta[use] <= 0.0):
        return {"error": 50.0}

    common = (t_nr >= tau[use][0]) & (t_nr <= tau[use][-1])
    coverage = np.count_nonzero(common) / len(t_nr)
    if coverage < min_coverage:
        return {"error": 50.0 + (min_coverage - coverage)}

    h_model = creative.interp_complex(tau[use], h_scaled[use], t_nr[common])
    return {
        "error": creative.mathcalE_error(h_nr[common], h_model),
        "coverage": coverage,
        "common": common,
        "h_ref": h_nr[common],
        "h_model": h_model,
        "tau": tau[use],
        "alpha": alpha[use],
        "beta": beta[use],
        "p_loss": losses["p_loss"][use],
        "e_hat": losses["e_hat"][use],
        "j_hat": losses["j_hat"][use],
    }


# ---------------------------------------------------------------------------
# 64-q grid: all waveforms cached in [q_min, q_max]
# ---------------------------------------------------------------------------

def get_training_grid(q_min: float = 3.0, q_max: float = 8.0) -> list[float]:
    qs = []
    for path in sorted(WAVEFORM_CACHE_DIR.glob("waveforms_q*.npz")):
        try:
            q = float(path.stem[len("waveforms_q"):])
            if q_min - 1e-9 <= q <= q_max + 1e-9:
                qs.append(q)
        except ValueError:
            pass
    return sorted(qs)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def anchored_t0(params: np.ndarray, meta: dict) -> np.ndarray:
    """Set t0_nr so tau(T_ANCHOR) ≈ t_nr_merger - beta_i*(t_bhpt_merger - T_ANCHOR)."""
    out = params.copy()
    p = unpack(out)
    out[7] = meta["t_nr_merger"] - p["beta_i"] * (meta["t_bhpt_merger"] - T_ANCHOR)
    return out


def q_scaled_seed(q: float, meta: dict) -> np.ndarray:
    scale = (q / (1.0 + q)) / (5.0 / 6.0)
    c = creative.ACCEPTED_CREATIVE_PARAMS
    # creative layout: [p0, w, alpha_i, alpha_E, alpha_J, beta_i, beta_r, t0_nr, phi0, beta_L]
    seed = np.array([
        c[2] * scale,  # alpha_i
        0.0,           # alpha_E — start with no PN coupling
        0.0,           # alpha_J
        c[5] * scale,  # beta_i
        0.0,           # beta_E
        0.0,           # beta_J
        c[9],          # beta_L (keep inspiral drift)
        0.0,           # t0_nr (fixed below)
        0.0,           # phi0
    ], dtype=float)
    return anchored_t0(seed, meta)


def phase_seed(params: np.ndarray, case: dict) -> np.ndarray:
    trial = params.copy()
    trial[8] = 0.0
    ev = evaluate_model(
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
    base = q_scaled_seed(q, meta)
    candidates: list[np.ndarray] = [base.copy()]

    # Vary PN slope pairs; opposite-sign pattern expected for inspiral cancellation
    for ae, aj in [(-0.3, 0.3), (0.0, 0.0), (0.3, -0.3),
                   (-0.2, 0.0), (0.2, 0.0), (0.0, -0.2), (0.0, 0.2)]:
        for be, bj in [(0.0, 0.0), (-0.05, 0.05), (0.05, -0.05)]:
            for bl_scale in (0.5, 1.0, 1.5):
                s = base.copy()
                s[1] = ae; s[2] = aj
                s[4] = be; s[5] = bj
                s[6] *= bl_scale
                candidates.append(anchored_t0(s, meta))

    # Scale alpha_i and beta_i variations
    for ai_scale in (0.85, 1.15):
        for bi_scale in (0.9, 1.1):
            s = base.copy()
            s[0] *= ai_scale
            s[3] *= bi_scale
            candidates.append(anchored_t0(s, meta))

    valid = [c for c in candidates if validate_basic(c)]
    return [phase_seed(c, case) for c in valid]


# ---------------------------------------------------------------------------
# Per-q optimizer
# ---------------------------------------------------------------------------

def _fit_error(params: np.ndarray, case: dict) -> float:
    return evaluate_model(
        params,
        case["fit_t_bhpt"], case["fit_h_bhpt"],
        case["fit_t_nr"], case["fit_h_nr"],
        case["fit_losses"], min_coverage=MIN_COVERAGE,
    )["error"]


def optimize_case(case: dict, top_n: int, maxiter: int) -> dict:
    q = case["q"]
    starts = candidate_starts(case)

    ranked = [(float(_fit_error(s, case)), s) for s in starts]
    ranked = [(e, s) for e, s in ranked if e < 10.0]
    ranked.sort(key=lambda x: x[0])

    if not ranked:
        fallback = np.array([0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.002, -80.0, 0.0])
        return {"q": q, "params": fallback.tolist(),
                "fit_error": 50.0, "error": 50.0, "coverage": float("nan")}

    best_params = ranked[0][1].copy()
    best_fit_err = ranked[0][0]

    for _, (_, start) in enumerate(ranked[:top_n]):
        result = minimize(
            lambda p: _fit_error(p, case),
            start,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-12},
        )
        if result.fun < best_fit_err:
            best_fit_err = float(result.fun)
            best_params = result.x.copy()

    full = evaluate_model(
        best_params,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"],
        case["losses"], min_coverage=MIN_COVERAGE,
    )
    print(f"q={q:.6g}  fit={best_fit_err:.6g}  full={full['error']:.6g}", flush=True)
    return {
        "q": q,
        "params": [float(x) for x in best_params],
        "fit_error": float(best_fit_err),
        "error": float(full["error"]),
        "coverage": float(full.get("coverage", float("nan"))),
    }


def optimize_case_worker(args: tuple) -> dict:
    q, source_stride, nr_stride, top_n, maxiter = args
    case = qdep.load_case(q, source_stride, nr_stride)
    return optimize_case(case, top_n, maxiter)


# ---------------------------------------------------------------------------
# Polynomial regression in chi_f
# ---------------------------------------------------------------------------

def _chi_f_design(q_values: np.ndarray, degree: int) -> np.ndarray:
    x = np.array([get_chi_f(float(q)) for q in q_values])
    return np.vstack([x ** k for k in range(degree + 1)]).T


def fit_poly(q_values: np.ndarray, values: np.ndarray, degree: int) -> np.ndarray:
    return np.linalg.lstsq(_chi_f_design(q_values, degree), values, rcond=None)[0]


def eval_poly(q: float, coeffs: np.ndarray) -> float:
    x = get_chi_f(q)
    return float(sum(coeffs[i] * x ** i for i in range(len(coeffs))))


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
    params[0] = float(np.clip(params[0], 0.05, 2.5))    # alpha_i
    params[1] = float(np.clip(params[1], -2.0, 2.0))    # alpha_E
    params[2] = float(np.clip(params[2], -2.0, 2.0))    # alpha_J
    params[3] = float(np.clip(params[3], 0.2,  1.8))    # beta_i
    params[4] = float(np.clip(params[4], -0.6, 0.6))    # beta_E
    params[5] = float(np.clip(params[5], -0.6, 0.6))    # beta_J
    params[6] = float(np.clip(params[6], -0.05, 0.05))  # beta_L
    return params


# ---------------------------------------------------------------------------
# Master model evaluation + nuisance polish
# ---------------------------------------------------------------------------

def polish_nuisance(params: np.ndarray, case: dict) -> tuple[np.ndarray, float]:
    def obj(x: np.ndarray) -> float:
        trial = params.copy()
        trial[7] = x[0]  # t0_nr
        trial[8] = x[1]  # phi0
        return evaluate_model(
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
        ev = evaluate_model(
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
# Statistics
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

    lines = [
        "# Full-PN q-dependent scaling fit (no logistic switch)",
        "",
        "Direct coupling of amplitude and time-stretch to PN-loss coordinates,",
        "valid everywhere without a logistic transition:",
        "",
        "```python",
        "alpha(t) = alpha_i + alpha_E * Ehat(t) + alpha_J * Jhat(t)",
        "beta(t)  = beta_i  + beta_E  * Ehat(t) + beta_J  * Jhat(t) + beta_L * p_loss(t)",
        "",
        "tau(t) = t0_nr + integral_{T_ANCHOR}^{t} beta(t') dt'",
        "h_model(tau(t)) = alpha(t) * exp(1j * phi0) * h_BHPT(t)",
        "```",
        "",
        "At merger (Ehat=0, Jhat=0, p_loss≈0): alpha=alpha_i, beta=beta_i.",
        "",
        "During inspiral Ehat≈Jhat, so alpha_E/alpha_J with opposite signs",
        "approximately cancel and alpha stays near alpha_i.  After merger",
        "Ehat and Jhat diverge, giving independent ringdown handles.",
        "beta_L provides the slow linear drift in p_loss that was critical",
        "in the logistic-switch model.",
        "",
        "9 parameters (logistic model had 10; p0, w, beta_r dropped; beta_E, beta_J added).",
        "",
        "## Regression",
        "",
        "```python",
        "C(q) = c0 + c1*chi_f + c2*chi_f**2 [+ c3*chi_f**3]",
        "# chi_f(q) from NRSur3dq8Remnant (surfinBH)",
        "```",
        "",
        f"Training grid: {len(rows)} q values in"
        f" [{min(r['q'] for r in rows):.4g}, {max(r['q'] for r in rows):.4g}]"
        f" (all waveforms cached in that range).",
        f"chi_f training range: [{chi_f_q8:.4f}, {chi_f_q3:.4f}].",
        f"chi_f at extrapolation target q=2: {chi_f_q2:.4f}.",
        f"Rows used in regression: {len(fit_rows)}.",
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
        "| q | chi_f | per-q E | master E | coverage |"
        " alpha_i | alpha_E | alpha_J | beta_i | beta_E | beta_J | beta_L |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    sel_by_q = {f"{r['q']:.10f}": r for r in selected}
    for row in rows:
        key = f"{row['q']:.10f}"
        chi_f_q = get_chi_f(float(row["q"]))
        mr = sel_by_q.get(key, {"error": float("nan"), "coverage": float("nan"), "params": [0.0] * 9})
        p = mr.get("params", [0.0] * 9)
        lines.append(
            f"| {row['q']:.6g} | {chi_f_q:.4f} | {row['error']:.4g} | {mr['error']:.4g}"
            f" | {mr['coverage']:.4f}"
            f" | {p[0]:.4g} | {p[1]:.4g} | {p[2]:.4g}"
            f" | {p[3]:.4g} | {p[4]:.4g} | {p[5]:.4g} | {p[6]:.6g} |"
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
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full-PN q-dep calibration fit")
    parser.add_argument("--q-min",  type=float, default=3.0)
    parser.add_argument("--q-max",  type=float, default=8.0)
    parser.add_argument("--source-stride", type=int, default=3)
    parser.add_argument("--nr-stride",     type=int, default=5)
    parser.add_argument("--top-n",   type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=6000)
    parser.add_argument("--master-outlier-cut", type=float, default=MASTER_OUTLIER_CUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force",   action="store_true",
                        help="Ignore existing per-q cache and refit all q values")
    args = parser.parse_args()

    # 64-q grid: every waveform cached in [q_min, q_max]
    q_values = get_training_grid(args.q_min, args.q_max)
    if not q_values:
        raise RuntimeError(
            "No cached waveforms found. Run fit_scaling_PN_opt_creative_q_dep.py first."
        )
    print(
        f"Training grid: {len(q_values)} q values in"
        f" [{q_values[0]:.4g}, {q_values[-1]:.4g}]",
        flush=True,
    )

    # Phase 1: per-q optimisation with the full-PN model (new per-q cache)
    cache = {} if args.force else load_cache()
    todo = [q for q in q_values if q_key(q) not in cache]
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
        [v for v in cache.values()
         if args.q_min - 1e-9 <= v["q"] <= args.q_max + 1e-9],
        key=lambda r: r["q"],
    )

    # Phase 2: polynomial fitting in chi_f
    fit_rows = [
        r for r in rows
        if np.isfinite(r.get("coverage", float("nan"))) and r["error"] < args.master_outlier_cut
    ]
    if len(fit_rows) < 8:
        raise RuntimeError(f"Too few valid rows ({len(fit_rows)}) for regression.")
    print(f"Fitting polynomials on {len(fit_rows)} rows (chi_f basis) ...", flush=True)

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
