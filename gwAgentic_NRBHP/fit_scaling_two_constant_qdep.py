from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.optimize import minimize

import gwsurrogate
import fit_scaling_physical_smooth_qdep as baseq


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "two_constant_qdep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "per_q_cache.json"
MD_PATH = ROOT / "scaling_two_constant_qdep.md"
SMOOTH_CACHE_PATH = ROOT / "physical_smooth_qdep_results" / "per_q_cache.json"

MODE = baseq.MODE
NR_T_START = baseq.NR_T_START
NR_T_END = baseq.NR_T_END
MAX_TRANSITION_WIDTH = 60.0

PARAM_NAMES = [
    "theta_cut",
    "theta_transition_width",
    "Theta_cut_nr",
    "beta_left",
    "beta_right",
    "alpha_left",
    "alpha_right",
    "phi0",
]


def q_key(q: float) -> str:
    return f"{q:.10f}"


def load_cache(path: Path = CACHE_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def unpack(params: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, params)}


def smooth_two_constants(
    theta: np.ndarray,
    theta_cut: float,
    width: float,
    left_value: float,
    right_value: float,
) -> np.ndarray:
    dtheta = theta - theta_cut
    out = np.empty_like(theta, dtype=float)

    left = dtheta <= 0.0
    right = dtheta >= width
    middle = ~(left | right)

    out[left] = left_value
    out[right] = right_value
    if np.any(middle):
        z = dtheta[middle] / width
        smoothstep = 3.0 * z**2 - 2.0 * z**3
        out[middle] = left_value + (right_value - left_value) * smoothstep
    return out


def model_arrays(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    meta: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = unpack(params)
    theta = meta["nu"] * (t_bhpt - meta["t_bhpt_merger"])

    beta = smooth_two_constants(
        theta,
        p["theta_cut"],
        p["theta_transition_width"],
        p["beta_left"],
        p["beta_right"],
    )
    alpha = smooth_two_constants(
        theta,
        p["theta_cut"],
        p["theta_transition_width"],
        p["alpha_left"],
        p["alpha_right"],
    )

    s_cut = meta["t_bhpt_merger"] + p["theta_cut"] / meta["nu"]
    t_cut_nr = meta["t_nr_merger"] + p["Theta_cut_nr"] / meta["nu"]
    beta_integral = baseq.cumulative_trapezoid_integral(t_bhpt, beta)
    beta_integral_cut = float(np.interp(s_cut, t_bhpt, beta_integral))
    tau = t_cut_nr + beta_integral - beta_integral_cut
    h_scaled = alpha * np.exp(1j * p["phi0"]) * h_bhpt
    return tau, h_scaled, alpha, beta, theta


def validate_basic(params: np.ndarray, meta: dict) -> bool:
    if len(params) != len(PARAM_NAMES) or not np.all(np.isfinite(params)):
        return False
    p = unpack(params)
    theta_end = p["theta_cut"] + p["theta_transition_width"]
    s_cut = meta["t_bhpt_merger"] + p["theta_cut"] / meta["nu"]
    s_end = meta["t_bhpt_merger"] + theta_end / meta["nu"]
    t_cut_nr = meta["t_nr_merger"] + p["Theta_cut_nr"] / meta["nu"]

    if not (-180.0 <= p["theta_cut"] <= -5.0):
        return False
    if not (2.0 <= p["theta_transition_width"] <= MAX_TRANSITION_WIDTH):
        return False
    if not (-150.0 <= theta_end <= 60.0):
        return False
    if not (-1400.0 <= s_cut <= -40.0):
        return False
    if not (-1200.0 <= s_end <= 500.0):
        return False
    if not (-400.0 <= p["Theta_cut_nr"] <= -1.0):
        return False
    if not (-2500.0 <= t_cut_nr <= -20.0):
        return False
    if not (0.12 <= p["beta_left"] <= 1.8):
        return False
    if not (0.12 <= p["beta_right"] <= 1.8):
        return False
    if not (0.03 <= p["alpha_left"] <= 2.8):
        return False
    if not (0.03 <= p["alpha_right"] <= 2.8):
        return False
    return True


def evaluate_model(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    meta: dict,
    min_coverage: float = 0.90,
) -> dict:
    if not validate_basic(params, meta):
        return {"error": 50.0}

    tau, h_scaled, alpha, beta, theta = model_arrays(params, t_bhpt, h_bhpt, meta)
    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    if np.count_nonzero(use) < 800:
        return {"error": 50.0}

    tau_use = tau[use]
    if np.any(np.diff(tau_use) <= 0.0):
        return {"error": 50.0}
    if np.any(alpha[use] <= 0.0) or np.any(beta[use] <= 0.0):
        return {"error": 50.0}

    common = (t_nr >= tau_use[0]) & (t_nr <= tau_use[-1])
    coverage = np.count_nonzero(common) / len(t_nr)
    if coverage < min_coverage:
        return {"error": 50.0 + (min_coverage - coverage)}

    h_model = baseq.interp_complex(tau_use, h_scaled[use], t_nr[common])
    err = baseq.mathcalE_error(h_nr[common], h_model)
    return {
        "error": err,
        "coverage": coverage,
        "common": common,
        "h_ref": h_nr[common],
        "h_model": h_model,
        "tau": tau_use,
        "alpha": alpha[use],
        "beta": beta[use],
        "theta": theta[use],
    }


def evaluate_case(params: np.ndarray, case: dict, subset: str = "fit", min_coverage: float = 0.90) -> dict:
    arrays = case[subset] if subset != "full" else case
    return evaluate_model(
        params,
        arrays["t_bhpt"],
        arrays["h_bhpt"],
        arrays["t_nr"],
        arrays["h_nr"],
        case["meta"],
        min_coverage=min_coverage,
    )


def phase_seed(params: np.ndarray, case: dict) -> np.ndarray:
    seeded = params.copy()
    seeded[-1] = 0.0
    ev = evaluate_case(seeded, case, subset="fit", min_coverage=0.84)
    if ev["error"] >= 10.0:
        return params.copy()
    seeded[-1] = baseq.principal_phase(
        float(np.angle(np.sum(ev["h_ref"] * ev["h_model"].conjugate())))
    )
    return seeded


def mass_norm_seed(q: float, theta_cut: float, width: float) -> np.ndarray:
    scale = q / (1.0 + q)
    return np.array(
        [
            theta_cut,
            width,
            scale * theta_cut,
            scale,
            scale,
            scale,
            scale,
            0.0,
        ],
        dtype=float,
    )


def convert_smooth_params(params: np.ndarray, theta_probe: float | None = None) -> np.ndarray:
    p = baseq.unpack(params)
    theta_end = p["theta_cut"] + p["theta_transition_width"]
    if theta_probe is None:
        theta_probe = max(theta_end, min(8.0, theta_end + 0.5 * max(10.0, -theta_end)))
    beta_right = p["beta_right_edge"] + p["beta_right_slope"] * (theta_probe - theta_end)
    alpha_right = p["alpha_right_edge"] + p["alpha_right_slope"] * (theta_probe - theta_end)
    return np.array(
        [
            p["theta_cut"],
            p["theta_transition_width"],
            p["Theta_cut_nr"],
            p["beta_left"],
            beta_right,
            p["alpha_left"],
            alpha_right,
            p["phi0"],
        ],
        dtype=float,
    )


def nearest_smooth_params(q: float, smooth_cache: dict) -> np.ndarray | None:
    usable = {
        key: value
        for key, value in smooth_cache.items()
        if value.get("error", 50.0) < 10.0 and "params" in value
    }
    if not usable:
        return None
    best_key = min(usable, key=lambda key: abs(float(key) - q))
    return np.asarray(usable[best_key]["params"], dtype=float)


def candidate_starts(
    case: dict,
    nearest_params: np.ndarray | None,
    smooth_cache: dict,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    if nearest_params is not None:
        starts.append(nearest_params.copy())
        for scale in (0.98, 1.02):
            shifted = nearest_params.copy()
            shifted[3:7] *= scale
            starts.append(shifted)
        for theta_shift in (-8.0, 8.0):
            shifted = nearest_params.copy()
            shifted[0] += theta_shift
            shifted[2] = shifted[3] * shifted[0]
            starts.append(shifted)

    smooth_params = nearest_smooth_params(case["q"], smooth_cache)
    if smooth_params is not None:
        p = baseq.unpack(smooth_params)
        theta_end = p["theta_cut"] + p["theta_transition_width"]
        for theta_probe in (theta_end, 0.0, 10.0, 25.0):
            starts.append(convert_smooth_params(smooth_params, theta_probe))

    for theta_cut in (-75.0, -60.0, -45.0, -35.0, -25.0, -16.0, -8.0):
        for width in (8.0, 16.0, 28.0, 45.0, 58.0):
            starts.append(mass_norm_seed(case["q"], theta_cut, width))

    seeded = []
    seen = set()
    for start in starts:
        if not validate_basic(start, case["meta"]):
            continue
        rounded = tuple(np.round(start, 10))
        if rounded in seen:
            continue
        seen.add(rounded)
        seeded.append(phase_seed(start, case))
    return seeded


def optimize_case(
    case: dict,
    nearest_params: np.ndarray | None,
    smooth_cache: dict,
    top_n: int,
    maxiter: int,
    refine_threshold: float,
    refine_maxiter: int,
) -> dict:
    starts = candidate_starts(case, nearest_params, smooth_cache)
    ranked: list[tuple[float, np.ndarray]] = []
    for start in starts:
        ev = evaluate_case(start, case, subset="fit", min_coverage=0.84)
        if ev["error"] < 10.0:
            ranked.append((float(ev["error"]), start))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        raise RuntimeError(f"No usable starts for q={case['q']:.8g}")

    best_params = ranked[0][1].copy()
    best_fit_error = ranked[0][0]
    for _start_error, start in ranked[:top_n]:
        result = minimize(
            lambda trial: evaluate_case(trial, case, subset="fit", min_coverage=0.84)[
                "error"
            ],
            start,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1.0e-7, "fatol": 1.0e-9},
        )
        if float(result.fun) < best_fit_error:
            best_fit_error = float(result.fun)
            best_params = result.x.copy()

    full = evaluate_case(best_params, case, subset="full", min_coverage=0.84)
    refined_error = np.nan
    if full["error"] > refine_threshold and refine_maxiter > 0:
        result = minimize(
            lambda trial: evaluate_case(trial, case, subset="full", min_coverage=0.84)[
                "error"
            ],
            best_params,
            method="Nelder-Mead",
            options={"maxiter": refine_maxiter, "xatol": 1.0e-7, "fatol": 1.0e-9},
        )
        refined_error = float(result.fun)
        if refined_error < full["error"]:
            best_params = result.x.copy()
            full = evaluate_case(best_params, case, subset="full", min_coverage=0.84)

    return {
        "q": float(case["q"]),
        "params": [float(x) for x in best_params],
        "fit_error": float(best_fit_error),
        "refined_error": float(refined_error),
        "error": float(full["error"]),
        "coverage": float(full.get("coverage", np.nan)),
    }


def calibration_order(q_values: np.ndarray) -> list[float]:
    return [float(q) for q in sorted(q_values, key=lambda item: (abs(item - 5.0), item))]


def nearest_completed_params(q: float, completed: dict[str, dict]) -> np.ndarray | None:
    usable = {
        key: value
        for key, value in completed.items()
        if value.get("error", 50.0) < 10.0 and "params" in value
    }
    if not usable:
        return None
    best_key = min(usable, key=lambda key: abs(float(key) - q))
    return np.asarray(usable[best_key]["params"], dtype=float)


def design(q_values: np.ndarray, degree: int) -> np.ndarray:
    y = 1.0 / q_values
    return np.vstack([y**power for power in range(degree + 1)]).T


def fit_poly(q_values: np.ndarray, values: np.ndarray, degree: int) -> np.ndarray:
    return np.linalg.lstsq(design(q_values, degree), values, rcond=None)[0]


def eval_poly(q: float, coeffs: np.ndarray) -> float:
    y = 1.0 / q
    return float(sum(coeffs[i] * y**i for i in range(len(coeffs))))


def fit_master(rows: list[dict], degree: int) -> dict[str, np.ndarray]:
    q_values = np.asarray([row["q"] for row in rows], dtype=float)
    params = np.asarray([row["params"] for row in rows], dtype=float)
    coeffs: dict[str, np.ndarray] = {}
    for index, name in enumerate(PARAM_NAMES):
        values = params[:, index].copy()
        if name == "phi0":
            values = np.unwrap(values)
        coeffs[name] = fit_poly(q_values, values, degree)
    return coeffs


def raw_master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([eval_poly(q, coeffs[name]) for name in PARAM_NAMES], dtype=float)


def constrained_master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    params = raw_master_params(q, coeffs)
    nu = q / (1.0 + q) ** 2
    params[0] = float(np.clip(params[0], -180.0, min(-5.0, -45.0 * nu)))
    params[1] = float(np.clip(params[1], 2.0, MAX_TRANSITION_WIDTH))
    if params[0] + params[1] > 60.0:
        params[1] = 60.0 - params[0]
    if params[0] + params[1] < -150.0:
        params[1] = -150.0 - params[0]
    params[1] = float(np.clip(params[1], 2.0, MAX_TRANSITION_WIDTH))
    params[2] = float(np.clip(params[2], -400.0, min(-1.0, -25.0 * nu)))
    params[3] = float(np.clip(params[3], 0.12, 1.8))
    params[4] = float(np.clip(params[4], 0.12, 1.8))
    params[5] = float(np.clip(params[5], 0.03, 2.8))
    params[6] = float(np.clip(params[6], 0.03, 2.8))
    params[7] = baseq.principal_phase(params[7])
    return params


def master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    return constrained_master_params(q, coeffs)


def polish_time_phase(params: np.ndarray, case: dict) -> tuple[np.ndarray, float]:
    start = np.array([params[2], params[7]], dtype=float)

    def objective(offset_phase: np.ndarray) -> float:
        trial = params.copy()
        trial[2] = float(offset_phase[0])
        trial[7] = float(offset_phase[1])
        return evaluate_case(trial, case, subset="fit", min_coverage=0.84)["error"]

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1.0e-7, "fatol": 1.0e-9},
    )
    out = params.copy()
    out[2] = float(result.x[0])
    out[7] = baseq.principal_phase(float(result.x[1]))
    return out, float(result.fun)


def evaluate_master(
    rows: list[dict],
    cases: dict[str, dict],
    coeffs: dict[str, np.ndarray],
) -> list[dict]:
    out = []
    for row in rows:
        q = row["q"]
        case = cases[q_key(q)]
        params = master_params(q, coeffs)
        params, fit_error = polish_time_phase(params, case)
        ev = evaluate_case(params, case, subset="full", min_coverage=0.84)
        out.append(
            {
                "q": q,
                "params": [float(x) for x in params],
                "fit_error": fit_error,
                "error": float(ev["error"]),
                "coverage": float(ev.get("coverage", np.nan)),
            }
        )
    return out


def evaluate_extrapolation(q_values: list[float], coeffs: dict[str, np.ndarray], args: argparse.Namespace) -> list[dict]:
    out = []
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    for q in q_values:
        case = baseq.load_case(nrsur, q, args.source_stride, args.nr_stride)
        raw = raw_master_params(q, coeffs)
        params = master_params(q, coeffs)
        clipped = bool(np.max(np.abs(raw[:7] - params[:7])) > 1.0e-8)
        params, fit_error = polish_time_phase(params, case)
        ev = evaluate_case(params, case, subset="full", min_coverage=0.84)
        out.append(
            {
                "q": q,
                "error": float(ev["error"]),
                "fit_error": float(fit_error),
                "coverage": float(ev.get("coverage", np.nan)),
                "clipped": clipped,
                "params": [float(x) for x in params],
            }
        )
    return out


def summarize_errors(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def should_use_cubic(summary2: dict, summary3: dict, extrap2: list[dict], extrap3: list[dict]) -> bool:
    if summary3["max"] >= 10.0:
        return False
    if summary2["max"] >= 10.0:
        return True
    max_gain = (summary2["max"] - summary3["max"]) / max(summary2["max"], 1.0e-15)
    q2_2 = next(row["error"] for row in extrap2 if abs(row["q"] - 2.0) < 1.0e-12)
    q2_3 = next(row["error"] for row in extrap3 if abs(row["q"] - 2.0) < 1.0e-12)
    q2_gain = (q2_2 - q2_3) / max(q2_2, 1.0e-15)
    median_loss = (summary3["median"] - summary2["median"]) / max(summary2["median"], 1.0e-15)
    return bool(max_gain > 0.05 or q2_gain > 0.10 or median_loss < 0.10)


def coeff_table_lines(coeffs: dict[str, np.ndarray], degree: int) -> list[str]:
    header = "| coefficient | " + " | ".join(f"c{i}" for i in range(degree + 1)) + " |"
    sep = "|:---|" + "|".join("---:" for _ in range(degree + 1)) + "|"
    lines = [header, sep]
    for name in PARAM_NAMES:
        values = " | ".join(f"{value:.12g}" for value in coeffs[name])
        lines.append(f"| {name} | {values} |")
    return lines


def write_markdown(
    rows: list[dict],
    fit_rows: list[dict],
    master2: list[dict],
    master3: list[dict],
    coeffs2: dict[str, np.ndarray],
    coeffs3: dict[str, np.ndarray],
    extrap2: list[dict],
    extrap3: list[dict],
    selected_degree: int,
    args: argparse.Namespace,
) -> None:
    selected = master3 if selected_degree == 3 else master2
    selected_coeffs = coeffs3 if selected_degree == 3 else coeffs2
    selected_extrap = extrap3 if selected_degree == 3 else extrap2

    valid_rows = [
        row for row in rows if np.isfinite(row.get("coverage", np.nan)) and row["error"] < 10.0
    ]
    perq_summary = summarize_errors([row["error"] for row in valid_rows])
    fitrow_summary = summarize_errors([row["error"] for row in fit_rows])
    master2_summary = summarize_errors([row["error"] for row in master2])
    master3_summary = summarize_errors([row["error"] for row in master3])
    selected_summary = summarize_errors([row["error"] for row in selected])
    q_at_max = max(selected, key=lambda row: row["error"])["q"]

    lines = [
        "# Two-constant q-dependent scaling fit",
        "",
        "This run tests a lower-capacity ansatz than the physical-smooth linear",
        "post-cutoff model. Both `alpha` and `beta` are constants before the",
        "cutoff and different constants after the transition. The two regimes are",
        "connected by a cubic smoothstep window, so the functions are continuous",
        "and have zero slope at both edges of the transition.",
        "",
        "Physical coordinates:",
        "",
        "```python",
        "nu = q / (1 + q)**2",
        "theta = nu * (t_bhpt - t_bhpt_merger)",
        "Theta_cut_nr = nu * (t_cut_nr - t_nr_merger)",
        "```",
        "",
        "For each q:",
        "",
        "```python",
        "theta_end = theta_cut + theta_transition_width",
        "",
        "# theta <= theta_cut",
        "alpha(theta) = alpha_left",
        "beta(theta) = beta_left",
        "",
        "# theta_cut < theta < theta_end",
        "z = (theta - theta_cut) / theta_transition_width",
        "S = 3*z**2 - 2*z**3",
        "alpha(theta) = alpha_left + (alpha_right - alpha_left) * S",
        "beta(theta) = beta_left + (beta_right - beta_left) * S",
        "",
        "# theta >= theta_end",
        "alpha(theta) = alpha_right",
        "beta(theta) = beta_right",
        "",
        "t_cut_nr = t_nr_merger + Theta_cut_nr / nu",
        "d tau / d t_bhpt = beta(theta)",
        "tau(t_bhpt_merger + theta_cut / nu) = t_cut_nr",
        "h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)",
        "```",
        "",
        "Each q-dependent coefficient is represented as a polynomial in `y = 1/q`:",
        "",
        "```python",
        "C(q) = c0 + c1*y + c2*y**2 [+ c3*y**3]",
        "```",
        "",
        f"Calibration grid: `{len(rows)}` q values from `{min(row['q'] for row in rows):.8g}` to `{max(row['q'] for row in rows):.8g}`.",
        f"Rows used in the polynomial coefficient regression: `{len(fit_rows)}`. Rows with invalid support or `mathcalE >= {args.master_outlier_cut:g}` are held out.",
        f"Fit settings: source stride `{args.source_stride}`, NR stride `{args.nr_stride}`, top starts `{args.top_n}`, maxiter `{args.maxiter}`, full-refine threshold `{args.refine_threshold:g}`.",
        f"Transition-width safety bound: `{MAX_TRANSITION_WIDTH:g}` in `theta`.",
        "",
        "The merger-centered time anchor `Theta_cut_nr` and phase `phi0` have",
        "formal q-polynomial values, but they are also treated as nuisance",
        "alignment parameters and re-optimized when reporting master and",
        "extrapolation errors.",
        "",
        "The selected polynomial degree is:",
        "",
        "```python",
        f"degree = {selected_degree}",
        "```",
        "",
        "## Selected Master Coefficients",
        "",
        *coeff_table_lines(selected_coeffs, selected_degree),
        "",
        "## Error Summary",
        "",
        "| model | min mathcalE | median mathcalE | mean mathcalE | max mathcalE | q at selected max |",
        "|:---|---:|---:|---:|---:|---:|",
        f"| independent per-q two-constant, valid-support rows | {perq_summary['min']:.8g} | {perq_summary['median']:.8g} | {perq_summary['mean']:.8g} | {perq_summary['max']:.8g} |  |",
        f"| independent per-q rows used for q-regression | {fitrow_summary['min']:.8g} | {fitrow_summary['median']:.8g} | {fitrow_summary['mean']:.8g} | {fitrow_summary['max']:.8g} |  |",
        f"| quadratic q-polynomial, optimized `Theta_cut_nr`/`phi0` | {master2_summary['min']:.8g} | {master2_summary['median']:.8g} | {master2_summary['mean']:.8g} | {master2_summary['max']:.8g} |  |",
        f"| cubic q-polynomial, optimized `Theta_cut_nr`/`phi0` | {master3_summary['min']:.8g} | {master3_summary['median']:.8g} | {master3_summary['mean']:.8g} | {master3_summary['max']:.8g} |  |",
        f"| selected q-polynomial | {selected_summary['min']:.8g} | {selected_summary['median']:.8g} | {selected_summary['mean']:.8g} | {selected_summary['max']:.8g} | {q_at_max:.8g} |",
        "",
        "## Extrapolation Diagnostics",
        "",
        "These rows are diagnostics outside or at the edge of the requested",
        "calibration domain. They are not included in the q-polynomial regression.",
        "",
        "| degree | q | mathcalE | fit-grid mathcalE | coverage | clipped by bounds |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for degree, extrap in ((2, extrap2), (3, extrap3)):
        for row in extrap:
            lines.append(
                f"| {degree} | {row['q']:.8g} | {row['error']:.8g} | "
                f"{row['fit_error']:.8g} | {row['coverage']:.6g} | {row['clipped']} |"
            )
    selected_q2 = next(row for row in selected_extrap if abs(row["q"] - 2.0) < 1.0e-12)
    if selected_q2["error"] > 1.0e-3:
        lines.extend(
            [
                "",
                "The requested q=2 extrapolation target was not achieved by this",
                "two-constant ansatz. The model remains useful as a deliberately",
                "low-capacity comparison, but its q=2 extrapolation is much worse",
                "than the calibrated-domain errors.",
            ]
        )

    lines.extend(
        [
            "",
            "## Per-q Results",
            "",
            "| q | per-q mathcalE | selected master mathcalE | coverage | used in q-fit | theta_cut | theta_width | beta_left | beta_right | alpha_left | alpha_right | master phi0 |",
            "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    selected_by_q = {q_key(row["q"]): row for row in selected}
    fit_keys = {q_key(row["q"]) for row in fit_rows}
    for row in rows:
        params = np.asarray(row["params"], dtype=float)
        master_row = selected_by_q[q_key(row["q"])]
        lines.append(
            f"| {row['q']:.8g} | {row['error']:.8g} | {master_row['error']:.8g} | "
            f"{master_row['coverage']:.6g} | {q_key(row['q']) in fit_keys} | "
            f"{params[0]:.8g} | {params[1]:.8g} | {params[3]:.8g} | "
            f"{params[4]:.8g} | {params[5]:.8g} | {params[6]:.8g} | "
            f"{master_row['params'][7]:.8g} |"
        )

    lines.extend(
        [
            "",
            "No extrapolated BHPT samples are used in the error calculation; each",
            "evaluation masks the NR waveform to the common support of the transformed",
            "BHPT time array before interpolation.",
            "",
        ]
    )
    MD_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-q", type=int, default=40)
    parser.add_argument("--q-min", type=float, default=3.0)
    parser.add_argument("--q-max", type=float, default=8.0)
    parser.add_argument("--source-stride", type=int, default=3)
    parser.add_argument("--nr-stride", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=1200)
    parser.add_argument("--refine-threshold", type=float, default=4.0e-4)
    parser.add_argument("--refine-maxiter", type=int, default=350)
    parser.add_argument("--master-outlier-cut", type=float, default=2.0e-3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    q_values = np.linspace(args.q_min, args.q_max, args.n_q)
    cache = {} if args.force else load_cache()
    smooth_cache = load_cache(SMOOTH_CACHE_PATH)
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    cases: dict[str, dict] = {}

    for q in calibration_order(q_values):
        key = q_key(q)
        case = baseq.load_case(nrsur, q, args.source_stride, args.nr_stride)
        cases[key] = case
        if key in cache and not args.force:
            print(f"cached q={q:.8g} error={cache[key]['error']:.8g}", flush=True)
            continue
        nearest = nearest_completed_params(q, cache)
        row = optimize_case(
            case,
            nearest,
            smooth_cache,
            args.top_n,
            args.maxiter,
            args.refine_threshold,
            args.refine_maxiter,
        )
        cache[key] = row
        save_cache(cache)
        print(
            f"fit q={q:.8g} fit={row['fit_error']:.8g} full={row['error']:.8g}",
            flush=True,
        )

    rows = [cache[q_key(float(q))] for q in sorted(q_values)]
    for q in sorted(q_values):
        key = q_key(float(q))
        if key not in cases:
            cases[key] = baseq.load_case(nrsur, float(q), args.source_stride, args.nr_stride)

    fit_rows = [
        row
        for row in rows
        if np.isfinite(row.get("coverage", np.nan))
        and row["error"] < args.master_outlier_cut
    ]
    if len(fit_rows) < 8:
        raise RuntimeError("Too few valid rows for q-polynomial regression.")

    coeffs2 = fit_master(fit_rows, degree=2)
    coeffs3 = fit_master(fit_rows, degree=3)
    master2 = evaluate_master(rows, cases, coeffs2)
    master3 = evaluate_master(rows, cases, coeffs3)
    extrap2 = evaluate_extrapolation([2.0, 10.0], coeffs2, args)
    extrap3 = evaluate_extrapolation([2.0, 10.0], coeffs3, args)
    summary2 = summarize_errors([row["error"] for row in master2])
    summary3 = summarize_errors([row["error"] for row in master3])
    selected_degree = 3 if should_use_cubic(summary2, summary3, extrap2, extrap3) else 2
    write_markdown(
        rows,
        fit_rows,
        master2,
        master3,
        coeffs2,
        coeffs3,
        extrap2,
        extrap3,
        selected_degree,
        args,
    )
    print(f"selected_degree={selected_degree}")
    print(f"quadratic median={summary2['median']:.8g} max={summary2['max']:.8g}")
    print(f"cubic median={summary3['median']:.8g} max={summary3['max']:.8g}")
    for row in extrap2:
        print(f"quadratic extrap q={row['q']:.8g} error={row['error']:.8g}")
    for row in extrap3:
        print(f"cubic extrap q={row['q']:.8g} error={row['error']:.8g}")
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()
