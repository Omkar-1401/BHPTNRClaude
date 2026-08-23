from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.optimize import minimize

import gwsurrogate


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.append(str(REPO_ROOT / "BHPTNRSurrogate" / "surrogates"))
import BHPTNRSur1dq1e4 as bhptsur  # noqa: E402

try:
    import fit_scaling_PN as pn_base  # noqa: E402
    import fit_scaling_PN_opt as pn_opt  # noqa: E402
except Exception:  # pragma: no cover - optional seeding only
    pn_base = None
    pn_opt = None


MODE = (2, 2)
NR_T_START = -5000.1
NR_T_END = 100.0
RESULTS_DIR = ROOT / "physical_smooth_qdep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "per_q_cache.json"
MD_PATH = ROOT / "scaling_physical_smooth_qdep.md"
PN_CACHE_PATH = ROOT / "PN_opt_q_dep_results" / "per_q_cache.json"
MAX_TRANSITION_WIDTH = 130.0

PARAM_NAMES = [
    "theta_cut",
    "theta_transition_width",
    "Theta_cut_nr",
    "beta_left",
    "beta_right_edge",
    "beta_right_slope",
    "alpha_left",
    "alpha_right_edge",
    "alpha_right_slope",
    "phi0",
]

# q=5 smooth physical fit from scaling_stricter_physical_smooth.md, rewritten
# with the NR time anchor as Theta_cut_nr = nu * (t_cut_nr - t_nr_merger).
Q5_REFERENCE_PARAMS = np.array(
    [
        -4.021138655181800e01,
        2.695563186103480e01,
        -3.210990848857200e01,
        8.042785177552890e-01,
        8.083571781548630e-01,
        1.611390106648210e-03,
        8.087788979899620e-01,
        8.133727109615770e-01,
        -6.800188331781020e-03,
        1.249764560818640e00,
    ],
    dtype=float,
)


def q_key(q: float) -> str:
    return f"{q:.10f}"


def mathcalE_error(h_ref: np.ndarray, h_model: np.ndarray) -> float:
    return float(
        np.sum(np.abs(h_ref - h_model) ** 2)
        / (2.0 * np.sum(np.abs(h_ref) ** 2))
    )


def interp_complex(t_src: np.ndarray, h_src: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    return np.interp(t_eval, t_src, h_src.real) + 1j * np.interp(
        t_eval, t_src, h_src.imag
    )


def principal_phase(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def cumulative_trapezoid_integral(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    return np.r_[0.0, np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)]


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def load_pn_cache() -> dict:
    if not PN_CACHE_PATH.exists():
        return {}
    return json.loads(PN_CACHE_PATH.read_text())


def subsample_case_arrays(case: dict, source_stride: int, nr_stride: int) -> dict:
    t_bhpt = case["t_bhpt"]
    t_nr = case["t_nr"]
    source_idx = np.unique(np.r_[np.arange(0, len(t_bhpt), source_stride), len(t_bhpt) - 1])
    nr_idx = np.unique(np.r_[np.arange(0, len(t_nr), nr_stride), len(t_nr) - 1])
    return {
        "t_bhpt": t_bhpt[source_idx],
        "h_bhpt": case["h_bhpt"][source_idx],
        "t_nr": t_nr[nr_idx],
        "h_nr": case["h_nr"][nr_idx],
    }


def load_case(nrsur, q: float, source_stride: int, nr_stride: int) -> dict:
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=[MODE], neg_modes=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_nr, h_nr, _ = nrsur(q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    nr_mask = (t_nr >= NR_T_START) & (t_nr <= NR_T_END)
    h_bhpt_22 = h_bhpt[MODE]
    h_nr_22 = h_nr[MODE][nr_mask]
    t_nr = t_nr[nr_mask]
    case = {
        "q": float(q),
        "t_bhpt": t_bhpt,
        "h_bhpt": h_bhpt_22,
        "t_nr": t_nr,
        "h_nr": h_nr_22,
        "meta": {
            "q": float(q),
            "nu": float(q / (1.0 + q) ** 2),
            "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt_22))]),
            "t_nr_merger": float(t_nr[np.argmax(np.abs(h_nr_22))]),
        },
    }
    case["fit"] = subsample_case_arrays(case, source_stride, nr_stride)
    return case


def unpack(params: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, params)}


def smooth_constant_to_line(
    theta: np.ndarray,
    theta_cut: float,
    width: float,
    left_value: float,
    right_edge_value: float,
    right_slope: float,
) -> np.ndarray:
    dtheta = theta - theta_cut
    out = np.empty_like(theta, dtype=float)

    left = dtheta <= 0.0
    right = dtheta >= width
    middle = ~(left | right)

    out[left] = left_value
    out[right] = right_edge_value + right_slope * (dtheta[right] - width)

    if np.any(middle):
        z = dtheta[middle] / width
        h00 = 2.0 * z**3 - 3.0 * z**2 + 1.0
        h01 = -2.0 * z**3 + 3.0 * z**2
        h11 = z**3 - z**2
        out[middle] = (
            h00 * left_value
            + h01 * right_edge_value
            + h11 * width * right_slope
        )

    return out


def model_arrays(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    meta: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = unpack(params)
    nu = meta["nu"]
    theta = nu * (t_bhpt - meta["t_bhpt_merger"])

    beta = smooth_constant_to_line(
        theta,
        p["theta_cut"],
        p["theta_transition_width"],
        p["beta_left"],
        p["beta_right_edge"],
        p["beta_right_slope"],
    )
    alpha = smooth_constant_to_line(
        theta,
        p["theta_cut"],
        p["theta_transition_width"],
        p["alpha_left"],
        p["alpha_right_edge"],
        p["alpha_right_slope"],
    )

    s_cut = meta["t_bhpt_merger"] + p["theta_cut"] / nu
    t_cut_nr = meta["t_nr_merger"] + p["Theta_cut_nr"] / nu
    beta_integral = cumulative_trapezoid_integral(t_bhpt, beta)
    beta_integral_cut = float(np.interp(s_cut, t_bhpt, beta_integral))
    tau = t_cut_nr + beta_integral - beta_integral_cut
    h_scaled = alpha * np.exp(1j * p["phi0"]) * h_bhpt
    return tau, h_scaled, alpha, beta, theta


def validate_basic(params: np.ndarray, meta: dict) -> bool:
    if len(params) != len(PARAM_NAMES) or not np.all(np.isfinite(params)):
        return False
    p = unpack(params)
    nu = meta["nu"]
    theta_end = p["theta_cut"] + p["theta_transition_width"]
    s_cut = meta["t_bhpt_merger"] + p["theta_cut"] / nu
    s_end = meta["t_bhpt_merger"] + theta_end / nu
    t_cut_nr = meta["t_nr_merger"] + p["Theta_cut_nr"] / nu

    if not (-165.0 <= p["theta_cut"] <= -8.0):
        return False
    if not (2.0 <= p["theta_transition_width"] <= MAX_TRANSITION_WIDTH):
        return False
    if not (-125.0 <= theta_end <= 35.0):
        return False
    if not (-1200.0 <= s_cut <= -90.0):
        return False
    if not (-1000.0 <= s_end <= 350.0):
        return False
    if not (-320.0 <= p["Theta_cut_nr"] <= -2.0):
        return False
    if not (-2000.0 <= t_cut_nr <= -30.0):
        return False
    if not (0.15 <= p["beta_left"] <= 1.7):
        return False
    if not (0.15 <= p["beta_right_edge"] <= 1.7):
        return False
    if not (-3.0e-2 <= p["beta_right_slope"] <= 3.0e-2):
        return False
    if not (0.03 <= p["alpha_left"] <= 2.8):
        return False
    if not (0.03 <= p["alpha_right_edge"] <= 2.8):
        return False
    if not (-5.0e-2 <= p["alpha_right_slope"] <= 5.0e-2):
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

    h_model = interp_complex(tau_use, h_scaled[use], t_nr[common])
    err = mathcalE_error(h_nr[common], h_model)
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


def phase_seed(params: np.ndarray, case: dict, min_coverage: float = 0.84) -> np.ndarray:
    seeded = params.copy()
    seeded[-1] = 0.0
    ev = evaluate_case(seeded, case, subset="fit", min_coverage=min_coverage)
    if ev["error"] >= 10.0:
        return params.copy()
    h_model = ev["h_model"]
    seeded[-1] = principal_phase(
        float(np.angle(np.sum(ev["h_ref"] * h_model.conjugate())))
    )
    return seeded


def anchored_q5_seed(q: float, theta_shift: float = 0.0, width_scale: float = 1.0) -> np.ndarray:
    params = Q5_REFERENCE_PARAMS.copy()
    mass_scale = (q / (1.0 + q)) / (5.0 / 6.0)
    params[0] += theta_shift
    params[1] *= width_scale
    params[3] *= mass_scale
    params[4] *= mass_scale
    params[5] *= mass_scale
    params[6] *= mass_scale
    params[7] *= mass_scale
    params[8] *= mass_scale
    params[2] = params[3] * params[0]
    return params


def pn_seed_from_cache(case: dict, pn_cache: dict) -> np.ndarray | None:
    if pn_base is None or pn_opt is None:
        return None
    key = q_key(case["q"])
    if key not in pn_cache:
        return None
    row = pn_cache[key]
    if row.get("error", 50.0) >= 0.01:
        return None

    pn_params = np.asarray(row["params"], dtype=float)
    try:
        losses = pn_base.pn_loss_coordinates(case["t_bhpt"], case["h_bhpt"], case["meta"])
        p = pn_opt.unpack(pn_params)
        p_loss = losses["p_loss"]
        t_bhpt = case["t_bhpt"]
        theta = case["meta"]["nu"] * (t_bhpt - case["meta"]["t_bhpt_merger"])
        p_start = p["p_start"]
        p_end = p["p_end"]
        t_start = float(np.interp(p_start, p_loss, t_bhpt))
        t_end = float(np.interp(p_end, p_loss, t_bhpt))
        theta_cut = float(np.interp(t_start, t_bhpt, theta))
        theta_end = float(np.interp(t_end, t_bhpt, theta))
        width = theta_end - theta_cut

        _tau, _h_scaled, alpha, beta, _parts = pn_opt.model_arrays(
            pn_params, t_bhpt, case["h_bhpt"], losses
        )
        alpha_edge = float(np.interp(t_end, t_bhpt, alpha))
        beta_edge = float(np.interp(t_end, t_bhpt, beta))
        alpha_slope = float(np.interp(theta_end, theta, np.gradient(alpha, theta, edge_order=2)))
        beta_slope = float(np.interp(theta_end, theta, np.gradient(beta, theta, edge_order=2)))
        theta_cut_nr = case["meta"]["nu"] * (p["t_start_nr"] - case["meta"]["t_nr_merger"])

        params = np.array(
            [
                theta_cut,
                width,
                theta_cut_nr,
                p["beta_left"],
                beta_edge,
                beta_slope,
                p["alpha_left"],
                alpha_edge,
                alpha_slope,
                p["phi0"],
            ],
            dtype=float,
        )
    except Exception:
        return None

    if validate_basic(params, case["meta"]):
        return params
    return None


def candidate_starts(
    case: dict,
    nearest_params: np.ndarray | None,
    pn_cache: dict,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    if nearest_params is not None:
        starts.append(nearest_params.copy())
        for theta_shift in (-6.0, 6.0):
            shifted = nearest_params.copy()
            shifted[0] += theta_shift
            shifted[2] = shifted[3] * shifted[0]
            starts.append(shifted)
        for scale in (0.985, 1.015):
            shifted = nearest_params.copy()
            shifted[3] *= scale
            shifted[4] *= scale
            shifted[6] *= scale
            shifted[7] *= scale
            starts.append(shifted)

    pn_seed = pn_seed_from_cache(case, pn_cache)
    if pn_seed is not None:
        starts.append(pn_seed)

    for theta_shift in (-22.0, -12.0, 0.0, 10.0, 20.0):
        for width_scale in (0.65, 0.9, 1.0, 1.2, 1.45):
            starts.append(anchored_q5_seed(case["q"], theta_shift, width_scale))

    for theta_cut in (-80.0, -60.0, -45.0, -35.0, -25.0):
        for width in (10.0, 18.0, 28.0, 40.0):
            seed = anchored_q5_seed(case["q"])
            seed[0] = theta_cut
            seed[1] = width
            seed[2] = seed[3] * theta_cut
            starts.append(seed)

    seeded = []
    seen = set()
    for start in starts:
        if not validate_basic(start, case["meta"]):
            continue
        rounded = tuple(np.round(start, decimals=10))
        if rounded in seen:
            continue
        seen.add(rounded)
        seeded.append(phase_seed(start, case))
    return seeded


def optimize_case(
    case: dict,
    nearest_params: np.ndarray | None,
    pn_cache: dict,
    top_n: int,
    maxiter: int,
    refine_threshold: float,
    refine_maxiter: int,
) -> dict:
    starts = candidate_starts(case, nearest_params, pn_cache)
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
    for _rank, (_start_error, start) in enumerate(ranked[:top_n]):
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


def constrained_master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    params = np.asarray([eval_poly(q, coeffs[name]) for name in PARAM_NAMES], dtype=float)
    nu = q / (1.0 + q) ** 2

    params[0] = float(np.clip(params[0], -165.0, -95.0 * nu))
    params[1] = float(np.clip(params[1], 2.0, MAX_TRANSITION_WIDTH))
    if params[0] + params[1] > 35.0:
        params[1] = 35.0 - params[0]
    if params[0] + params[1] < -125.0:
        params[1] = -125.0 - params[0]
    params[1] = float(np.clip(params[1], 2.0, MAX_TRANSITION_WIDTH))

    params[2] = float(np.clip(params[2], -320.0, -2.0))
    params[3] = float(np.clip(params[3], 0.15, 1.7))
    params[4] = float(np.clip(params[4], 0.15, 1.7))
    params[5] = float(np.clip(params[5], -3.0e-2, 3.0e-2))
    params[6] = float(np.clip(params[6], 0.03, 2.8))
    params[7] = float(np.clip(params[7], 0.03, 2.8))
    params[8] = float(np.clip(params[8], -5.0e-2, 5.0e-2))
    params[9] = principal_phase(params[9])
    return params


def master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    return constrained_master_params(q, coeffs)


def polish_time_phase(params: np.ndarray, case: dict) -> tuple[np.ndarray, float]:
    start = np.array([params[2], params[9]], dtype=float)

    def objective(offset_phase: np.ndarray) -> float:
        trial = params.copy()
        trial[2] = float(offset_phase[0])
        trial[9] = float(offset_phase[1])
        return evaluate_case(trial, case, subset="fit", min_coverage=0.84)["error"]

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1.0e-7, "fatol": 1.0e-9},
    )
    out = params.copy()
    out[2] = float(result.x[0])
    out[9] = principal_phase(float(result.x[1]))
    return out, float(result.fun)


def evaluate_master(
    rows: list[dict],
    cases: dict[str, dict],
    coeffs: dict[str, np.ndarray],
    polish_nuisance: bool,
) -> list[dict]:
    out = []
    for row in rows:
        q = row["q"]
        params = master_params(q, coeffs)
        case = cases[q_key(q)]
        fit_error = np.nan
        if polish_nuisance:
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


def summarize_errors(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def should_use_cubic(summary2: dict, summary3: dict) -> bool:
    median_gain = (summary2["median"] - summary3["median"]) / max(summary2["median"], 1.0e-15)
    max_gain = (summary2["max"] - summary3["max"]) / max(summary2["max"], 1.0e-15)
    return bool(median_gain > 0.03 or max_gain > 0.03)


def coeff_table_lines(coeffs: dict[str, np.ndarray], degree: int) -> list[str]:
    header = "| coefficient | " + " | ".join(f"c{i}" for i in range(degree + 1)) + " |"
    sep = "|:---|" + "|".join("---:" for _ in range(degree + 1)) + "|"
    lines = [header, sep]
    for name in PARAM_NAMES:
        values = " | ".join(f"{value:.12g}" for value in coeffs[name])
        lines.append(f"| {name} | {values} |")
    return lines


def raw_master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([eval_poly(q, coeffs[name]) for name in PARAM_NAMES], dtype=float)


def formal_parameter_row(q: float, coeffs: dict[str, np.ndarray]) -> str:
    raw = raw_master_params(q, coeffs)
    params = master_params(q, coeffs)
    note = "clipped by bounds" if np.max(np.abs(raw[:9] - params[:9])) > 1.0e-8 else ""
    return (
        f"| {q:.8g} | {params[0]:.8g} | {params[1]:.8g} | {params[2]:.8g} | "
        f"{params[3]:.8g} | {params[4]:.8g} | {params[5]:.8g} | "
        f"{params[6]:.8g} | {params[7]:.8g} | {params[8]:.8g} | {note} |"
    )


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
    valid_rows = [
        row
        for row in rows
        if np.isfinite(row.get("coverage", np.nan)) and row["error"] < 10.0
    ]
    perq_summary = summarize_errors([row["error"] for row in valid_rows])
    fitrow_summary = summarize_errors([row["error"] for row in fit_rows])
    master2_summary = summarize_errors([row["error"] for row in master2])
    master3_summary = summarize_errors([row["error"] for row in master3])
    selected_summary = summarize_errors([row["error"] for row in selected])
    q_at_max = max(selected, key=lambda row: row["error"])["q"]

    lines = [
        "# Physical-smooth q-dependent scaling fit",
        "",
        "This run generalizes the q=5 `stricter_physical_smooth` ansatz over",
        "mass ratios between 3 and 8. The ansatz is unchanged in structure:",
        "`alpha` and `beta` are constant before a physical cutoff, pass through",
        "a cubic Hermite transition, and are linear in the merger-centered",
        "physical source time coordinate after the transition.",
        "",
        "The physical coordinates are:",
        "",
        "```python",
        "nu = q / (1 + q)**2",
        "theta = nu * (t_bhpt - t_bhpt_merger)",
        "Theta_nr = nu * (t_nr - t_nr_merger)",
        "Theta_cut_nr = nu * (t_cut_nr - t_nr_merger)",
        "```",
        "",
        "The q-dependence is represented by polynomials in `y = 1/q`:",
        "",
        "```python",
        "C(q) = c0 + c1*y + c2*y**2 [+ c3*y**3]",
        "```",
        "",
        "For a given q, the coefficients define:",
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
        "H00 = 2*z**3 - 3*z**2 + 1",
        "H01 = -2*z**3 + 3*z**2",
        "H11 = z**3 - z**2",
        "y_smooth = H00*y_left + H01*y_right_edge + H11*theta_transition_width*y_right_slope",
        "",
        "# theta >= theta_end",
        "alpha(theta) = alpha_right_edge + alpha_right_slope * (theta - theta_end)",
        "beta(theta) = beta_right_edge + beta_right_slope * (theta - theta_end)",
        "",
        "t_cut_nr = t_nr_merger + Theta_cut_nr / nu",
        "d tau / d t_bhpt = beta(theta)",
        "tau(t_bhpt_merger + theta_cut / nu) = t_cut_nr",
        "h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)",
        "```",
        "",
        f"Calibration grid: `{len(rows)}` q values from `{min(row['q'] for row in rows):.8g}` to `{max(row['q'] for row in rows):.8g}`.",
        f"Rows used in the polynomial coefficient regression: `{len(fit_rows)}`. Rows with invalid support or `mathcalE >= {args.master_outlier_cut:g}` are held out of the regression.",
        f"Fit settings: source stride `{args.source_stride}`, NR stride `{args.nr_stride}`, top starts `{args.top_n}`, maxiter `{args.maxiter}`, full-refine threshold `{args.refine_threshold:g}`.",
        f"The transition-width safety bound used by this script is `{MAX_TRANSITION_WIDTH:g}` in `theta`.",
        "For the saved cache used here, the low-q rows were also given a targeted",
        "denser cleanup pass after the wider transition bound was introduced;",
        "the per-q table below is the authoritative record of the final cached",
        "independent fits.",
        "",
        "The master q-polynomial describes the physical shape coefficients.",
        "The merger-centered time anchor `Theta_cut_nr` and phase `phi0` have",
        "formal q-polynomial values, but they are also treated as nuisance",
        "alignment parameters and re-optimized when reporting the master-model",
        "error statistics. This mirrors the previous PN-opt q-dependent workflow",
        "and keeps the alpha/beta calibration from absorbing arbitrary absolute",
        "time and phase offsets.",
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
        f"| independent per-q physical-smooth, valid-support rows | {perq_summary['min']:.8g} | {perq_summary['median']:.8g} | {perq_summary['mean']:.8g} | {perq_summary['max']:.8g} |  |",
        f"| independent per-q rows used for q-regression | {fitrow_summary['min']:.8g} | {fitrow_summary['median']:.8g} | {fitrow_summary['mean']:.8g} | {fitrow_summary['max']:.8g} |  |",
        f"| quadratic q-polynomial, optimized `Theta_cut_nr`/`phi0` | {master2_summary['min']:.8g} | {master2_summary['median']:.8g} | {master2_summary['mean']:.8g} | {master2_summary['max']:.8g} |  |",
        f"| cubic q-polynomial, optimized `Theta_cut_nr`/`phi0` | {master3_summary['min']:.8g} | {master3_summary['median']:.8g} | {master3_summary['mean']:.8g} | {master3_summary['max']:.8g} |  |",
        f"| selected q-polynomial | {selected_summary['min']:.8g} | {selected_summary['median']:.8g} | {selected_summary['mean']:.8g} | {selected_summary['max']:.8g} | {q_at_max:.8g} |",
        "",
        "The extrapolation request to q=2 and q=10 is treated as formal polynomial",
        "extrapolation only. The calibration and error statistics above use q in [3, 8].",
        "",
        "## Formal Extrapolated Parameter Values",
        "",
        "| q | theta_cut | theta_width | Theta_cut_nr | beta_left | beta_edge | beta_slope | alpha_left | alpha_edge | alpha_slope | note |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
        formal_parameter_row(2.0, selected_coeffs),
        formal_parameter_row(10.0, selected_coeffs),
        "",
        "## Per-q Results",
        "",
        "| q | per-q mathcalE | selected master mathcalE | coverage | used in q-fit | theta_cut | theta_width | beta_left | alpha_left | master phi0 |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    selected_by_q = {q_key(row["q"]): row for row in selected}
    fit_keys = {q_key(row["q"]) for row in fit_rows}
    for row in rows:
        params = np.asarray(row["params"], dtype=float)
        master_row = selected_by_q[q_key(row["q"])]
        lines.append(
            f"| {row['q']:.8g} | {row['error']:.8g} | {master_row['error']:.8g} | "
            f"{master_row['coverage']:.6g} | {q_key(row['q']) in fit_keys} | "
            f"{params[0]:.8g} | {params[1]:.8g} | {params[3]:.8g} | "
            f"{params[6]:.8g} | {master_row['params'][9]:.8g} |"
        )
    lines.extend(
        [
            "",
            "No extrapolated BHPT samples are used in the error calculation; each",
            "evaluation masks the NR waveform to the common support of the transformed",
            "BHPT time array before interpolation.",
            "",
            "No plotting files were generated in this run, per the current instruction",
            "to focus on the calibration script and markdown first.",
            "",
        ]
    )
    MD_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-q", type=int, default=41)
    parser.add_argument("--q-min", type=float, default=3.0)
    parser.add_argument("--q-max", type=float, default=8.0)
    parser.add_argument("--source-stride", type=int, default=3)
    parser.add_argument("--nr-stride", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=850)
    parser.add_argument("--refine-threshold", type=float, default=2.5e-4)
    parser.add_argument("--refine-maxiter", type=int, default=180)
    parser.add_argument("--master-outlier-cut", type=float, default=2.0e-3)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    q_values = np.linspace(args.q_min, args.q_max, args.n_q)
    cache = {} if args.force else load_cache()
    pn_cache = load_pn_cache()
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    cases: dict[str, dict] = {}

    for q in calibration_order(q_values):
        key = q_key(q)
        case = load_case(nrsur, q, args.source_stride, args.nr_stride)
        cases[key] = case
        if key in cache and not args.force:
            print(f"cached q={q:.8g} error={cache[key]['error']:.8g}", flush=True)
            continue
        nearest = nearest_completed_params(q, cache)
        row = optimize_case(
            case,
            nearest,
            pn_cache,
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
            cases[key] = load_case(nrsur, float(q), args.source_stride, args.nr_stride)

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
    master2 = evaluate_master(rows, cases, coeffs2, polish_nuisance=True)
    master3 = evaluate_master(rows, cases, coeffs3, polish_nuisance=True)
    summary2 = summarize_errors([row["error"] for row in master2])
    summary3 = summarize_errors([row["error"] for row in master3])
    selected_degree = 3 if should_use_cubic(summary2, summary3) else 2
    write_markdown(rows, fit_rows, master2, master3, coeffs2, coeffs3, selected_degree, args)
    print(f"selected_degree={selected_degree}")
    print(f"quadratic median={summary2['median']:.8g} max={summary2['max']:.8g}")
    print(f"cubic median={summary3['median']:.8g} max={summary3['max']:.8g}")
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()
