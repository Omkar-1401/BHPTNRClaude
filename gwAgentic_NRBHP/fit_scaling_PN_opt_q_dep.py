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
import fit_scaling_PN as base
import fit_scaling_PN_opt as smooth


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.append(str(REPO_ROOT / "BHPTNRSurrogate" / "surrogates"))
import BHPTNRSur1dq1e4 as bhptsur  # noqa: E402


MODE = base.MODE
NR_T_START = base.NR_T_START
NR_T_END = base.NR_T_END
T_REF = base.T_REF
RESULTS_DIR = ROOT / "PN_opt_q_dep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "per_q_cache.json"
MD_PATH = ROOT / "scaling_PN_opt_q_dep.md"

PARAM_NAMES = [
    "p_start",
    "p_width",
    "t_start_nr",
    "beta_left",
    "beta_right_edge",
    "beta_right_e",
    "beta_right_j",
    "alpha_left",
    "alpha_right_edge",
    "alpha_right_e",
    "alpha_right_j",
    "phi0",
]


def q_key(q: float) -> str:
    return f"{q:.10f}"


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def load_case(nrsur, q: float, source_stride: int, nr_stride: int) -> dict:
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=[MODE], neg_modes=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_nr, h_nr, _ = nrsur(q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    nr_mask = (t_nr >= NR_T_START) & (t_nr <= NR_T_END)
    t_nr = t_nr[nr_mask]
    h_nr_22 = h_nr[MODE][nr_mask]
    h_bhpt_22 = h_bhpt[MODE]
    meta = {
        "nu": q / (1.0 + q) ** 2,
        "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt_22))]),
        "t_nr_merger": float(t_nr[np.argmax(np.abs(h_nr_22))]),
    }
    losses = base.pn_loss_coordinates(t_bhpt, h_bhpt_22, meta)
    source_idx = np.unique(np.r_[np.arange(0, len(t_bhpt), source_stride), len(t_bhpt) - 1])
    nr_idx = np.unique(np.r_[np.arange(0, len(t_nr), nr_stride), len(t_nr) - 1])
    return {
        "q": q,
        "t_bhpt": t_bhpt,
        "h_bhpt": h_bhpt_22,
        "t_nr": t_nr,
        "h_nr": h_nr_22,
        "losses": losses,
        "meta": meta,
        "fit": {
            "t_bhpt": t_bhpt[source_idx],
            "h_bhpt": h_bhpt_22[source_idx],
            "t_nr": t_nr[nr_idx],
            "h_nr": h_nr_22[nr_idx],
            "losses": subsample_losses(losses, source_idx),
        },
    }


def subsample_losses(losses: dict, idx: np.ndarray) -> dict:
    out = {}
    for key, value in losses.items():
        if isinstance(value, np.ndarray):
            out[key] = value[idx]
        else:
            out[key] = value
    return out


def anchored_start(params: np.ndarray, case: dict) -> np.ndarray:
    out = params.copy()
    p = smooth.unpack(out)
    t_bhpt_start = float(np.interp(p["p_start"], case["losses"]["p_loss"], case["t_bhpt"]))
    out[2] = case["meta"]["t_nr_merger"] + p["beta_left"] * (
        t_bhpt_start - case["meta"]["t_bhpt_merger"]
    )
    return out


def q_scaled_seed(q: float, case: dict) -> np.ndarray:
    mass_norm = q / (1.0 + q)
    seed = smooth.ACCEPTED_PN_OPT_PARAMS.copy()
    seed[3] = mass_norm
    seed[4] = mass_norm
    seed[7] = mass_norm
    seed[8] = mass_norm
    return anchored_start(seed, case)


def phase_seed(params: np.ndarray, case: dict) -> np.ndarray:
    seeded = params.copy()
    seeded[-1] = 0.0
    ev = smooth.evaluate_model(
        seeded,
        case["fit"]["t_bhpt"],
        case["fit"]["h_bhpt"],
        case["fit"]["t_nr"],
        case["fit"]["h_nr"],
        case["fit"]["losses"],
        min_coverage=0.84,
    )
    if ev["error"] >= 10.0:
        return params.copy()
    h_model = ev["h_model"]
    h_ref = ev["h_ref"]
    seeded[-1] = float(np.angle(np.sum(h_ref * h_model.conjugate())))
    return seeded


def candidate_starts(case: dict, nearest_params: np.ndarray | None) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    if nearest_params is not None:
        starts.append(anchored_start(nearest_params, case))
        for alpha_scale in (0.98, 1.02):
            shifted = anchored_start(nearest_params, case)
            shifted[7] *= alpha_scale
            shifted[8] *= alpha_scale
            starts.append(shifted)

    starts.append(anchored_start(smooth.ACCEPTED_PN_OPT_PARAMS, case))
    starts.append(q_scaled_seed(case["q"], case))

    for p_shift, width_scale in ((-0.06, 1.0), (0.0, 0.75), (0.0, 1.25), (0.06, 1.0)):
        shifted = anchored_start(smooth.ACCEPTED_PN_OPT_PARAMS, case)
        shifted[0] += p_shift
        shifted[1] *= width_scale
        shifted = anchored_start(shifted, case)
        starts.append(shifted)

    return [phase_seed(start, case) for start in starts if smooth.validate_basic(start)]


def optimize_case(
    case: dict,
    nearest_params: np.ndarray | None,
    top_n: int,
    maxiter: int,
) -> dict:
    starts = candidate_starts(case, nearest_params)
    ranked = []
    for start in starts:
        ev = smooth.evaluate_model(
            start,
            case["fit"]["t_bhpt"],
            case["fit"]["h_bhpt"],
            case["fit"]["t_nr"],
            case["fit"]["h_nr"],
            case["fit"]["losses"],
            min_coverage=0.84,
        )
        if ev["error"] < 10.0:
            ranked.append((float(ev["error"]), start))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        raise RuntimeError(f"No usable starts for q={case['q']:.8g}")

    best_params = ranked[0][1].copy()
    best_fit_error = ranked[0][0]
    for _rank, (start_error, start) in enumerate(ranked[:top_n]):
        result = minimize(
            lambda params: smooth.evaluate_model(
                params,
                case["fit"]["t_bhpt"],
                case["fit"]["h_bhpt"],
                case["fit"]["t_nr"],
                case["fit"]["h_nr"],
                case["fit"]["losses"],
                min_coverage=0.84,
            )["error"],
            start,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1.0e-7, "fatol": 1.0e-9},
        )
        if float(result.fun) < best_fit_error:
            best_fit_error = float(result.fun)
            best_params = result.x.copy()

    full = smooth.evaluate_model(
        best_params,
        case["t_bhpt"],
        case["h_bhpt"],
        case["t_nr"],
        case["h_nr"],
        case["losses"],
        min_coverage=0.84,
    )
    return {
        "q": float(case["q"]),
        "params": [float(x) for x in best_params],
        "fit_error": float(best_fit_error),
        "error": float(full["error"]),
        "coverage": float(full.get("coverage", np.nan)),
    }


def calibration_order(q_values: np.ndarray) -> list[float]:
    return [float(q) for q in sorted(q_values, key=lambda item: (abs(item - 5.0), item))]


def nearest_completed_params(q: float, completed: dict[str, dict]) -> np.ndarray | None:
    if not completed:
        return None
    best_key = min(completed, key=lambda key: abs(float(key) - q))
    return np.asarray(completed[best_key]["params"], dtype=float)


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
    params[0] = float(np.clip(params[0], -3.0, 0.05))
    params[1] = float(np.clip(params[1], 0.015, 0.45))
    if params[0] + params[1] > 0.20:
        params[1] = 0.20 - params[0]
    params[3] = float(np.clip(params[3], 0.2, 1.6))
    params[4] = float(np.clip(params[4], 0.2, 1.6))
    params[7] = float(np.clip(params[7], 0.05, 2.5))
    params[8] = float(np.clip(params[8], 0.05, 2.5))
    return params


def master_params(q: float, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    return constrained_master_params(q, coeffs)


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
            def objective(offset_phase: np.ndarray) -> float:
                trial = params.copy()
                trial[2] = offset_phase[0]
                trial[11] = offset_phase[1]
                return smooth.evaluate_model(
                    trial,
                    case["fit"]["t_bhpt"],
                    case["fit"]["h_bhpt"],
                    case["fit"]["t_nr"],
                    case["fit"]["h_nr"],
                    case["fit"]["losses"],
                    min_coverage=0.84,
                )["error"]

            result = minimize(
                objective,
                [params[2], params[11]],
                method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1.0e-7, "fatol": 1.0e-9},
            )
            params[2] = float(result.x[0])
            params[11] = float(result.x[1])
            fit_error = float(result.fun)
        ev = smooth.evaluate_model(
            params,
            case["t_bhpt"],
            case["h_bhpt"],
            case["t_nr"],
            case["h_nr"],
            case["losses"],
            min_coverage=0.84,
        )
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
    median_gain = (summary2["median"] - summary3["median"]) / max(summary2["median"], 1e-15)
    max_gain = (summary2["max"] - summary3["max"]) / max(summary2["max"], 1e-15)
    return bool(median_gain > 0.03 or max_gain > 0.03)


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
        "# PN-opt q-dependent scaling fit",
        "",
        "This fit generalizes the smooth PN-loss q=5 optimized ansatz over mass",
        "ratios between 3 and 8. The per-q model uses a constant pre-transition",
        "branch, a cubic Hermite smooth transition, and a post-transition branch",
        "linear in the normalized PN radiated-energy and angular-momentum losses.",
        "",
        "The q-dependence is represented by polynomials in `y = 1/q`:",
        "",
        "```python",
        "C(q) = c0 + c1*y + c2*y**2 [+ c3*y**3]",
        "```",
        "",
        f"Calibration grid: `{len(rows)}` q values from `{min(row['q'] for row in rows):.8g}` to `{max(row['q'] for row in rows):.8g}`.",
        f"Rows used in the polynomial coefficient regression: `{len(fit_rows)}`. Rows with invalid support or `mathcalE >= {args.master_outlier_cut:g}` are held out of the regression.",
        f"Fit settings: source stride `{args.source_stride}`, NR stride `{args.nr_stride}`, top starts `{args.top_n}`, maxiter `{args.maxiter}`.",
        "",
        "The q-polynomial master model uses q-dependent PN-transition and",
        "alpha/beta coefficients. The arbitrary absolute time anchor `t_start_nr`",
        "and phase rotation `phi0` are treated as nuisance alignments and",
        "re-optimized when reporting the master-model error statistics.",
        "The constrained transition parameters are clipped to the same physical",
        "bounds used by the per-q optimizer, most importantly `p_width >= 0.015`,",
        "so that the polynomial fit cannot create an invalid negative-width",
        "transition at the low-q edge.",
        "",
        "The final selected polynomial degree is:",
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
        f"| independent per-q smooth PN-opt, valid-support rows | {perq_summary['min']:.8g} | {perq_summary['median']:.8g} | {perq_summary['mean']:.8g} | {perq_summary['max']:.8g} |  |",
        f"| independent per-q rows used for q-regression | {fitrow_summary['min']:.8g} | {fitrow_summary['median']:.8g} | {fitrow_summary['mean']:.8g} | {fitrow_summary['max']:.8g} |  |",
        f"| quadratic q-polynomial, optimized time/phase | {master2_summary['min']:.8g} | {master2_summary['median']:.8g} | {master2_summary['mean']:.8g} | {master2_summary['max']:.8g} |  |",
        f"| cubic q-polynomial, optimized time/phase | {master3_summary['min']:.8g} | {master3_summary['median']:.8g} | {master3_summary['mean']:.8g} | {master3_summary['max']:.8g} |  |",
        f"| selected q-polynomial | {selected_summary['min']:.8g} | {selected_summary['median']:.8g} | {selected_summary['mean']:.8g} | {selected_summary['max']:.8g} | {q_at_max:.8g} |",
        "",
        "The extrapolation request to q=2 and q=10 is treated as a formal polynomial",
        "extrapolation. The calibration and error statistics above only use q in [3, 8].",
        "",
        "## Per-q Results",
        "",
        "| q | per-q mathcalE | selected master mathcalE | coverage | used in q-fit | p_start | p_width | beta_left | alpha_left | phi0 master |",
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
            f"{params[7]:.8g} | {master_row['params'][11]:.8g} |"
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
    parser.add_argument("--n-q", type=int, default=41)
    parser.add_argument("--q-min", type=float, default=3.0)
    parser.add_argument("--q-max", type=float, default=8.0)
    parser.add_argument("--source-stride", type=int, default=3)
    parser.add_argument("--nr-stride", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=900)
    parser.add_argument("--master-outlier-cut", type=float, default=2.0e-3)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    q_values = np.linspace(args.q_min, args.q_max, args.n_q)
    cache = {} if args.force else load_cache()
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
        row = optimize_case(case, nearest, args.top_n, args.maxiter)
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
