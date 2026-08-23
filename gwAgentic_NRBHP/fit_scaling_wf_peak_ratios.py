from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

import gwsurrogate


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.append(str(REPO_ROOT / "BHPTNRSurrogate" / "surrogates"))
import BHPTNRSur1dq1e4 as bhptsur  # noqa: E402


MODE = (2, 2)
NR_T_START = -5000.1
NR_T_END = 100.0

RESULTS_DIR = ROOT / "wf_peak_ratios_results"
RESULTS_DIR.mkdir(exist_ok=True)
PEAK_CACHE_PATH = RESULTS_DIR / "peak_ratio_cache.json"
RESULTS_PATH = RESULTS_DIR / "fit_results.json"
MD_PATH = ROOT / "scaling_wf_peak_ratios.md"

ALPHA_BOUNDS = (0.03, 2.8)
BETA_BOUNDS = (0.05, 2.5)
STANDARD_EVAL_MIN_COVERAGE = 0.84


def q_key(q: float) -> str:
    return f"{q:.10f}"


def mathcalE_error(h_ref: np.ndarray, h_model: np.ndarray) -> float:
    return float(
        np.sum(np.abs(h_ref - h_model) ** 2)
        / (2.0 * np.sum(np.abs(h_ref) ** 2))
    )


def interp_complex(
    t_src: np.ndarray, h_src: np.ndarray, t_eval: np.ndarray
) -> np.ndarray:
    return np.interp(t_eval, t_src, h_src.real) + 1j * np.interp(
        t_eval, t_src, h_src.imag
    )


def principal_phase(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def q_coordinate_value(q: np.ndarray | float, coordinate: str) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    if coordinate == "q":
        return q_arr
    if coordinate == "mass_norm":
        return q_arr / (1.0 + q_arr)
    if coordinate == "delta":
        return (q_arr - 1.0) / (q_arr + 1.0)
    if coordinate == "nu":
        return q_arr / (1.0 + q_arr) ** 2
    if coordinate == "invq":
        return 1.0 / q_arr
    raise ValueError(f"Unsupported q coordinate: {coordinate}")


def load_case(nrsur, q: float) -> dict:
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
    return {
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


def rotate_to_merger_peak(h: np.ndarray) -> np.ndarray:
    peak_index = int(np.argmax(np.abs(h)))
    return h * np.exp(-1j * np.angle(h[peak_index]))


def refined_peak(
    t: np.ndarray, y: np.ndarray, index: int, half_width: int = 3
) -> tuple[float, float]:
    lo = max(0, index - half_width)
    hi = min(len(t), index + half_width + 1)
    if hi - lo < 4:
        return float(t[index]), float(y[index])

    ts = t[lo:hi]
    ys = y[lo:hi]
    order = np.argsort(ts)
    ts = ts[order]
    ys = ys[order]
    spline = CubicSpline(ts, ys)
    result = minimize_scalar(
        lambda x: -float(spline(x)),
        bounds=(float(ts[0]), float(ts[-1])),
        method="bounded",
    )
    if not result.success:
        return float(t[index]), float(y[index])
    return float(result.x), float(spline(result.x))


def waveform_peaks(
    t: np.ndarray,
    h: np.ndarray,
    t_merger: float,
    min_peak_time: float,
    max_peak_time: float,
    prominence_fraction: float,
) -> list[dict]:
    h_rot = rotate_to_merger_peak(h)
    y = np.real(h_rot)
    dt = float(np.median(np.diff(t)))
    distance = max(8, int(round(8.0 / max(dt, 1.0e-6))))
    prominence = prominence_fraction * float(np.max(np.abs(y)))
    peak_indices, _props = find_peaks(y, distance=distance, prominence=prominence)

    peaks: list[dict] = []
    for index in peak_indices:
        s0 = float(t[index] - t_merger)
        if not (min_peak_time <= s0 <= max_peak_time):
            continue
        t_peak, h_peak = refined_peak(t, y, int(index))
        s_peak = t_peak - t_merger
        if not (min_peak_time <= s_peak <= max_peak_time) or h_peak <= 0.0:
            continue
        peaks.append({"t": t_peak, "s": s_peak, "value": h_peak})

    peaks.sort(key=lambda row: row["s"], reverse=True)
    for k, peak in enumerate(peaks, start=1):
        peak["k_before_merger"] = k
    return peaks


def peak_ratio_rows(case: dict, args: argparse.Namespace) -> list[dict]:
    meta = case["meta"]
    nr_peaks = waveform_peaks(
        case["t_nr"],
        case["h_nr"],
        meta["t_nr_merger"],
        args.min_peak_time,
        args.max_peak_time,
        args.prominence_fraction,
    )
    bhpt_peaks = waveform_peaks(
        case["t_bhpt"],
        case["h_bhpt"],
        meta["t_bhpt_merger"],
        args.min_peak_time / BETA_BOUNDS[0],
        args.max_peak_time,
        args.prominence_fraction,
    )

    rows: list[dict] = []
    for idx in range(min(len(nr_peaks), len(bhpt_peaks))):
        nr = nr_peaks[idx]
        bhpt = bhpt_peaks[idx]
        alpha = nr["value"] / bhpt["value"]
        beta = nr["s"] / bhpt["s"]
        if not (ALPHA_BOUNDS[0] <= alpha <= ALPHA_BOUNDS[1]):
            continue
        if not (BETA_BOUNDS[0] <= beta <= BETA_BOUNDS[1]):
            continue
        rows.append(
            {
                "q": float(case["q"]),
                "nu": float(meta["nu"]),
                "k_before_merger": int(idx + 1),
                "t_nr_peak": float(nr["t"]),
                "t_bhpt_peak": float(bhpt["t"]),
                "s_nr_peak": float(nr["s"]),
                "s_bhpt_peak": float(bhpt["s"]),
                "theta": float(meta["nu"] * bhpt["s"]),
                "theta_nr": float(meta["nu"] * nr["s"]),
                "alpha_peak": float(alpha),
                "beta_peak": float(beta),
                "h_nr_peak": float(nr["value"]),
                "h_bhpt_peak": float(bhpt["value"]),
            }
        )
    return rows


def load_peak_cache() -> dict:
    if not PEAK_CACHE_PATH.exists():
        return {}
    return json.loads(PEAK_CACHE_PATH.read_text())


def save_peak_cache(cache: dict) -> None:
    PEAK_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def build_peak_tables(rows_by_q: dict[str, list[dict]], max_k: int) -> dict:
    tables: dict[str, dict[str, dict[str, list[float]]]] = {
        "alpha_peak": {},
        "beta_peak": {},
    }
    for rows in rows_by_q.values():
        for row in rows:
            k = int(row["k_before_merger"])
            if k > max_k:
                continue
            for value_name in ("alpha_peak", "beta_peak"):
                bucket = tables[value_name].setdefault(
                    str(k), {"q": [], "value": []}
                )
                bucket["q"].append(float(row["q"]))
                bucket["value"].append(float(row[value_name]))

    for value_tables in tables.values():
        for bucket in value_tables.values():
            q_arr = np.asarray(bucket["q"], dtype=float)
            value_arr = np.asarray(bucket["value"], dtype=float)
            order = np.argsort(q_arr)
            bucket["q"] = q_arr[order].tolist()
            bucket["value"] = value_arr[order].tolist()
    return tables


def extrapolate_edge(
    x: np.ndarray,
    y: np.ndarray,
    xt: float,
    edge_n: int,
    edge_degree: int,
) -> float:
    n_edge = min(edge_n, len(x))
    if xt < x[0]:
        xs = x[:n_edge]
        ys = y[:n_edge]
    else:
        xs = x[-n_edge:]
        ys = y[-n_edge:]
    degree = min(edge_degree, len(xs) - 1)
    coeffs = np.polyfit(xs, ys, degree)
    return float(np.polyval(coeffs, xt))


def predict_peak_value(
    q: float,
    value_name: str,
    k: int,
    model: dict,
) -> float:
    table = model["peak_tables"][value_name].get(str(k))
    if table is None or len(table["q"]) < 2:
        return float("nan")

    q_nodes = np.asarray(table["q"], dtype=float)
    values = np.asarray(table["value"], dtype=float)
    x_nodes = q_coordinate_value(q_nodes, model["q_coordinate"])
    xt = float(q_coordinate_value(q, model["q_coordinate"]))
    order = np.argsort(x_nodes)
    x_nodes = x_nodes[order]
    values = values[order]

    if x_nodes[0] <= xt <= x_nodes[-1]:
        return float(PchipInterpolator(x_nodes, values, extrapolate=False)(xt))
    return extrapolate_edge(
        x_nodes,
        values,
        xt,
        int(model["edge_n"]),
        int(model["edge_degree"]),
    )


def target_peak_nodes(case: dict, args: argparse.Namespace, model: dict) -> dict:
    meta = case["meta"]
    bhpt_peaks = waveform_peaks(
        case["t_bhpt"],
        case["h_bhpt"],
        meta["t_bhpt_merger"],
        args.min_peak_time / BETA_BOUNDS[0],
        args.max_peak_time,
        args.prominence_fraction,
    )

    theta = []
    alpha = []
    beta = []
    used_k = []
    for peak in bhpt_peaks:
        k = int(peak["k_before_merger"])
        if k > int(model["max_k"]):
            continue
        alpha_k = predict_peak_value(case["q"], "alpha_peak", k, model)
        beta_k = predict_peak_value(case["q"], "beta_peak", k, model)
        if not (np.isfinite(alpha_k) and np.isfinite(beta_k)):
            continue
        if not (ALPHA_BOUNDS[0] <= alpha_k <= ALPHA_BOUNDS[1]):
            continue
        if not (BETA_BOUNDS[0] <= beta_k <= BETA_BOUNDS[1]):
            continue
        theta.append(float(meta["nu"] * peak["s"]))
        alpha.append(float(alpha_k))
        beta.append(float(beta_k))
        used_k.append(k)

    theta_arr = np.asarray(theta, dtype=float)
    order = np.argsort(theta_arr)
    return {
        "theta": theta_arr[order],
        "alpha": np.asarray(alpha, dtype=float)[order],
        "beta": np.asarray(beta, dtype=float)[order],
        "k": np.asarray(used_k, dtype=int)[order],
    }


def model_arrays(
    case: dict,
    model: dict,
    args: argparse.Namespace,
    phi0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    nodes = target_peak_nodes(case, args, model)
    if len(nodes["theta"]) < args.min_model_peaks:
        raise RuntimeError(
            f"Only {len(nodes['theta'])} usable model peaks for q={case['q']}"
        )

    alpha_interp = PchipInterpolator(nodes["theta"], nodes["alpha"], extrapolate=False)
    beta_interp = PchipInterpolator(nodes["theta"], nodes["beta"], extrapolate=False)

    meta = case["meta"]
    theta = meta["nu"] * (case["t_bhpt"] - meta["t_bhpt_merger"])
    theta_eval = np.clip(theta, nodes["theta"][0], nodes["theta"][-1])
    alpha = np.asarray(alpha_interp(theta_eval), dtype=float)
    beta = np.asarray(beta_interp(theta_eval), dtype=float)
    alpha = np.clip(alpha, *ALPHA_BOUNDS)
    beta = np.clip(beta, *BETA_BOUNDS)

    tau = meta["t_nr_merger"] + beta * (case["t_bhpt"] - meta["t_bhpt_merger"])
    h_scaled = alpha * np.exp(1j * phi0) * case["h_bhpt"]
    return tau, h_scaled, alpha, beta, theta, nodes


def evaluate_case(
    case: dict,
    model: dict,
    args: argparse.Namespace,
    min_coverage: float = STANDARD_EVAL_MIN_COVERAGE,
    include_arrays: bool = False,
) -> dict:
    try:
        tau, h_scaled, alpha, beta, theta, nodes = model_arrays(case, model, args)
    except RuntimeError as exc:
        return {"error": 50.0, "coverage": np.nan, "failure": str(exc)}

    use = (tau >= case["t_nr"][0]) & (tau <= case["t_nr"][-1])
    if np.count_nonzero(use) < 800:
        return {"error": 50.0, "coverage": np.nan, "failure": "too_few_samples"}
    tau_use = tau[use]
    if np.any(np.diff(tau_use) <= 0.0):
        return {"error": 50.0, "coverage": np.nan, "failure": "nonmonotonic_tau"}
    if np.any(alpha[use] <= 0.0) or np.any(beta[use] <= 0.0):
        return {"error": 50.0, "coverage": np.nan, "failure": "nonpositive_scale"}

    common = (case["t_nr"] >= tau_use[0]) & (case["t_nr"] <= tau_use[-1])
    coverage = np.count_nonzero(common) / len(case["t_nr"])
    if coverage < min_coverage:
        return {
            "error": 50.0 + (min_coverage - coverage),
            "coverage": float(coverage),
            "failure": "low_coverage",
        }

    h_model0 = interp_complex(tau_use, h_scaled[use], case["t_nr"][common])
    phi0 = principal_phase(
        float(np.angle(np.sum(case["h_nr"][common] * h_model0.conjugate())))
    )
    h_model = h_model0 * np.exp(1j * phi0)
    result = {
        "error": mathcalE_error(case["h_nr"][common], h_model),
        "coverage": float(coverage),
        "common_start": float(case["t_nr"][common][0]),
        "common_end": float(case["t_nr"][common][-1]),
        "phi0": float(phi0),
        "used_peak_count": int(len(nodes["theta"])),
        "min_k": int(np.min(nodes["k"])),
        "max_k": int(np.max(nodes["k"])),
    }
    if include_arrays:
        result.update(
            {
                "tau": tau_use,
                "alpha": alpha[use],
                "beta": beta[use],
                "theta": theta[use],
            }
        )
    return result


def summarize_errors(rows: list[dict]) -> dict[str, float]:
    errors = np.asarray([row["error"] for row in rows], dtype=float)
    return {
        "min": float(np.min(errors)),
        "median": float(np.median(errors)),
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
    }


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.8g}"


def write_markdown(
    peak_rows: list[dict],
    eval_rows: list[dict],
    extrap_rows: list[dict],
    model: dict,
    args: argparse.Namespace,
) -> None:
    summary = summarize_errors(eval_rows)
    q_at_max = max(eval_rows, key=lambda row: row["error"])["q"]
    q3_row = min(eval_rows, key=lambda row: abs(row["q"] - 3.0))
    q8_row = min(eval_rows, key=lambda row: abs(row["q"] - 8.0))
    peak_counts = {
        key: len(rows) for key, rows in model["peak_counts_by_q"].items()
    }

    lines = [
        "# Waveform peak-ratio q-dependent scaling",
        "",
        "This run follows the waveform-peak ratio construction suggested by",
        "Section II A 3 of arXiv:2307.03155. The calibration uses the raw",
        "BHPT `(2, 2)` mode and the nonspinning `NRHybSur3dq8` `(2, 2)` mode.",
        "For each q, both waveforms are rotated so the merger-amplitude peak",
        "has zero phase, and positive real-part peaks are matched by their",
        "index counted backward from merger.",
        "",
        "For every matched peak:",
        "",
        "```python",
        "nu = q / (1 + q)**2",
        "s_bhpt = t_bhpt_peak - t_bhpt_merger",
        "s_nr = t_nr_peak - t_nr_merger",
        "theta = nu * s_bhpt",
        "alpha_peak = h_nr_peak / h_bhpt_peak",
        "beta_peak = s_nr / s_bhpt",
        "```",
        "",
        "The final model is indexed by peak number rather than by a global",
        "polynomial in theta. For each peak index `k`, `alpha_peak(k, q)` and",
        "`beta_peak(k, q)` are represented as follows:",
        "",
        "```python",
        f"Q(q) = {model['q_coordinate']}",
        "inside 3 <= q <= 8: PCHIP interpolation in Q(q)",
        "outside the fitted interval: local edge polynomial extrapolation in Q(q)",
        f"edge_n = {model['edge_n']}",
        f"edge_degree = {model['edge_degree']}",
        f"max_k = {model['max_k']}",
        "```",
        "",
        "For a target q, the BHPT peak locations determine the target theta",
        "nodes. The predicted peak-index ratios are then interpolated in theta",
        "with PCHIP to form continuous arrays:",
        "",
        "```python",
        "alpha(q, theta) = PCHIP_theta({theta_k(q), alpha_peak(k, q)})",
        "beta(q, theta) = PCHIP_theta({theta_k(q), beta_peak(k, q)})",
        "tau(t_bhpt) = t_nr_merger + beta(q, theta) * (t_bhpt - t_bhpt_merger)",
        "h_model(tau) = alpha(q, theta) * exp(1j * phi0) * h_bhpt(t_bhpt)",
        "```",
        "",
        "The constant phase `phi0` is optimized analytically for each q",
        "evaluation and is not part of the fitted q-dependent model.",
        "",
        "## Fit Setup",
        "",
        f"- Calibration grid: `{args.n_q}` uniformly spaced q values from `3` to `8`.",
        f"- Total accepted peak-ratio samples: `{len(peak_rows)}`.",
        f"- Peak search window: `{args.min_peak_time:g} <= t - t_merger <= {args.max_peak_time:g}`.",
        f"- Peak prominence fraction: `{args.prominence_fraction:g}`.",
        f"- q coordinate for interpolation/extrapolation: `{model['q_coordinate']}`.",
        f"- Edge extrapolator: degree `{model['edge_degree']}` polynomial using `{model['edge_n']}` nearest q nodes.",
        f"- Maximum retained peak index: `{model['max_k']}`.",
        f"- Minimum target model peaks required: `{args.min_model_peaks}`.",
        "",
        "This is the best low-cost checkpoint from the current run, not a",
        "fully accepted q-dependent calibration. The quadratic edge",
        "extrapolator keeps q=2 below the requested 1% level, but it is not",
        "uniformly reliable over 2 < q < 3. A separate lower-edge scan showed",
        "that the more conservative linear edge rule behaves better at q=2.5",
        "but leaves q=2 slightly above 1%.",
        "",
        "## Error Summary",
        "",
        "| range | min mathcalE | median mathcalE | mean mathcalE | max mathcalE |",
        "|:---|---:|---:|---:|---:|",
        f"| 3 <= q <= 8 | {summary['min']:.8g} | {summary['median']:.8g} | {summary['mean']:.8g} | {summary['max']:.8g} |",
        "",
        f"Maximum in-range mathcalE occurs at q `{q_at_max:.8g}`.",
        f"Nearest grid point to q=3 has mathcalE `{q3_row['error']:.8g}`.",
        f"Nearest grid point to q=8 has mathcalE `{q8_row['error']:.8g}`.",
        "",
        "## Extrapolation Diagnostics",
        "",
        "These q values were not used in the 64-point q calibration grid.",
        "",
        "| q | mathcalE | coverage | phi0 | model peaks | common support | status |",
        "|---:|---:|---:|---:|---:|:---|:---|",
    ]
    for row in extrap_rows:
        status = row.get("failure", "ok")
        lines.append(
            f"| {row['q']:.8g} | {format_float(row['error'])} | "
            f"{format_float(row.get('coverage', np.nan))} | "
            f"{format_float(row.get('phi0', np.nan))} | "
            f"{row.get('used_peak_count', 0)} | "
            f"[{format_float(row.get('common_start', np.nan))}, "
            f"{format_float(row.get('common_end', np.nan))}] | {status} |"
        )

    lines.extend(
        [
            "",
            "## Per-q Calibration Results",
            "",
            "| q | mathcalE | coverage | phi0 | accepted peak samples | model peaks | common support |",
            "|---:|---:|---:|---:|---:|---:|:---|",
        ]
    )
    for row in eval_rows:
        key = q_key(row["q"])
        lines.append(
            f"| {row['q']:.8g} | {row['error']:.8g} | "
            f"{row['coverage']:.8g} | {row.get('phi0', np.nan):.8g} | "
            f"{peak_counts.get(key, 0)} | {row.get('used_peak_count', 0)} | "
            f"[{row.get('common_start', np.nan):.8g}, "
            f"{row.get('common_end', np.nan):.8g}] |"
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The q extrapolation is empirical. It is accurate at q=2 in this",
            "  diagnostic, but the same construction is not reliable across the",
            "  whole q < 3 interval. In particular, q=2.5 is a poor point for",
            "  this quadratic-edge checkpoint.",
            "- The largest in-range error is caused by a peak-matching pathology",
            "  near q=3.1587: the direct per-q peak-ratio reconstruction is",
            "  already bad there, so this is not only a q-interpolation error.",
            "- The q=1.5 failure should not be interpreted as a normal waveform",
            "  mismatch value; the extrapolated time map fails the model validity",
            "  checks before producing an acceptable comparison.",
            "- `beta_peak` is the peak-location ratio `s_nr_peak / s_bhpt_peak`.",
            "  The continuous time map uses this as a local multiplicative map,",
            "  not as a separately integrated `d tau / d t_bhpt`.",
            "- Values in theta outside the retained peak-node range are clipped",
            "  to the nearest endpoint before evaluating alpha and beta.",
            "- The complete peak-index q tables are stored in",
            f"  `{RESULTS_PATH.relative_to(ROOT)}` rather than expanded in this",
            "  markdown file.",
            "",
        ]
    )
    MD_PATH.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit q-dependent alpha/beta arrays from matched waveform peak ratios."
    )
    parser.add_argument("--n-q", type=int, default=64)
    parser.add_argument("--min-peak-time", type=float, default=-5000.0)
    parser.add_argument("--max-peak-time", type=float, default=-20.0)
    parser.add_argument("--prominence-fraction", type=float, default=0.01)
    parser.add_argument("--min-peaks-per-q", type=int, default=12)
    parser.add_argument("--min-model-peaks", type=int, default=6)
    parser.add_argument("--max-k", type=int, default=38)
    parser.add_argument(
        "--q-coordinate",
        choices=["q", "mass_norm", "delta", "nu", "invq"],
        default="mass_norm",
    )
    parser.add_argument("--edge-n", type=int, default=12)
    parser.add_argument("--edge-degree", type=int, default=2)
    parser.add_argument(
        "--extrap-q", type=float, nargs="+", default=[2.5, 2.0, 1.5, 10.0]
    )
    parser.add_argument("--force-peaks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_values = np.linspace(3.0, 8.0, args.n_q)
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    peak_cache = load_peak_cache()

    rows_by_q: dict[str, list[dict]] = {}
    cases: dict[str, dict] = {}
    for q in q_values:
        key = q_key(float(q))
        case = load_case(nrsur, float(q))
        cases[key] = case
        if not args.force_peaks and key in peak_cache:
            rows = peak_cache[key]["rows"]
            print(f"cached peaks q={q:.8g} n={len(rows)}", flush=True)
        else:
            rows = peak_ratio_rows(case, args)
            peak_cache[key] = {"q": float(q), "rows": rows}
            save_peak_cache(peak_cache)
            print(f"extracted peaks q={q:.8g} n={len(rows)}", flush=True)
        if len(rows) < args.min_peaks_per_q:
            raise RuntimeError(f"Only {len(rows)} accepted peaks for q={q:.8g}")
        rows_by_q[key] = rows

    peak_rows = [row for rows in rows_by_q.values() for row in rows]
    model = {
        "identifier": "wf_peak_ratios",
        "mode": list(MODE),
        "n_q": int(args.n_q),
        "q_min": 3.0,
        "q_max": 8.0,
        "q_values": [float(q) for q in q_values],
        "q_coordinate": args.q_coordinate,
        "edge_n": int(args.edge_n),
        "edge_degree": int(args.edge_degree),
        "max_k": int(args.max_k),
        "alpha_bounds": list(ALPHA_BOUNDS),
        "beta_bounds": list(BETA_BOUNDS),
        "peak_tables": build_peak_tables(rows_by_q, args.max_k),
        "peak_counts_by_q": {key: rows for key, rows in rows_by_q.items()},
    }
    model["peak_counts_by_q"] = {
        key: list(range(1, len(rows) + 1)) for key, rows in rows_by_q.items()
    }

    eval_rows = []
    for q in q_values:
        ev = evaluate_case(cases[q_key(float(q))], model, args)
        row = {
            "q": float(q),
            "error": float(ev["error"]),
            "coverage": float(ev.get("coverage", np.nan)),
            "phi0": float(ev.get("phi0", np.nan)),
            "common_start": float(ev.get("common_start", np.nan)),
            "common_end": float(ev.get("common_end", np.nan)),
            "used_peak_count": int(ev.get("used_peak_count", 0)),
            "failure": ev.get("failure", "ok"),
        }
        eval_rows.append(row)
        print(
            f"q={q:.8g} mathcalE={row['error']:.8g} peaks={row['used_peak_count']}",
            flush=True,
        )

    extrap_rows = []
    for q in args.extrap_q:
        case = load_case(nrsur, float(q))
        ev = evaluate_case(case, model, args)
        row = {
            "q": float(q),
            "error": float(ev["error"]),
            "coverage": float(ev.get("coverage", np.nan)),
            "phi0": float(ev.get("phi0", np.nan)),
            "common_start": float(ev.get("common_start", np.nan)),
            "common_end": float(ev.get("common_end", np.nan)),
            "used_peak_count": int(ev.get("used_peak_count", 0)),
            "failure": ev.get("failure", "ok"),
        }
        extrap_rows.append(row)
        print(
            f"extrap q={q:.8g} mathcalE={row['error']:.8g} status={row['failure']}",
            flush=True,
        )

    output = {
        "model": model,
        "eval_summary": summarize_errors(eval_rows),
        "eval_rows": eval_rows,
        "extrap_rows": extrap_rows,
        "args": vars(args),
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    write_markdown(peak_rows, eval_rows, extrap_rows, model, args)

    print(f"wrote {RESULTS_PATH}", flush=True)
    print(f"wrote {MD_PATH}", flush=True)


if __name__ == "__main__":
    main()
