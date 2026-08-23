from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PLOT_DIR = ROOT / "Notebook_plots/Smooth_q_dep"
PLOT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(REPO_ROOT / "BHPTNRSurrogate" / "surrogates"))
sys.path.insert(0, str(REPO_ROOT / "BHPTutils"))

import bhpt_utils  # noqa: E402
import fit_scaling_physical_smooth_qdep as smoothq  # noqa: E402
import gwsurrogate  # noqa: E402


mpl.rcdefaults()

Q_INPUT = 8.0
MODEL_TAG = "physical_smooth_qdep_q8"
TITLE = "Physical-smooth q-dependent Agentic fit (q = 8)"
MD_PATH = ROOT / "scaling_physical_smooth_qdep.md"


def q_label(q: float) -> str:
    return f"{q:g}"


def q_tag(q: float) -> str:
    return q_label(q).replace(".", "p").replace("-", "m")


def configure_plot(q_input: float) -> None:
    global Q_INPUT, MODEL_TAG, TITLE
    Q_INPUT = float(q_input)
    MODEL_TAG = f"physical_smooth_qdep_q{q_tag(Q_INPUT)}"
    TITLE = f"Physical-smooth q-dependent Agentic fit (q = {q_label(Q_INPUT)})"


def read_markdown_coefficients(md_path: Path, heading: str, names: list[str]) -> dict[str, np.ndarray]:
    text = md_path.read_text()
    start = text.index(heading)
    lines = text[start:].splitlines()

    table_lines: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
            in_table = True
        elif in_table:
            break

    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(cell.replace(":", "").replace("-", "") == "" for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        raise ValueError(f"No coefficient table found in {md_path}")

    coeffs: dict[str, np.ndarray] = {}
    for row in rows[1:]:
        coeffs[row[0]] = np.asarray([float(value) for value in row[1:]], dtype=float)

    missing = [name for name in names if name not in coeffs]
    if missing:
        raise ValueError(f"Missing coefficient rows in {md_path}: {missing}")
    return {name: coeffs[name] for name in names}


def eval_poly(q: float, coeffs: np.ndarray) -> float:
    y = 1.0 / q
    return float(sum(value * y**power for power, value in enumerate(coeffs)))


def diagnostic_evaluate_model(params: np.ndarray, case: dict) -> dict:
    tau, h_scaled, alpha, beta, _theta = smoothq.model_arrays(
        params, case["t_bhpt"], case["h_bhpt"], case["meta"]
    )
    use = (tau >= case["t_nr"][0]) & (tau <= case["t_nr"][-1])
    if np.count_nonzero(use) < 2:
        raise RuntimeError(f"No diagnostic source support for {TITLE}")

    tau_use = tau[use]
    if np.any(np.diff(tau_use) <= 0.0):
        raise RuntimeError(f"Non-monotonic diagnostic time map for {TITLE}")
    if np.any(alpha[use] <= 0.0) or np.any(beta[use] <= 0.0):
        raise RuntimeError(f"Non-positive diagnostic scaling for {TITLE}")

    common = (case["t_nr"] >= tau_use[0]) & (case["t_nr"] <= tau_use[-1])
    if np.count_nonzero(common) < 2:
        raise RuntimeError(f"No diagnostic common support for {TITLE}")

    h_model = smoothq.interp_complex(tau_use, h_scaled[use], case["t_nr"][common])
    return {
        "error": smoothq.mathcalE_error(case["h_nr"][common], h_model),
        "coverage": np.count_nonzero(common) / len(case["t_nr"]),
        "common": common,
        "h_ref": case["h_nr"][common],
        "h_model": h_model,
        "diagnostic": True,
    }


def build_model() -> tuple[dict[str, np.ndarray], dict, np.ndarray, dict, tuple]:
    coeffs = read_markdown_coefficients(
        MD_PATH, "## Selected Master Coefficients", smoothq.PARAM_NAMES
    )
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    case = smoothq.load_case(nrsur, Q_INPUT, source_stride=3, nr_stride=8)
    params = smoothq.master_params(Q_INPUT, coeffs)
    params, _fit_error = smoothq.polish_time_phase(params, case)
    eval_data = smoothq.evaluate_case(params, case, subset="full", min_coverage=0.84)
    if "common" not in eval_data:
        eval_data = smoothq.evaluate_case(params, case, subset="full", min_coverage=0.0)
    if "common" not in eval_data:
        eval_data = diagnostic_evaluate_model(params, case)
    arrays = smoothq.model_arrays(params, case["t_bhpt"], case["h_bhpt"], case["meta"])
    return coeffs, case, params, eval_data, arrays


def t_cut_nr(params: np.ndarray, meta: dict) -> float:
    p = smoothq.unpack(params)
    return float(meta["t_nr_merger"] + p["Theta_cut_nr"] / meta["nu"])


def save_waveform_plot(common_t: np.ndarray, h_nr: np.ndarray, h_bhpt: np.ndarray, split_t: float, error: float) -> Path:
    split = -250.0
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    ax1.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax1.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax1.set_xlim(-1000, split)
    ax1.grid(True)
    ax1.set_xlabel("t/M")
    ax1.legend()

    ax2.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax2.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax2.set_xlim(split, 100)
    ax2.grid(True)
    ax2.set_xlabel("t/M")

    fig.suptitle(f"{TITLE}; mathcalE = {error:.3g}")
    fig.tight_layout()
    path = PLOT_DIR / f"{MODEL_TAG}_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(tau: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(Q_INPUT, [smoothq.MODE])
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [2, 2]},
    )

    ax1.plot(tau, alpha, label="Physical-smooth q-dependent")
    ax1.plot(
        tau,
        np.ones_like(tau) * ab["alpha_l2m2"],
        label="BHPTNRSurrogate",
        linestyle="dashed",
        color="k",
    )
    ax1.set_xlim(-1000, 100)
    ax1.grid(True)
    ax1.set_xlabel("t/M")
    ax1.set_ylabel(r"$\alpha(t)$")
    ax1.legend()

    ax2.plot(tau, beta, label="Physical-smooth q-dependent")
    ax2.plot(
        tau,
        np.ones_like(tau) * ab["beta"],
        label="BHPTNRSurrogate",
        linestyle="dashed",
        color="k",
    )
    ax2.set_xlim(-1000, 100)
    ax2.grid(True)
    ax2.set_xlabel("t/M")
    ax2.set_ylabel(r"$\beta(t)$")

    fig.suptitle(TITLE)
    fig.tight_layout()
    path = PLOT_DIR / f"{MODEL_TAG}_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_zoomed_plot(common_t: np.ndarray, h_nr: np.ndarray, h_bhpt: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax.set_xlim(-100, 100)
    ax.set_title(TITLE)
    ax.grid(True)
    ax.set_xlabel("t/M")
    ax.legend()
    fig.tight_layout()
    path = PLOT_DIR / f"{MODEL_TAG}_zoomed.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_coefficient_table(coeffs: dict[str, np.ndarray]) -> Path:
    degree = len(next(iter(coeffs.values()))) - 1
    columns = ["coefficient", *[f"c{i}" for i in range(degree + 1)], f"value q={q_label(Q_INPUT)}"]
    rows = []
    for name in smoothq.PARAM_NAMES:
        values = coeffs[name]
        rows.append(
            [name, *[f"{value:.5g}" for value in values], f"{eval_poly(Q_INPUT, values):.5g}"]
        )

    fig_height = max(4.0, 0.36 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="right",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    ax.set_title(f"{TITLE}: 1/q coefficient table")
    fig.tight_layout()
    path = PLOT_DIR / f"{MODEL_TAG}_coefficients.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the physical-smooth q-dependent calibration at a requested mass ratio."
    )
    parser.add_argument(
        "--q-input",
        type=float,
        default=Q_INPUT,
        help="Mass ratio to evaluate. Values outside [3, 8] are extrapolations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot(args.q_input)
    coeffs, case, params, eval_data, arrays = build_model()
    tau, _h_scaled, alpha, beta, _theta = arrays
    use = (tau >= case["t_nr"][0]) & (tau <= case["t_nr"][-1])
    common_t = case["t_nr"][eval_data["common"]]

    paths = [
        save_waveform_plot(common_t, eval_data["h_ref"], eval_data["h_model"], t_cut_nr(params, case["meta"]), eval_data["error"]),
        save_parameter_plot(tau[use], alpha[use], beta[use]),
        save_zoomed_plot(common_t, eval_data["h_ref"], eval_data["h_model"]),
        save_coefficient_table(coeffs),
    ]

    print(f"q={Q_INPUT:.8g}")
    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"common_support=[{common_t[0]:.15g}, {common_t[-1]:.15g}]")
    print(f"coverage={eval_data['coverage']:.15g}")
    print(f"diagnostic_eval={bool(eval_data.get('diagnostic', False))}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
