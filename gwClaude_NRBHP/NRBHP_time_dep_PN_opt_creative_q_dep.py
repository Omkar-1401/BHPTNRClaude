from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTutils"))

import bhpt_utils  # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402
import fit_scaling_PN_opt_creative_q_dep as qdep  # noqa: E402

mpl.rcdefaults()

Q_INPUT = 2.0
MODEL_TAG = f"PN_opt_creative_q_dep_q{int(Q_INPUT)}"
TITLE = f"Creative PN-loss q-dep fit (q = {int(Q_INPUT)})"
PLOT_DIR = ROOT / "Agentic_plots" / "PN_opt_creative"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MD_PATH = ROOT / "scaling_PN_opt_creative_q_dep.md"


# ---------------------------------------------------------------------------
# Read coefficients from markdown
# ---------------------------------------------------------------------------

def read_markdown_coefficients(md_path: Path) -> dict[str, np.ndarray]:
    text = md_path.read_text()
    heading = "## Selected Master Coefficients"
    start = text.index(heading)
    lines = text[start:].splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            table_lines.append(s)
            in_table = True
        elif in_table:
            break
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c.replace(":", "").replace("-", "") == "" for c in cells):
            continue
        rows.append(cells)
    if not rows:
        raise ValueError(f"No coefficient table found under '{heading}' in {md_path}")
    coeffs: dict[str, np.ndarray] = {}
    for row in rows[1:]:
        coeffs[row[0]] = np.array([float(v) for v in row[1:]], dtype=float)
    missing = [n for n in qdep.PARAM_NAMES if n not in coeffs]
    if missing:
        raise ValueError(f"Missing rows in markdown table: {missing}")
    return {n: coeffs[n] for n in qdep.PARAM_NAMES}


# ---------------------------------------------------------------------------
# Build model at Q_INPUT
# ---------------------------------------------------------------------------

def build_model() -> tuple[dict[str, np.ndarray], dict, np.ndarray, dict, tuple]:
    coeffs = read_markdown_coefficients(MD_PATH)
    qdep.generate_and_cache_waveform(Q_INPUT)  # no-op if already cached
    case = qdep.load_case(Q_INPUT, source_stride=3, nr_stride=5)
    params = qdep.constrained_master_params(Q_INPUT, coeffs)
    params, _fit_err = qdep.polish_nuisance(params, case)
    eval_data = creative.evaluate_model(
        params,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"],
        case["losses"], min_coverage=qdep.MIN_COVERAGE,
    )
    arrays = creative.model_arrays(params, case["t_bhpt"], case["h_bhpt"], case["losses"])
    return coeffs, case, params, eval_data, arrays


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_waveform_plot(
    common_t: np.ndarray, h_nr: np.ndarray, h_bhpt: np.ndarray,
    t_switch: float, error: float,
) -> Path:
    split = float(np.clip(t_switch, -900.0, -50.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [3, 2]})
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
    fig.suptitle(f"{TITLE};  mathcalE = {error:.3g}")
    fig.tight_layout()
    path = PLOT_DIR / f"{MODEL_TAG}_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(tau: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(Q_INPUT, [creative.MODE])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [2, 2]})
    ax1.plot(tau, alpha, label=TITLE)
    ax1.axhline(ab["alpha_l2m2"], color="k", linestyle="dashed", label="BHPTNRSurrogate")
    ax1.set_xlim(-1000, 100)
    ax1.grid(True)
    ax1.set_xlabel("t/M")
    ax1.set_ylabel(r"$\alpha(t)$")
    ax1.legend()
    ax2.plot(tau, beta, label=TITLE)
    ax2.axhline(ab["beta"], color="k", linestyle="dashed", label="BHPTNRSurrogate")
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
    columns = ["parameter", *[f"c{i}" for i in range(degree + 1)], f"value q={int(Q_INPUT)}"]
    table_rows = []
    for name in qdep.PARAM_NAMES:
        val_at_q = qdep.eval_poly(Q_INPUT, coeffs[name])
        table_rows.append([name, *[f"{v:.5g}" for v in coeffs[name]], f"{val_at_q:.5g}"])
    fig_h = max(4.0, 0.36 * len(table_rows) + 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=columns,
                   loc="center", cellLoc="right", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    ax.set_title(f"{TITLE}: 1/q polynomial coefficient table")
    fig.tight_layout()
    path = PLOT_DIR / f"{MODEL_TAG}_coefficients.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    coeffs, case, params, eval_data, arrays = build_model()
    tau, _h_scaled, alpha, beta, _masks = arrays
    use = (tau >= case["t_nr"][0]) & (tau <= case["t_nr"][-1])
    common_t = case["t_nr"][eval_data["common"]]
    p = creative.unpack(params)
    t_switch_nr = float(np.interp(p["p0"], eval_data["p_loss"], eval_data["tau"]))

    paths = [
        save_waveform_plot(common_t, eval_data["h_ref"], eval_data["h_model"],
                           t_switch_nr, eval_data["error"]),
        save_parameter_plot(tau[use], alpha[use], beta[use]),
        save_zoomed_plot(common_t, eval_data["h_ref"], eval_data["h_model"]),
        save_coefficient_table(coeffs),
    ]

    print(f"q={Q_INPUT}")
    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"common_support=[{common_t[0]:.15g}, {common_t[-1]:.15g}]")
    print(f"coverage={eval_data['coverage']:.15g}")
    print(f"t_switch_nr={t_switch_nr:.15g}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
