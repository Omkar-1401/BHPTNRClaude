from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import fit_scaling_PN as base
import fit_scaling_PN_opt as pnopt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PLOT_DIR = ROOT / "Notebook_plots"
PLOT_DIR.mkdir(exist_ok=True)

sys.path.append(str(REPO_ROOT / "BHPTutils"))
import bhpt_utils  # noqa: E402

mpl.rcdefaults()


def save_waveform_plot(common_t: np.ndarray, h_nr: np.ndarray, h_bhpt: np.ndarray, t_cut_nr: float) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    ax1.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax1.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax1.set_xlim(-1000, t_cut_nr)
    ax1.grid(True)
    ax1.set_xlabel("t/M")
    ax1.legend()

    ax2.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax2.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax2.set_xlim(t_cut_nr, 100)
    ax2.grid(True)
    ax2.set_xlabel("t/M")

    fig.suptitle("Smooth optimized PN-loss Agentic fit (q = 5)")
    fig.tight_layout()

    path = PLOT_DIR / "PN_opt_agentic_fit_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(tau: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(base.Q, [base.MODE])

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [2, 2]},
    )

    ax1.plot(tau, alpha, label="Smooth PN-loss opt agentic")
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

    ax2.plot(tau, beta, label="Smooth PN-loss opt agentic")
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

    fig.suptitle("Smooth optimized PN-loss Agentic fit (q = 5)")
    fig.tight_layout()

    path = PLOT_DIR / "PN_opt_agentic_fit_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_zoomed_plot(common_t: np.ndarray, h_nr: np.ndarray, h_bhpt: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax.set_xlim(-100, 100)
    ax.set_title("Smooth optimized PN-loss Agentic fit (q = 5)")
    ax.grid(True)
    ax.set_xlabel("t/M")
    ax.legend()
    fig.tight_layout()

    path = PLOT_DIR / "PN_opt_agentic_fit_zoomed.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    t_bhpt, h_bhpt, t_nr, h_nr = base.load_waveforms()
    meta = base.physical_metadata(t_bhpt, h_bhpt, t_nr, h_nr)
    losses = base.pn_loss_coordinates(t_bhpt, h_bhpt, meta)

    params = pnopt.ACCEPTED_PN_OPT_PARAMS.copy()
    tau, _h_scaled, alpha, beta, _ = pnopt.model_arrays(params, t_bhpt, h_bhpt, losses)
    eval_data = pnopt.evaluate_model(params, t_bhpt, h_bhpt, t_nr, h_nr, losses, min_coverage=0.88)
    p = pnopt.unpack(params)

    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    common_t = t_nr[eval_data["common"]]
    h_nr_common = eval_data["h_ref"]
    h_bhpt_common = eval_data["h_model"]

    paths = [
        save_waveform_plot(common_t, h_nr_common, h_bhpt_common, p["t_start_nr"]),
        save_parameter_plot(tau[use], alpha[use], beta[use]),
        save_zoomed_plot(common_t, h_nr_common, h_bhpt_common),
    ]

    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"common_support=[{common_t[0]:.15g}, {common_t[-1]:.15g}]")
    print(f"coverage={eval_data['coverage']:.15g}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
