from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import fit_scaling_PN as pnfit


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PLOT_DIR = ROOT / "Notebook_plots"
PLOT_DIR.mkdir(exist_ok=True)

sys.path.append(str(REPO_ROOT / "BHPTutils"))
import bhpt_utils  # noqa: E402

mpl.rcdefaults()


PN_PARAMS = np.array(
    [
        -1.049129116009673e00,
        -9.971550514754898e01,
        8.030470800477922e-01,
        -1.861200358580629e-02,
        4.650021494793548e-03,
        8.128584306270605e-01,
        -3.301182718867643e-02,
        5.523103049275632e-02,
        8.140831180521366e-01,
        -5.073169349277392e-03,
        5.120477269831416e-03,
        8.315690350856837e-01,
        2.373765252714617e-01,
        -3.460310858141835e-01,
        1.372921745805103e00,
    ],
    dtype=float,
)


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

    fig.suptitle("PN-loss Agentic fit (q = 5)")
    fig.tight_layout()

    path = PLOT_DIR / "PN_agentic_fit_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(tau: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(pnfit.Q, [pnfit.MODE])

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [2, 2]},
    )

    ax1.plot(tau, alpha, label="PN-loss agentic")
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

    ax2.plot(tau, beta, label="PN-loss agentic")
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

    fig.suptitle("PN-loss Agentic fit (q = 5)")
    fig.tight_layout()

    path = PLOT_DIR / "PN_agentic_fit_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_zoomed_plot(common_t: np.ndarray, h_nr: np.ndarray, h_bhpt: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax.set_xlim(-100, 100)
    ax.set_title("PN-loss Agentic fit (q = 5)")
    ax.grid(True)
    ax.set_xlabel("t/M")
    ax.legend()
    fig.tight_layout()

    path = PLOT_DIR / "PN_agentic_fit_zoomed.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    t_bhpt, h_bhpt, t_nr, h_nr = pnfit.load_waveforms()
    meta = pnfit.physical_metadata(t_bhpt, h_bhpt, t_nr, h_nr)
    losses = pnfit.pn_loss_coordinates(t_bhpt, h_bhpt, meta)

    tau, h_scaled, alpha, beta, _ = pnfit.model_arrays(PN_PARAMS, t_bhpt, h_bhpt, losses)
    eval_data = pnfit.evaluate_model(PN_PARAMS, t_bhpt, h_bhpt, t_nr, h_nr, losses)
    p = pnfit.unpack(PN_PARAMS)

    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    common_t = t_nr[eval_data["common"]]
    h_nr_common = eval_data["h_ref"]
    h_bhpt_common = eval_data["h_model"]

    paths = [
        save_waveform_plot(common_t, h_nr_common, h_bhpt_common, p["t_cut_nr"]),
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
