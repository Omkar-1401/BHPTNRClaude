from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import fit_scaling_PN_opt_creative as creative


ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
PLOT_DIR = ROOT / "Agentic_plots" / "PN_opt_creative"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(UT_ROOT / "BHPTutils"))
import bhpt_utils  # noqa: E402

mpl.rcdefaults()


def save_waveform_plot(
    common_t: np.ndarray, h_nr: np.ndarray, h_bhpt: np.ndarray, t_switch_nr: float
) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    ax1.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax1.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax1.set_xlim(-1000, t_switch_nr)
    ax1.grid(True)
    ax1.set_xlabel("t/M")
    ax1.legend()

    ax2.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax2.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax2.set_xlim(t_switch_nr, 100)
    ax2.grid(True)
    ax2.set_xlabel("t/M")

    fig.suptitle("Creative smooth PN-loss fit (q = 5)")
    fig.tight_layout()

    path = PLOT_DIR / "PN_opt_creative_fit_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(tau: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(creative.Q, [creative.MODE])

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [2, 2]},
    )

    ax1.plot(tau, alpha, label="Creative PN-loss fit")
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

    ax2.plot(tau, beta, label="Creative PN-loss fit")
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

    fig.suptitle("Creative smooth PN-loss fit (q = 5)")
    fig.tight_layout()

    path = PLOT_DIR / "PN_opt_creative_fit_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_zoomed_plot(common_t: np.ndarray, h_nr: np.ndarray, h_bhpt: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax.plot(common_t, np.real(h_bhpt), label="BHPT 22")
    ax.set_xlim(-100, 100)
    ax.set_title("Creative smooth PN-loss fit (q = 5)")
    ax.grid(True)
    ax.set_xlabel("t/M")
    ax.legend()
    fig.tight_layout()

    path = PLOT_DIR / "PN_opt_creative_fit_zoomed.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    t_bhpt, h_bhpt, t_nr, h_nr = creative.load_waveforms()
    meta = creative.physical_metadata(t_bhpt, h_bhpt, t_nr, h_nr)
    losses = creative.pn_loss_coordinates(t_bhpt, h_bhpt, meta)

    params = creative.ACCEPTED_CREATIVE_PARAMS.copy()
    tau, _h_scaled, alpha, beta, _masks = creative.model_arrays(params, t_bhpt, h_bhpt, losses)
    eval_data = creative.evaluate_model(
        params, t_bhpt, h_bhpt, t_nr, h_nr, losses, min_coverage=0.88
    )
    p = creative.unpack(params)

    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    common_t = t_nr[eval_data["common"]]
    h_nr_common = eval_data["h_ref"]
    h_bhpt_common = eval_data["h_model"]
    t_switch_nr = float(np.interp(p["p0"], eval_data["p_loss"], eval_data["tau"]))

    paths = [
        save_waveform_plot(common_t, h_nr_common, h_bhpt_common, t_switch_nr),
        save_parameter_plot(tau[use], alpha[use], beta[use]),
        save_zoomed_plot(common_t, h_nr_common, h_bhpt_common),
    ]

    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"common_support=[{common_t[0]:.15g}, {common_t[-1]:.15g}]")
    print(f"coverage={eval_data['coverage']:.15g}")
    print(f"t_switch_nr={t_switch_nr:.15g}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
