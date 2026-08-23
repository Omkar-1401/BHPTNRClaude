"""
Plotting script for the full-PN q-dependent model (no logistic switch).

Generates diagnostic plots for Q_INPUTS in Agentic_plots/full_PN_q_dep/.
q = 2 is an extrapolation test (outside training range [3, 8]).

Run fit_scaling_PN_opt_full_PN_q_dep.py first to generate the coefficient
markdown and per-q cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTutils"))

import bhpt_utils
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_PN_opt_full_PN_q_dep as fpn

mpl.rcdefaults()

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
MD_PATH  = ROOT / "scaling_PN_opt_full_PN_q_dep.md"
PLOT_DIR = ROOT / "Agentic_plots" / "full_PN_q_dep"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Coefficient loading
# ---------------------------------------------------------------------------

def read_markdown_coefficients(md_path: Path) -> dict[str, np.ndarray]:
    text = md_path.read_text()
    heading = "## Selected Master Coefficients"
    start = text.index(heading)
    lines = text[start:].splitlines()
    table_lines, in_table = [], False
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
        raise ValueError(f"No coefficient table found in {md_path}")
    coeffs: dict[str, np.ndarray] = {}
    for row in rows[1:]:
        name = row[0].strip()
        coeffs[name] = np.array([float(v) for v in row[1:]], dtype=float)
    missing = [n for n in fpn.PARAM_NAMES if n not in coeffs]
    if missing:
        raise ValueError(f"Missing coefficient rows: {missing}")
    return {n: coeffs[n] for n in fpn.PARAM_NAMES}


# ---------------------------------------------------------------------------
# Build model for a given q
# ---------------------------------------------------------------------------

def build_model(q: float, coeffs: dict[str, np.ndarray]) -> dict:
    qdep.generate_and_cache_waveform(q)
    case = qdep.load_case(q, source_stride=3, nr_stride=5)
    params = fpn.constrained_master_params(q, coeffs)
    params, _ = fpn.polish_nuisance(params, case)
    ev = fpn.evaluate_model(
        params,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"],
        case["losses"], min_coverage=qdep.MIN_COVERAGE,
    )
    return {"eval": ev, "params": params, "case": case}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _extrap(q: float) -> bool:
    return q < 3.0 or q > 8.0


def _tag(q: float) -> str:
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if _extrap(q) else "")


def _title(q: float, error: float = float("nan")) -> str:
    chi_f_val = fpn.get_chi_f(q)
    extrap = " [extrapolation]" if _extrap(q) else ""
    err_str = f";  mathcalE = {error:.3g}" if np.isfinite(error) else ""
    return f"Full-PN  q={q}{extrap};  chi_f={chi_f_val:.3f}{err_str}"


def save_waveform_plot(q: float, common_t: np.ndarray,
                       h_nr: np.ndarray, h_model: np.ndarray,
                       error: float) -> Path:
    split = -200.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [3, 2]})
    for ax, xlim in [(ax1, (-1000, split)), (ax2, (split, 100))]:
        ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
        ax.plot(common_t, np.real(h_model), label="BHPT calibrated", alpha=0.8)
        ax.set_xlim(*xlim)
        ax.grid(True)
        ax.set_xlabel("t/M")
    ax1.legend(fontsize=8)
    fig.suptitle(_title(q, error))
    fig.tight_layout()
    path = PLOT_DIR / f"full_PN_{_tag(q)}_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(q: float, tau_use: np.ndarray,
                        alpha_use: np.ndarray, beta_use: np.ndarray,
                        e_hat_use: np.ndarray, j_hat_use: np.ndarray) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(q, [creative.MODE])
    chi_f_val = fpn.get_chi_f(q)
    label = f"full-PN  chi_f={chi_f_val:.3f}"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(tau_use, alpha_use, label=label)
    axes[0].axhline(ab["alpha_l2m2"], color="k", ls="--", label="BHPTNRSurrogate α")
    axes[0].set_xlim(-1000, 100); axes[0].grid(True)
    axes[0].set_xlabel("t_NR / M"); axes[0].set_ylabel(r"$\alpha(t)$")
    axes[0].legend(fontsize=7)

    axes[1].plot(tau_use, beta_use, label=label)
    axes[1].axhline(ab["beta"], color="k", ls="--", label="BHPTNRSurrogate β")
    axes[1].set_xlim(-1000, 100); axes[1].grid(True)
    axes[1].set_xlabel("t_NR / M"); axes[1].set_ylabel(r"$\beta(t)$")
    axes[1].legend(fontsize=7)

    # alpha vs Ehat — shows the PN amplitude coupling in action
    sort_idx = np.argsort(e_hat_use)
    axes[2].plot(e_hat_use[sort_idx], alpha_use[sort_idx], lw=1, label=label)
    axes[2].set_xlabel(r"$\hat{E}$"); axes[2].set_ylabel(r"$\alpha$")
    axes[2].set_title(r"$\alpha$ vs $\hat{E}$ (PN coupling)")
    axes[2].grid(True)

    fig.suptitle(_title(q))
    fig.tight_layout()
    path = PLOT_DIR / f"full_PN_{_tag(q)}_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_zoomed_plot(q: float, common_t: np.ndarray,
                     h_nr: np.ndarray, h_model: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax.plot(common_t, np.real(h_model), label="BHPT calibrated", alpha=0.8)
    ax.set_xlim(-100, 100); ax.grid(True)
    ax.set_xlabel("t/M"); ax.legend(fontsize=8)
    ax.set_title(_title(q))
    fig.tight_layout()
    path = PLOT_DIR / f"full_PN_{_tag(q)}_zoomed.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_coefficient_table(coeffs: dict[str, np.ndarray]) -> Path:
    degree = len(next(iter(coeffs.values()))) - 1
    columns = ["parameter", *[f"c{i}" for i in range(degree + 1)], "value q=5"]
    table_rows = []
    for name in fpn.PARAM_NAMES:
        val = fpn.eval_poly(5.0, coeffs[name])
        table_rows.append([name, *[f"{v:.5g}" for v in coeffs[name]], f"{val:.5g}"])
    fig_h = max(4.0, 0.4 * len(table_rows) + 1.2)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=columns,
                   loc="center", cellLoc="right", colLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.3)
    ax.set_title("Full-PN: polynomial coefficients (chi_f basis)")
    fig.tight_layout()
    path = PLOT_DIR / "full_PN_coefficients.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_coupling_vs_q(coeffs: dict[str, np.ndarray]) -> Path:
    """PN coupling parameters (alpha_E, alpha_J, beta_E, beta_J, beta_L) vs q."""
    q_fine = np.linspace(1.5, 9.0, 250)
    coupling_names = ["alpha_E", "alpha_J", "beta_E", "beta_J", "beta_L"]

    fig, axes = plt.subplots(1, len(coupling_names), figsize=(17, 4))
    for ax, name in zip(axes, coupling_names):
        vals = [fpn.eval_poly(float(q), coeffs[name]) for q in q_fine]
        ax.plot(q_fine, vals, lw=1.5)
        ax.axvline(3.0, color="gray", ls="--", lw=0.8)
        ax.axvline(8.0, color="gray", ls="--", lw=0.8)
        ax.axhline(0.0, color="k",    ls=":",  lw=0.6, alpha=0.5)
        ax.set_xlabel("q"); ax.set_ylabel(name, fontsize=9)
        ax.set_title(f"{name}(q)")
        ax.grid(True)

    axes[0].annotate("training\nboundary", xy=(3.0, axes[0].get_ylim()[0]),
                     xytext=(3.3, axes[0].get_ylim()[0]), fontsize=7, color="gray")
    fig.suptitle("Full-PN PN-coupling coefficients vs q  (gray dashed = training boundary)")
    fig.tight_layout()
    path = PLOT_DIR / "full_PN_couplings_q.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_alpha_beta_vs_q(coeffs: dict[str, np.ndarray]) -> Path:
    """alpha_i and beta_i (value at merger) vs q."""
    q_fine = np.linspace(1.5, 9.0, 250)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name, ylabel in [
        (ax1, "alpha_i", r"$\alpha_i$ (amplitude at merger)"),
        (ax2, "beta_i",  r"$\beta_i$ (time-stretch at merger)"),
    ]:
        vals = [fpn.eval_poly(float(q), coeffs[name]) for q in q_fine]
        ax.plot(q_fine, vals, lw=1.5)
        ax.axvline(3.0, color="gray", ls="--", lw=0.8, label="training boundary")
        ax.axvline(8.0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("q"); ax.set_ylabel(ylabel)
        ax.set_title(f"{name}(q)")
        ax.grid(True); ax.legend(fontsize=7)

    fig.suptitle("Full-PN: amplitude/time-stretch at merger vs q")
    fig.tight_layout()
    path = PLOT_DIR / "full_PN_alpha_beta_i_q.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    coeffs = read_markdown_coefficients(MD_PATH)
    all_paths: list[Path] = []
    errors: dict[float, float] = {}

    for q in Q_INPUTS:
        extrap_str = " [EXTRAPOLATION]" if _extrap(q) else ""
        print(f"\n--- q={q}{extrap_str} ---", flush=True)

        result = build_model(q, coeffs)
        ev = result["eval"]
        params = result["params"]
        case = result["case"]

        p = fpn.unpack(params)
        chi_f_val = fpn.get_chi_f(q)
        errors[q] = ev["error"]

        print(f"  chi_f    = {chi_f_val:.4f}")
        for name in fpn.PARAM_NAMES:
            print(f"  {name:<10} = {p[name]:.6g}")
        print(f"  mathcalE = {ev['error']:.6g}")

        # Arrays masked to NR window
        tau_use   = ev["tau"]
        alpha_use = ev["alpha"]
        beta_use  = ev["beta"]
        e_hat_use = ev["e_hat"]
        j_hat_use = ev["j_hat"]
        common_t  = case["t_nr"][ev["common"]]

        all_paths += [
            save_waveform_plot(q, common_t, ev["h_ref"], ev["h_model"], ev["error"]),
            save_parameter_plot(q, tau_use, alpha_use, beta_use, e_hat_use, j_hat_use),
            save_zoomed_plot(q, common_t, ev["h_ref"], ev["h_model"]),
        ]

    all_paths += [
        save_coefficient_table(coeffs),
        save_coupling_vs_q(coeffs),
        save_alpha_beta_vs_q(coeffs),
    ]

    print("\n=== Summary ===")
    for q in Q_INPUTS:
        flag = " [extrap]" if _extrap(q) else "        "
        chi_f_val = fpn.get_chi_f(q)
        print(f"  q={q:<4}{flag}  chi_f={chi_f_val:.4f}  mathcalE={errors[q]:.4e}")

    print(f"\nPlots: {PLOT_DIR}")
    for path in all_paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
