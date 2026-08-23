"""
Plots for the q_dep_classic constant alpha-beta model at q = 2, 3, 5.

Generates per-q:
  - waveform comparison  (2-panel: inspiral + merger/ringdown)
  - parameter plot       (constant alpha and beta vs BHPTNRSurrogate)
  - zoomed merger plot   (t in [-100, 100])

Plus global:
  - alpha and beta polynomial vs q  (training range shaded, Islam et al. overlay)
  - coefficient table               (values at q = 2, 3, 5, 8)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcdefaults()

ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTutils"))

import bhpt_utils
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_PN_opt_q_dep_classic as classic

# bhpt_utils sets text.usetex=True on import; reset it so we use matplotlib mathtext
mpl.rcParams["text.usetex"] = False

PLOT_DIR = ROOT / "Agentic_plots" / "classic_q_dep"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MD_PATH  = ROOT / "scaling_PN_opt_q_dep_classic.md"

Q_VALUES = [2, 3, 5]

# Islam et al. (2204.01972) reference polynomial coefficients (SXS-calibrated)
_ISLAM_ALPHA = np.array([1.0, -1.3301744361722947, 2.7201499322832317,
                         -5.9043496190355045, 5.548924434989228])
_ISLAM_BETA  = np.array([1.0, -1.2384720748177163, 1.5967739744828955,
                         -1.776560588418261,  1.0577827906015924])


# ---------------------------------------------------------------------------
# Coefficient loading
# ---------------------------------------------------------------------------

def read_coefficients() -> dict[str, np.ndarray]:
    text  = MD_PATH.read_text()
    start = text.index("## Selected Master Coefficients")
    lines = text[start:].splitlines()
    table: list[str] = []
    in_tbl = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            table.append(s)
            in_tbl = True
        elif in_tbl:
            break
    rows = []
    for ln in table:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(c.replace(":", "").replace("-", "") == "" for c in cells):
            continue
        rows.append(cells)
    if not rows:
        raise RuntimeError(f"No coefficient table found under '## Selected Master Coefficients' in {MD_PATH}")
    coeffs: dict[str, np.ndarray] = {}
    for row in rows[1:]:
        coeffs[row[0]] = np.array([float(v) for v in row[1:]])
    for name in classic.PHYS_NAMES:
        if name not in coeffs:
            raise RuntimeError(f"Missing '{name}' in markdown coefficient table.")
    return {name: coeffs[name] for name in classic.PHYS_NAMES}


# ---------------------------------------------------------------------------
# Build model at q
# ---------------------------------------------------------------------------

def build_model(q: float, coeffs: dict[str, np.ndarray]) -> dict:
    alpha = float(np.clip(classic.eval_poly(q, coeffs["alpha"]), 0.05, 2.5))
    beta  = float(np.clip(classic.eval_poly(q, coeffs["beta"]),  0.20, 1.6))
    qdep.generate_and_cache_waveform(q)
    case  = classic.load_case(q)
    t0_nr, phi0, _ = classic.polish_nuisance(alpha, beta, case)
    ev = classic.evaluate_model(
        alpha, beta, t0_nr, phi0,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"],   case["h_nr"],
    )
    return {
        "q":        q,
        "alpha":    alpha,
        "beta":     beta,
        "t0_nr":    t0_nr,
        "phi0":     phi0,
        "error":    ev["error"],
        "coverage": ev.get("coverage", float("nan")),
        "h_ref":    ev["h_ref"],
        "h_model":  ev["h_model"],
        "tau":      ev["tau"],
        "common":   ev["common"],
        "t_nr":     case["t_nr"],
    }


# ---------------------------------------------------------------------------
# Per-q plots
# ---------------------------------------------------------------------------

def save_waveform_plot(m: dict) -> Path:
    q     = m["q"]
    tag   = f"q{q:.4g}".replace(".", "p")
    t     = m["t_nr"][m["common"]]
    title = f"Classic const α-β, q = {q:.4g};  𝓔 = {m['error']:.3g}"

    # Choose split point: ~quarter of the window, clamped to [-2000, -50]
    split = float(np.clip(t[len(t) // 4], -2000.0, -50.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [3, 2]})
    for ax, xlim in [(ax1, (-1000, split)), (ax2, (split, 100))]:
        ax.plot(t, m["h_ref"].real,   label="NRHybSur3dq8")
        ax.plot(t, m["h_model"].real, label="BHPT calibrated", linestyle="--")
        ax.set_xlim(xlim)
        ax.grid(True)
        ax.set_xlabel("t/M")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path = PLOT_DIR / f"classic_q_dep_{tag}_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_param_plot(m: dict, coeffs: dict[str, np.ndarray]) -> Path:
    q   = m["q"]
    tag = f"q{q:.4g}".replace(".", "p")
    ab  = bhpt_utils.all_alpha_beta_1D_scaling_params(q, [(2, 2)])

    # Islam et al. predicted values at this q (for reference)
    alpha_islam = classic.eval_poly(q, _ISLAM_ALPHA)
    beta_islam  = classic.eval_poly(q, _ISLAM_BETA)

    t_range = [-1000, 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Alpha panel
    ax1.axhline(m["alpha"],        color="C0",  lw=2,   label=f"q_dep_classic  α={m['alpha']:.4f}")
    ax1.axhline(ab["alpha_l2m2"], color="k",   lw=1.5, linestyle="dashed",  label=f"BHPTNRSur native  α={ab['alpha_l2m2']:.4f}")
    ax1.axhline(alpha_islam,      color="C2",  lw=1,   linestyle="dotted",  label=f"Islam+22  α={alpha_islam:.4f}")
    ax1.set_xlim(*t_range)
    ax1.set_ylim(0.45, 1.08)
    ax1.grid(True)
    ax1.set_xlabel("t/M")
    ax1.set_ylabel(r"$\alpha$")
    ax1.legend(fontsize=7)
    ax1.set_title(f"Amplitude scaling α  (q = {q:.4g})")

    # Beta panel
    ax2.axhline(m["beta"],        color="C1",  lw=2,   label=f"q_dep_classic  β={m['beta']:.4f}")
    ax2.axhline(ab["beta"],       color="k",   lw=1.5, linestyle="dashed",  label=f"BHPTNRSur native  β={ab['beta']:.4f}")
    ax2.axhline(beta_islam,       color="C2",  lw=1,   linestyle="dotted",  label=f"Islam+22  β={beta_islam:.4f}")
    ax2.set_xlim(*t_range)
    ax2.set_ylim(0.45, 1.08)
    ax2.grid(True)
    ax2.set_xlabel("t/M")
    ax2.set_ylabel(r"$\beta$")
    ax2.legend(fontsize=7)
    ax2.set_title(f"Time-stretch β  (q = {q:.4g})")

    fig.suptitle(f"Constant α, β — q_dep_classic vs BHPTNRSurrogate  (q = {q:.4g})")
    fig.tight_layout()
    path = PLOT_DIR / f"classic_q_dep_{tag}_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_zoomed_plot(m: dict) -> Path:
    q   = m["q"]
    tag = f"q{q:.4g}".replace(".", "p")
    t   = m["t_nr"][m["common"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, m["h_ref"].real,   label="NRHybSur3dq8")
    ax.plot(t, m["h_model"].real, label="BHPT calibrated", linestyle="--")
    ax.set_xlim(-100, 100)
    ax.set_title(f"Classic const α-β, q = {q:.4g};  𝓔 = {m['error']:.3g}")
    ax.grid(True)
    ax.set_xlabel("t/M")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = PLOT_DIR / f"classic_q_dep_{tag}_zoomed.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Global plots
# ---------------------------------------------------------------------------

def save_alpha_beta_vs_q(coeffs: dict[str, np.ndarray]) -> Path:
    q_fine = np.linspace(1.5, 10.5, 400)

    alpha_ours  = np.array([classic.eval_poly(q, coeffs["alpha"]) for q in q_fine])
    beta_ours   = np.array([classic.eval_poly(q, coeffs["beta"])  for q in q_fine])
    alpha_islam = np.array([classic.eval_poly(q, _ISLAM_ALPHA) for q in q_fine])
    beta_islam  = np.array([classic.eval_poly(q, _ISLAM_BETA)  for q in q_fine])
    alpha_native = np.array([bhpt_utils.all_alpha_beta_1D_scaling_params(q, [(2, 2)])["alpha_l2m2"] for q in q_fine])
    beta_native  = np.array([bhpt_utils.all_alpha_beta_1D_scaling_params(q, [(2, 2)])["beta"]       for q in q_fine])

    # Per-q fit points (read from per-q cache if available)
    perq_q, perq_a, perq_b = [], [], []
    cache_path = classic.CACHE_PATH
    if cache_path.exists():
        import json
        cache = json.loads(cache_path.read_text())
        for v in sorted(cache.values(), key=lambda r: r["q"]):
            perq_q.append(v["q"])
            perq_a.append(v["alpha"])
            perq_b.append(v["beta"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for ax, y_ours, y_islam, y_native, perq_y, ylabel in [
        (ax1, alpha_ours, alpha_islam, alpha_native, perq_a, r"$\alpha$"),
        (ax2, beta_ours,  beta_islam,  beta_native,  perq_b, r"$\beta$"),
    ]:
        ax.fill_betweenx([0, 1.2], 3, 8, alpha=0.07, color="C0", label="training range [3,8]")
        ax.plot(q_fine, y_ours,   color="C0", lw=2,   label="q_dep_classic (NRHybSur)")
        ax.plot(q_fine, y_islam,  color="C2", lw=1.5, linestyle="--", label="Islam+22 (SXS NR)")
        ax.plot(q_fine, y_native, color="k",  lw=1,   linestyle=":",  label="BHPTNRSur native")
        if perq_q:
            ax.scatter(perq_q, perq_y, s=12, color="C0", alpha=0.6, zorder=5, label="per-q fit")
        ax.axvline(3, color="gray", lw=0.7, linestyle="--")
        ax.axvline(8, color="gray", lw=0.7, linestyle="--")
        ax.set_xlim(1.2, 11)
        ax.set_ylim(0.45, 1.08)
        ax.set_xlabel("q")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=7)

    ax1.set_title(r"$\alpha(q)$ polynomial")
    ax2.set_title(r"$\beta(q)$ polynomial")
    fig.suptitle("q_dep_classic: constant scaling polynomials vs Islam+22")
    fig.tight_layout()
    path = PLOT_DIR / "classic_q_dep_alpha_beta_vs_q.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_coeff_table(coeffs: dict[str, np.ndarray]) -> Path:
    degree  = len(coeffs["alpha"]) - 1
    q_spot  = [2, 3, 5, 8]
    columns = (["parameter"]
               + [f"c{k}" for k in range(degree + 1)]
               + [f"q={q}" for q in q_spot])
    table_data = []
    for name in classic.PHYS_NAMES:
        coeff_vals = [f"{v:.8g}" for v in coeffs[name]]
        spot_vals  = [f"{classic.eval_poly(q, coeffs[name]):.5g}" for q in q_spot]
        # Islam+22 reference
        ref_arr = _ISLAM_ALPHA if name == "alpha" else _ISLAM_BETA
        ref_vals = [f"[{classic.eval_poly(q, ref_arr):.5g}]" for q in q_spot]
        table_data.append([name, *coeff_vals, *spot_vals])
        table_data.append([f"{name} (Islam+22)", *["—"] * (degree + 1), *ref_vals])

    fig_h = max(3.5, 0.45 * len(table_data) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=columns,
                   loc="center", cellLoc="right", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.3)
    ax.set_title("q_dep_classic: polynomial coefficients and spot values vs Islam+22")
    fig.tight_layout()
    path = PLOT_DIR / "classic_q_dep_coefficients.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    coeffs = read_coefficients()
    degree = len(coeffs["alpha"]) - 1
    print(f"Loaded degree-{degree} PP-anchored polynomial from {MD_PATH}")

    all_paths: list[Path] = []

    # Global plots (once)
    all_paths.append(save_alpha_beta_vs_q(coeffs))
    all_paths.append(save_coeff_table(coeffs))

    # Per-q plots
    for q in Q_VALUES:
        print(f"\nBuilding model at q={q} ...", flush=True)
        m = build_model(q, coeffs)
        print(f"  alpha={m['alpha']:.5f}  beta={m['beta']:.5f}  "
              f"mathcalE={m['error']:.6g}  coverage={m['coverage']:.4f}")
        all_paths += [
            save_waveform_plot(m),
            save_param_plot(m, coeffs),
            save_zoomed_plot(m),
        ]

    print("\nPlots saved:")
    for p in all_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
