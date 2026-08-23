"""
Plotting script for the wf_nu_switchless_alpha model (ungated alpha, gated beta).
Run fit_scaling_wf_nu_switchless_alpha.py first.
Plots -> Agentic_plots/wf_nu_switchless_alpha/ (always saved).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTutils"))

import bhpt_utils
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_wf_nu_switchless_alpha as wfsla

mpl.rcdefaults()
mpl.rcParams["text.usetex"] = False

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
MD_PATH  = ROOT / "scaling_wf_nu_switchless_alpha.md"
PLOT_DIR = ROOT / "Agentic_plots" / "wf_nu_switchless_alpha"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def read_markdown_coefficients(md_path):
    text  = md_path.read_text()
    start = text.index("## Selected Master Coefficients")
    table_lines, in_table = [], False
    for line in text[start:].splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            table_lines.append(s); in_table = True
        elif in_table:
            break
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c.replace(":", "").replace("-", "") == "" for c in cells):
            continue
        rows.append(cells)
    coeffs = {row[0].strip(): np.array([float(v) for v in row[2:]], dtype=float) for row in rows[1:]}
    missing = [n for n in wfsla.PARAM_NAMES if n not in coeffs]
    if missing:
        raise ValueError(f"Missing coefficient rows: {missing}")
    return {n: coeffs[n] for n in wfsla.PARAM_NAMES}


def build_model(q, coeffs):
    qdep.generate_and_cache_waveform(q)
    case   = wfsla.load_case(q, source_stride=3, nr_stride=5)
    params = wfsla.constrained_master_params(q, coeffs)
    params, _ = wfsla.polish_nuisance(params, case)
    ev = wfsla.evaluate_model_switchless_alpha(
        params, case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"], case["losses"])
    return {"eval": ev, "params": params, "case": case}


def _extrap(q): return q < 3.0 or q > 8.0
def _tag(q):
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if _extrap(q) else "")
def _title(q, error=float("nan")):
    ex = " [extrapolation]" if _extrap(q) else ""
    es = f";  mathcalE = {error:.3g}" if np.isfinite(error) else ""
    return f"wf-nu-switchless-alpha  q={q}{ex};  nu={wfsla.get_nu(q):.4f}{es}"


def save_waveform_plot(q, t, h_nr, h_model, error):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})
    for ax, xl in [(a1, (-1000, -200)), (a2, (-200, 100))]:
        ax.plot(t, np.real(h_nr), label="NR 22"); ax.plot(t, np.real(h_model), label="BHPT ungated-alpha", alpha=0.8)
        ax.set_xlim(*xl); ax.grid(True); ax.set_xlabel("t/M")
    a1.legend(fontsize=8); fig.suptitle(_title(q, error)); fig.tight_layout()
    p = PLOT_DIR / f"sla_{_tag(q)}_waveform.pdf"; fig.savefig(p); plt.close(fig); return p


def save_parameter_plot(q, tau, alpha, beta, e_hat, j_hat):
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(q, [creative.MODE])
    lbl = f"wf-nu-sla nu={wfsla.get_nu(q):.4f}"
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(tau, alpha, label=lbl); ax[0].axhline(ab["alpha_l2m2"], color="k", ls="--", label="BHPTNRSur alpha")
    ax[0].set_xlim(-1000, 100); ax[0].grid(True); ax[0].set_xlabel("t_NR/M"); ax[0].set_ylabel("alpha(t)"); ax[0].legend(fontsize=7)
    ax[1].plot(tau, beta, label=lbl); ax[1].axhline(ab["beta"], color="k", ls="--", label="BHPTNRSur beta")
    ax[1].set_xlim(-1000, 100); ax[1].grid(True); ax[1].set_xlabel("t_NR/M"); ax[1].set_ylabel("beta(t)"); ax[1].legend(fontsize=7)
    si = np.argsort(e_hat); ax[2].plot(e_hat[si], alpha[si], lw=1)
    ax[2].set_xlabel("Ehat"); ax[2].set_ylabel("alpha"); ax[2].set_title("alpha vs Ehat"); ax[2].grid(True)
    fig.suptitle(_title(q)); fig.tight_layout()
    p = PLOT_DIR / f"sla_{_tag(q)}_params.pdf"; fig.savefig(p); plt.close(fig); return p


def save_alpha_beta_overlay(results):
    cols = {2.0: "C3", 3.0: "C0", 5.0: "C2", 8.0: "C4"}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
    for q in Q_INPUTS:
        ev = results[q]["eval"]; case = results[q]["case"]; tau = ev["tau"]; t_nr = case["t_nr"]
        m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1])); t = t_nr[m]
        lw = 2.2 if q == 2 else 1.3
        axa.plot(t, np.interp(t, tau, ev["alpha"]), color=cols[q], lw=lw,
                 label=f"q={q:.0f}{' [extrap]' if q == 2 else ''}  E={ev['error']:.2e}")
        axb.plot(t, np.interp(t, tau, ev["beta"]), color=cols[q], lw=lw, label=f"q={q:.0f}")
    for ax, yl, tl in [(axa, "alpha(t)", "alpha(t) — master (ungated alpha)"), (axb, "beta(t)", "beta(t) — master (gated)")]:
        ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80); ax.grid(True)
        ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl); ax.set_title(tl); ax.legend(fontsize=8)
    fig.suptitle("wf-nu-switchless-alpha: master alpha/beta across q"); fig.tight_layout()
    p = PLOT_DIR / "sla_alpha_beta_overlay.pdf"; fig.savefig(p); plt.close(fig); return p


def main():
    coeffs = read_markdown_coefficients(MD_PATH)
    results, errors, paths = {}, {}, []
    for q in Q_INPUTS:
        print(f"\n--- q={q}{' [EXTRAP]' if _extrap(q) else ''} ---", flush=True)
        r = build_model(q, coeffs); ev = r["eval"]; results[q] = r; errors[q] = ev["error"]
        print(f"  nu={wfsla.get_nu(q):.4f}  mathcalE={ev['error']:.6g}")
        t = r["case"]["t_nr"][ev["common"]]
        paths += [save_waveform_plot(q, t, ev["h_ref"], ev["h_model"], ev["error"]),
                  save_parameter_plot(q, ev["tau"], ev["alpha"], ev["beta"], ev["e_hat"], ev["j_hat"])]
    paths.append(save_alpha_beta_overlay(results))
    print("\n=== Summary ===")
    for q in Q_INPUTS:
        print(f"  q={q:<4}{' [extrap]' if _extrap(q) else '        '}  nu={wfsla.get_nu(q):.4f}  mathcalE={errors[q]:.4e}")
    print(f"\nPlots: {PLOT_DIR}")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
