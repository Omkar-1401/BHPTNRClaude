"""Constant-q / varying-chi1 alpha,beta overlays for all four spin models.

THE COMPLEMENT TO THE EXISTING OVERLAYS.  Every spin figure in this workspace so far
fixes chi1 and varies q (one curve per q in {2,3,5,8}).  This script does the reverse:
fixes q and varies chi1, so the SPIN trend at a given mass ratio reads directly off a
single axis instead of having to be inferred across four separate figures.

Both views are kept, sorted into two subfolders of each model's plot directory:
    Agentic_spin_plots/<model>/chi_const/   constant chi1, curves over q   (pre-existing)
    Agentic_spin_plots/<model>/q_const/     constant q,    curves over chi1 (this script)

HOUSE FORMAT, unchanged: 1 x 2 (alpha left, beta right), figsize (12, 4), resampled onto
the NR time grid, xlim (-1000, 80), grid on, default rcParams, mathtext axis labels,
PDF + PNG.  The only deliberate change is what the colour axis means -- here it is chi1,
on a cold-to-hot (retrograde-to-prograde) scale, with chi1 = 0 drawn black and dashed as
the no-spin reference wherever the model's chi1 list includes it.

EACH MODEL'S CURVES COME FROM ITS OWN EXISTING PLOT MODULE'S `curve()`, reusing the same
coefficients, the same t0 optimisation and the same mismatch call.  So the E values in
these legends are identical to the ones in the constant-chi1 figures -- these are a
re-slicing of the same data, not a re-fit.

q = 2 CARRIES TWO EXTRAPOLATIONS and the suptitle of the q = 2 panel says so: it is
outside the fits' training range nu in [0.0988, 0.1875], AND outside BHPTNRSur2dq1e3's
own domain (q_min = 3, where the surrogate only warns).  Any q = 2 pathology may be the
source waveform rather than the calibration.

Usage:
    python plot_spin_chi_overlays.py                  # all four models, then organise
    python plot_spin_chi_overlays.py --models physical fluxanchored
    python plot_spin_chi_overlays.py --organise-only  # just sort existing files
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

PLOTS = HERE / "Agentic_spin_plots"
Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
Q_TRAIN_LO = 3.0          # bottom of the fits' training range
Q_SUR_LO = 3.0            # BHPTNRSur2dq1e3 lower bound

# cold -> hot = retrograde -> prograde; chi1 = 0 is the no-spin reference
CHI_COLS = {-0.6: "tab:blue", -0.4: "tab:cyan", -0.2: "tab:purple",
            0.0: "black", 0.2: "tab:orange", 0.4: "tab:brown",
            0.6: "tab:red", 0.8: "tab:pink"}


# ============================================================ model registry
def _load_pn_anchored():
    import spin_alpha_beta_overlay as M
    th9 = M.S.model9()             # installed 9 (2026-09-27), not the nonspinning 9
    ths, src = M.load_theta_spin(False)
    print(f"  theta_spin from {src.name}: " +
          ", ".join(f"{n}={v:+.4f}" for n, v in zip(M.S.SNAMES, ths)), flush=True)
    return (lambda q, c: M.curve(th9, ths, q, c)), M.CHI_PLOT, M.OUT


def _load_fluxanchored():
    import plot_fluxanchored_spin_overlay as M
    d = json.loads((HERE / "fluxanchored_spin_results" / "coeffs_nu.json").read_text())
    th = np.array(d["theta"], float)
    M.FS.WEIGHT = d["weight"]
    print("  theta: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(d["names"], th)),
          flush=True)
    return (lambda q, c: M.curve(th, q, c)), M.CHI, M.OUT


def _load_energy_anchored():
    import plot_energy_anchored_spin_overlay as M
    d = json.loads((HERE / "energy_anchored_spin_results" / "coeffs_nu.json").read_text())
    th = np.array(d["theta"], float)
    M.EA.WEIGHT = d["weight"]
    print("  theta: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(d["names"], th)),
          flush=True)
    return (lambda q, c: M.curve(th, q, c)), M.CHI, M.OUT


def _load_physical():
    import plot_pn_anc_spin_physical_overlay as M
    th, src = M.P.shipped11(False)
    print(f"  theta from {src.name}: " +
          ", ".join(f"{n}={v:+.4f}" for n, v in zip(M.P.NAMES, th)), flush=True)
    return (lambda q, c: M.curve(th, q, c)), M.CHI_PLOT, M.OUT


MODELS = {
    "pn_anchored":    ("pn_anchored_spin",    _load_pn_anchored,
                       "pn-anchored + spin (13 coef; switch + merger fitted on 2d)"),
    "fluxanchored":   ("fluxanchored_spin",   _load_fluxanchored,
                       "fluxanchored + spin (9 coef, fitted on the 2d grid)"),
    "energy_anchored": ("energy_anchored_spin", _load_energy_anchored,
                        "energy_anchored + spin (8 coef, fitted on the 2d grid)"),
    "physical":       ("pn_anc_spin_physical", _load_physical,
                       "pn_anc_spin_physical (1.5PN SO + 2PN SS, 11 coef)"),
}


# ============================================================ drawing
def draw_chi(axa, axb, data, q, chis, legend=True):
    """One panel pair at FIXED q, one curve per chi1."""
    for c in chis:
        tau, al, be, err, t_nr = data[(q, c)]
        m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
        t = t_nr[m]
        ref = (c == 0.0)
        col = CHI_COLS.get(c, "tab:gray")
        ls = "--" if ref else "-"
        lw = 1.6 if ref else 1.3
        lab = r"$\chi_1=" + f"{c:+.1f}$"
        if ref:
            lab += "  [no spin terms]"
        lab += f"  E={err:.2e}"
        axa.plot(t, np.interp(t, tau, al), color=col, ls=ls, lw=lw, label=lab)
        axb.plot(t, np.interp(t, tau, be), color=col, ls=ls, lw=lw,
                 label=r"$\chi_1=" + f"{c:+.1f}$")
    for ax, yl in ((axa, r"$\alpha(t)$"), (axb, r"$\beta(t)$")):
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlim(-1000, 80)
        ax.grid(True)
        ax.set_xlabel(r"$t_{\rm NR}\,/\,M$")
        ax.set_ylabel(yl)
        if legend:
            ax.legend(fontsize=8)


def q_note(q):
    if q < Q_SUR_LO:
        return r"   [$q=2$: extrapolated in $\nu$ AND outside the surrogate domain]"
    if q < Q_TRAIN_LO:
        return r"   [extrapolated in $\nu$]"
    return ""


def run_model(key, organise=True):
    folder, loader, title = MODELS[key]
    print(f"\n{'='*74}\n{key}  ->  Agentic_spin_plots/{folder}/q_const/\n{'='*74}",
          flush=True)
    curve, chis, _ = loader()          # loader OUT now points INTO chi_const/; ignore it
    model_dir = PLOTS / folder         # always the model folder, never a subfolder
    out = model_dir / "q_const"
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for q in Q_INPUTS:
        for c in chis:
            data[(q, c)] = curve(q, c)
            print(f"  q={q:.0f} chi1={c:+.1f}  E={data[(q, c)][3]:.4e}", flush=True)

    stem0 = folder + "_alpha_beta_overlay"

    # one figure per q, chi1 as the colour axis
    for q in Q_INPUTS:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
        draw_chi(axa, axb, data, q, chis)
        axa.set_title(r"$\alpha(t)$ — master")
        axb.set_title(r"$\beta(t)$ — master")
        fig.suptitle(f"{title}: master " + r"$\alpha/\beta$ across $\chi_1$ at $q = "
                     + f"{q:.0f}$" + q_note(q))
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = out / f"{stem0}_q{q:.0f}.{ext}"
            fig.savefig(p)
            print("wrote", p, flush=True)
        plt.close(fig)

    # contact sheet: rows = q, so the mass-ratio trend reads down the column
    fig, axes = plt.subplots(len(Q_INPUTS), 2, figsize=(12, 4 * len(Q_INPUTS)))
    for i, q in enumerate(Q_INPUTS):
        draw_chi(axes[i][0], axes[i][1], data, q, chis, legend=(i == 0))
        axes[i][0].set_title(r"$\alpha(t)$,  $q=" + f"{q:.0f}$" + q_note(q))
        axes[i][1].set_title(r"$\beta(t)$,  $q=" + f"{q:.0f}$")
    fig.suptitle(f"{title}: master " + r"$\alpha/\beta$ across $\chi_1$, "
                 r"$q \in \{2,3,5,8\}$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out / f"{stem0}_q_all.{ext}"
        fig.savefig(p)
        print("wrote", p, flush=True)
    plt.close(fig)

    # split sheets: alpha alone and beta alone, 2 x 2 over q, all chi1 in every panel
    for idx, sym, name in ((1, r"\alpha", "alpha"), (2, r"\beta", "beta")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, q in zip(axes.flat, Q_INPUTS):
            for c in chis:
                tau, al, be, err, t_nr = data[(q, c)]
                y = (al, be)[idx - 1]
                m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
                t = t_nr[m]
                ref = (c == 0.0)
                lab = r"$\chi_1=" + f"{c:+.1f}$" + f"  E={err:.2e}"
                ax.plot(t, np.interp(t, tau, y), color=CHI_COLS.get(c, "tab:gray"),
                        ls="--" if ref else "-", lw=1.6 if ref else 1.3, label=lab)
            ax.axvline(0, color="gray", ls=":", lw=0.8)
            ax.set_xlim(-1000, 80)
            ax.grid(True)
            ax.set_xlabel(r"$t_{\rm NR}\,/\,M$")
            ax.set_ylabel(f"${sym}(t)$")
            ax.set_title(f"$q={q:.0f}$" + (r"  [extrapolated]" if q < Q_TRAIN_LO else ""))
            ax.legend(fontsize=7)
        fig.suptitle(f"{title}: master ${sym}(t)$ across " + r"$\chi_1$")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = out / f"{folder}_{name}_overlay_q_grid.{ext}"
            fig.savefig(p)
            print("wrote", p, flush=True)
        plt.close(fig)

    # the numbers behind the legends, transposed vs the constant-chi1 tables
    print(f"\n  mismatch grid  (rows q, cols chi1)")
    print(f"  {'q':>5} " + " ".join(f"{c:>+10.1f}" for c in chis))
    for q in Q_INPUTS:
        print(f"  {q:>5.0f} " + " ".join(f"{data[(q, c)][3]:10.3e}" for c in chis))

    # persist the grid so the heatmap can be redrawn without recomputing curves
    grid = {"q": Q_INPUTS, "chi1": list(chis),
            "E": [[data[(q, c)][3] for c in chis] for q in Q_INPUTS]}
    p = out / f"{folder}_mismatch_grid.json"
    p.write_text(json.dumps(grid, indent=1))
    print("wrote", p, flush=True)

    if organise:
        organise_folder(model_dir)


# ============================================================ E(q, chi1) heatmap
def plot_heatmaps(keys):
    """q on x, chi1 on y, mathcal{E} as colour -- one figure per model, shared log scale.

    Reads the <folder>_mismatch_grid.json that run_model writes.  Cells are drawn on
    equal-width categorical axes (the q grid 2,3,5,8 is uneven) and annotated with E.
    """
    from matplotlib.colors import LogNorm
    grids = {}
    for key in keys:
        folder = MODELS[key][0]
        p = PLOTS / folder / "q_const" / f"{folder}_mismatch_grid.json"
        if not p.exists():
            print(f"  [skip] {p} missing -- run the model first")
            continue
        grids[key] = json.loads(p.read_text())
    if not grids:
        return
    allE = np.concatenate([np.ravel(g["E"]) for g in grids.values()])
    norm = LogNorm(10 ** np.floor(np.log10(allE.min())),
                   10 ** np.ceil(np.log10(allE.max())))

    for key, g in grids.items():
        folder, _, title = MODELS[key]
        qs, cs = g["q"], g["chi1"]
        E = np.array(g["E"], float).T                     # rows chi1, cols q
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.pcolormesh(np.arange(len(qs) + 1), np.arange(len(cs) + 1), E,
                           norm=norm, cmap="viridis", edgecolors="w", linewidth=0.5)
        for i in range(len(cs)):
            for j in range(len(qs)):
                v = E[i, j]
                ax.text(j + 0.5, i + 0.5, f"{v:.2e}", ha="center", va="center",
                        fontsize=8, color="w" if norm(v) < 0.6 else "k")
        ax.set_xticks(np.arange(len(qs)) + 0.5)
        ax.set_xticklabels([f"{q:g}" for q in qs])
        ax.set_yticks(np.arange(len(cs)) + 0.5)
        ax.set_yticklabels([f"{c:+.1f}" for c in cs])
        jb = sum(q < Q_TRAIN_LO for q in qs)             # training-range boundary
        if 0 < jb < len(qs):
            ax.axvline(jb, color="r", ls="--", lw=1.2)
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$\chi_1$")
        fig.colorbar(im, ax=ax, label=r"$\mathcal{E}$")
        ax.set_title(f"{title}\n" + r"$\mathcal{E}(q,\chi_1)$  "
                     r"(red dashed: $q<3$ extrapolated)", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = PLOTS / folder / f"{folder}_mismatch_heatmap.{ext}"
            fig.savefig(p)
            print("wrote", p, flush=True)
        plt.close(fig)


# ============================================================ folder hygiene
def organise_folder(folder: Path):
    """Sort a model's plot folder into q_const/ and chi_const/.  Idempotent."""
    if not folder.is_dir():
        print(f"  [skip] {folder} does not exist")
        return
    if folder.name in ("q_const", "chi_const"):
        # never sort a sorted folder into itself -- that is how the figures ended up
        # nested two deep on the first run (job 3520143)
        print(f"  [skip] {folder.name} is already a sorted subfolder")
        return
    qc, cc = folder / "q_const", folder / "chi_const"
    qc.mkdir(exist_ok=True)
    cc.mkdir(exist_ok=True)
    moved = 0
    for p in sorted(folder.glob("*.*")):
        if not p.is_file() or p.suffix not in (".pdf", ".png"):
            continue
        name = p.name
        if "_overlay_q" in name:
            dest = qc
        elif "_overlay_chi" in name or name.endswith(("_overlay_all.pdf", "_overlay_all.png")):
            dest = cc
        elif "_vs_predecessor_diff" in name:
            dest = cc          # chi1 down the rows, same slicing as the chi figures
        else:
            continue
        shutil.move(str(p), str(dest / name))
        moved += 1
    print(f"  organised {folder.name}: moved {moved} file(s) "
          f"-> q_const/ ({len(list(qc.glob('*')))}), chi_const/ ({len(list(cc.glob('*')))})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS), help="which models to plot")
    ap.add_argument("--organise-only", action="store_true",
                    help="skip plotting; only sort existing files into the subfolders")
    ap.add_argument("--heatmap-only", action="store_true",
                    help="skip curves; redraw E(q,chi1) heatmaps from saved grid JSONs")
    args = ap.parse_args()

    mpl.rcdefaults()                     # AFTER the model imports happen in loaders
    mpl.rcParams["text.usetex"] = False

    if args.organise_only:
        for key in MODELS:
            organise_folder(PLOTS / MODELS[key][0])
        return

    if not args.heatmap_only:
        for key in args.models:
            run_model(key)
            mpl.rcdefaults()             # each loader may re-import a font-setting module
            mpl.rcParams["text.usetex"] = False
    plot_heatmaps(args.models)


if __name__ == "__main__":
    main()
