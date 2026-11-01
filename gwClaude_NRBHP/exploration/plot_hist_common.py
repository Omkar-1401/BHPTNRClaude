"""Train / validation / test histograms on ONE common x range and bin set (2026-10-15, user request).

Redraws, from stored per-case scores only (no model evaluation, no fitting):
    spin_dep/Agentic_spin_plots/pn_anchored_spin/paper/pn_anchored_spin_hist_train_vs_heldout.{pdf,png}
    spin_dep/Agentic_spin_plots/flux_anc_spin_with_orbit/paper/flux_anc_spin_with_orbit_hist_train_vs_heldout.{pdf,png}
    exploration/plots_p36/p36_hist_train_val_test.{pdf,png}
Sources: training = the recomputed T96 in each paper/heldout_mismatch.json (p36: plots_p36/p36_scores.json); validation =
H270 from the same files; test = the R100 score VALUES of exploration/t100/<model>.json (sealed set: values only, the
first 100 entries, which are R100; no locations are read or shown).
The originals are backed up to exploration/backup_hist_20261015/ before they are overwritten.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; SD = HERE.parent / "spin_dep" / "Agentic_spin_plots"
BK = HERE / "backup_hist_20261015"
C_TRAIN, C_BLUE, C_ORANGE = "#000000", "#2a78d6", "#eb6834"      # plot_spin_paper_figs palette


def test_values(name):
    return np.array(list(json.loads((HERE / "t100" / f"{name}.json").read_text())["cases"].values()))[:100]


def load():
    out = {}
    for m, t, lab in (("pn_anchored_spin", "pn13", "pn_anchored_spin (13 coef)"),
                      ("flux_anc_spin_with_orbit", "flux_orbit", "flux_anc_spin_with_orbit (9 coef + derived SO)")):
        h = json.loads((SD / m / "paper" / "heldout_mismatch.json").read_text())
        out[m] = dict(label=lab, train=np.array(list(h["training"].values())), val=np.array(list(h["heldout"].values())),
                      test=test_values(t), dest=SD / m / "paper", name=f"{m}_hist_train_vs_heldout")
    s = json.loads((HERE / "plots_p36" / "p36_scores.json").read_text())
    out["p36"] = dict(label="pass 36 D1 (18 coef)", train=np.array(list(s["T96"].values())),
                      val=np.array(list(s["H270"].values())), test=test_values("p36"),
                      dest=HERE / "plots_p36", name="p36_hist_train_val_test")
    if (HERE / "plots_p39" / "p39_scores.json").exists():          # pass 39 L4 (leader since 2026-10-18)
        s = json.loads((HERE / "plots_p39" / "p39_scores.json").read_text())
        out["p39"] = dict(label="pass 39 L4 (18 coef)", train=np.array(list(s["T96"].values())),
                          val=np.array(list(s["H270"].values())), test=test_values("p39"),
                          dest=HERE / "plots_p39", name="p39_hist_train_val_test")
    for tag, lab, t in (("p40", "pass 40 (18 coef)", "p40"), ("p40lean", "pass 40 lean (16 coef, OT onset)", "k1"), ("p48", "pass 48 (18 coef, closed form)", "m48"), ("p48lean", "pass 48 lean (16 coef, closed form, OT onset)", "lean48"), ("p49b12", "pass 49b lean-12 (12 coef)", "p49b_n12"), ("p49b9", "pass 49c lean-9 (9 coef)", "p49b_n9")):
        f = HERE / f"plots_{tag}" / f"{tag}_scores.json"
        if f.exists():
            s = json.loads(f.read_text())
            out[tag] = dict(label=lab, train=np.array(list(s["T96"].values())), val=np.array(list(s["H270"].values())),
                            test=test_values(t), dest=HERE / f"plots_{tag}", name=f"{tag}_hist_train_val_test")
    return out


def style(ax):
    ax.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    D = load()
    allv = np.log10(np.concatenate([np.r_[d["train"], d["val"], d["test"]] for d in D.values()]))
    lo, hi = np.floor(allv.min() * 4) / 4, np.ceil(allv.max() * 4) / 4
    bins = np.arange(lo, hi + 1e-9, 0.1)
    print(f"common x range [{lo}, {hi}], {len(bins) - 1} bins of 0.1 dex")
    BK.mkdir(exist_ok=True)
    for key, d in D.items():
        for ext in ("pdf", "png"):
            src = d["dest"] / f"{d['name']}.{ext}"
            if src.exists() and not (BK / f"{d['name']}.{ext}").exists():
                shutil.copy2(src, BK / f"{d['name']}.{ext}")
        fig, ax = plt.subplots(figsize=(6.4, 4.4)); top = 0
        for v, lab, ls, col in ((d["train"], "training T96", "--", C_TRAIN), (d["val"], "validation H270", "-", C_BLUE),
                                (d["test"], "test R100 (unseen)", "-", C_ORANGE)):
            lv = np.log10(v)
            ax.hist(lv, bins=bins, density=True, histtype="step", ls=ls, lw=1.8, color=col,
                    label=f"{lab} ({len(v)}), median {np.median(v):.2e}")
            top = max(top, np.histogram(lv, bins, density=True)[0].max())
        ax.set_xlim(lo, hi); ax.set_ylim(0, 1.5 * top)
        ax.set_xlabel(r"$\log_{10}\,\mathcal{E}$"); ax.set_ylabel("probability density")
        ax.set_title(d["label"], fontsize=11); ax.legend(frameon=False, fontsize=9, loc="upper left"); style(ax)
        for ext in ("pdf", "png"):
            fig.savefig(d["dest"] / f"{d['name']}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {d['dest'] / d['name']}.{{pdf,png}}  (train {np.median(d['train']):.2e}, val "
              f"{np.median(d['val']):.2e}, test {np.median(d['test']):.2e})")


if __name__ == "__main__":
    main()
