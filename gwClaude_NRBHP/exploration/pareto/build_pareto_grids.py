"""Pareto fronts per TRAINING-GRID SIZE (user, 2026-10-26), from the training-size study (../training_size/summary.json):
ten models refit on G24 / G48 / G96 / G192 (uniform in nu x chi1) and scored on V300 (300 fresh uniform points, score only).
Reported run per grid = the study's choice (lowest TRAINING mean over starts).  Coefficients = fitted coefficients
(pn_anchored_spin: 13, of which 10 refit in the study and 3 inherited from the nonspinning fit; split B: 13 + window).
    python pareto/build_pareto_grids.py      -> pareto/pareto_grids.{png,pdf}, pareto/pareto_grids.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
S = json.loads((HERE.parent / "training_size" / "summary.json").read_text())
NC = {"pn13": 13, "flux_orbit": 9, "splitA": 14, "splitB": 14, "p39L4": 18, "p40": 18, "lean": 16, "lean12": 12, "lean9": 9, "rink": 12}
LAB = {"pn13": "pn_anchored_spin", "flux_orbit": "flux_anc_spin_with_orbit", "splitA": "split A", "splitB": "split B",
       "p39L4": "pass 39 L4", "p40": "pass 40", "lean": "pass 40 lean (K1)", "lean12": "49b lean-12", "lean9": "49c lean-9", "rink": "Rink et al. form"}
FAM = {"pn13": "installed", "flux_orbit": "installed", "splitA": "split", "splitB": "split", "p39L4": "exploration",
       "p40": "exploration", "lean": "exploration", "lean12": "pruned", "lean9": "pruned", "rink": "published"}
MK = {"pn13": "s", "flux_orbit": "o", "splitA": "^", "splitB": "v", "p39L4": "D", "p40": "P", "lean": "h", "lean12": "p", "lean9": "X", "rink": "*"}
COL = {"published": "#e34948", "installed": "#8c8c8c", "split": "#eb6834", "exploration": "#2a78d6", "pruned": "#1baf7a"}


def front(pts, key):
    return sorted([p for p in pts if not any(o["n"] <= p["n"] and o[key] <= p[key] and (o["n"] < p["n"] or o[key] < p[key]) for o in pts)],
                  key=lambda p: p["n"])


def main():
    grids = ("24", "48", "96", "192"); out = {}
    for g in grids:
        out[g] = [dict(model=m, name=LAB[m], family=FAM[m], n=NC[m], **{f"V300_{k}": S[m]["curve"][g]["V300"][k] for k in ("median", "mean", "max")},
                       train_mean=S[m]["curve"][g]["train"]["mean"], chosen=S[m]["curve"][g]["chosen"]) for m in NC]
    res = {g: {k: [p["name"] for p in front(out[g], k)] for k in ("V300_median", "V300_mean")} for g in grids}
    (HERE / "pareto" / "pareto_grids.json").write_text(json.dumps(dict(points=out, fronts=res), indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.6), sharex=True, sharey="row")
    for j, g in enumerate(grids):
        for i, (key, lab) in enumerate((("V300_median", "V300 median"), ("V300_mean", "V300 mean"))):
            ax = axs[i, j]; pts = out[g]
            for p in pts:                              # identity by marker + one shared legend (labels overlapped at equal n)
                ax.scatter(p["n"], p[key], s=90 if MK[p["model"]] == "*" else 60, marker=MK[p["model"]], color=COL[p["family"]],
                           zorder=3, edgecolors="black", linewidths=0.5, alpha=0.9, label=p["name"])
            fr = front(pts, key)
            ax.step([p["n"] for p in fr], [p[key] for p in fr], where="post", color="#000000", lw=1.2, zorder=2)
            ax.set_yscale("log"); ax.grid(True, color="#e6e6e6", lw=0.6, which="both"); ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"trained on G{g}" + (" (= T96 layout)" if g == "96" else ""), fontsize=11)
            if j == 0:
                ax.set_ylabel(r"$\mathcal{E}$, " + lab)
            if i == 1:
                ax.set_xlabel("fitted coefficients")
    fig.suptitle("Pareto fronts by training-grid size (scored on V300, 300 fresh uniform points; black = front)", fontsize=12)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9.5, frameon=False, handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for e in ("png", "pdf"):
        fig.savefig(HERE / "pareto" / f"pareto_grids.{e}", dpi=160)
    for g in grids:
        for k in ("V300_median", "V300_mean"):
            print(f"G{g} {k} front: " + " -> ".join(f"{p['name']} ({p['n']}c, {p[k]:.2e})" for p in front(out[g], k)))


if __name__ == "__main__":
    main()
