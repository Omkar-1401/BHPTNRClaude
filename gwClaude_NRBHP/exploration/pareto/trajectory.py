"""The exploration as a trajectory through model space (user, 2026-10-28: collaborators' 'phase transition' question).

(a) coefficients vs T96 median: installed (hand-written) models, then the exploration passes joined in chronological
    order (best model of each pass), with the per-case FLOORS of each model class as horizontal lines -- a floor is the
    error reached by fitting that class's form to every case separately, i.e. the best any coefficient layer on it can do.
(b) the same models against pass order, with the running best.
Numbers: pareto_data.json (built by build_pareto.py) and the logged floors (EXPLORATION.md passes 5b, 12, 17, 27).
    python pareto/trajectory.py    -> pareto/trajectory.{png,pdf}
"""
import json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "pareto_data.json").read_text())["points"]
FLOORS = [(6.66e-4, "installed form (FLUX), per-case floor", "#8c8c8c"),          # pass 5b, T96 median
          (2.53e-4, "two-regime form (M2), per-case floor", "#2a78d6"),           # pass 5b
          (1.3e-5, "two-regime + spline shapes, per-case floor", "#1baf7a")]     # passes 12 / 17 / 27


def pass_key(name):
    m = re.match(r"pass (\d+)([a-z]?)", name)
    return None if m is None else int(m.group(1)) + (0.5 if m.group(2) else 0) + (0.2 if "lean" in name else 0)


def main():
    hand = [p for p in D if p["family"] in ("installed", "published", "split")]
    expl = [dict(p, k=pass_key(p["name"])) for p in D if p["family"] in ("exploration", "pruned")]
    side = [p for p in expl if p["k"] == 43]                        # theory-clock excursion (pass 43)
    main_ = [p for p in expl if p["k"] != 43]
    best = {}
    for p in main_:
        k = p["k"] if p["family"] == "exploration" else (49.5, p["n"])
        if k not in best or p["T96_median"] < best[k]["T96_median"]:
            best[k] = p
    path, rb = [], np.inf                                          # main line: per-pass best within 25% of the running best
    for p in sorted(best.values(), key=lambda p: (p["k"], -p["n"])):
        if p["family"] == "pruned" or p["T96_median"] <= 1.25 * rb:
            path.append(p); rb = min(rb, p["T96_median"])
    cm = plt.get_cmap("viridis")
    kmin, kmax = 6, 50
    c = lambda k: cm((k - kmin) / (kmax - kmin) * 0.9)

    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw=dict(width_ratios=(1.25, 1)))
    # --- (a) model space
    for y, lab, col in FLOORS:
        a.axhline(y, color=col, ls="--", lw=1.0, zorder=1)
        a.text(36.5, y * 1.06, lab, ha="right", va="bottom", fontsize=8, color=col)
    a.axhspan(6.66e-4, 3e-3, color="#8c8c8c", alpha=0.07, zorder=0)
    GOLD = "#f2b134"                                                # flux_anc_spin_with_orbit, the 9c comparison model
    LBL = {"junction_spin": ((0, -13), "center"), "flux_anc_spin_with_orbit": ((7, -11), "left"),
           "fluxanchored_spin": ((7, 0), "left"), "energy_anchored_spin": ((7, 0), "left"),
           "pn_anc_spin_physical": ((0, 9), "center"), "pn_anchored_spin": ((7, 2), "left"), "Rink et al. 2024": ((9, 0), "left")}
    for p in hand:
        mk = {"installed": "o", "published": "*", "split": "s"}[p["family"]]
        fo = p["name"] == "flux_anc_spin_with_orbit"
        a.scatter(p["n"], p["T96_median"], marker="D" if fo else mk, s=110 if mk == "*" else (70 if fo else 55),
                  color=GOLD if fo else ("#8c8c8c" if mk != "*" else "#e34948"), edgecolors="black", linewidths=0.6 if fo else 0.5, zorder=5)
        if p["name"] in LBL:
            off, ha = LBL[p["name"]]
            a.annotate(p["name"], (p["n"], p["T96_median"]), xytext=off, textcoords="offset points", ha=ha, va="center",
                       fontsize=8 if fo else 7.5, fontweight="bold" if fo else "normal", color="#1a1a1a", zorder=6,
                       bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for p in main_:
        a.scatter(p["n"], p["T96_median"], s=14, color=c(p["k"]), alpha=0.45, zorder=2)
    xs = [p["n"] for p in path]; ys = [p["T96_median"] for p in path]
    for i in range(len(path) - 1):
        a.annotate("", (xs[i + 1], ys[i + 1]), (xs[i], ys[i]), zorder=3,
                   arrowprops=dict(arrowstyle="-|>", color=c(path[i]["k"]), lw=1.4, shrinkA=2, shrinkB=2, mutation_scale=9))
    a.scatter(xs, ys, s=34, c=[c(p["k"]) for p in path], edgecolors="black", linewidths=0.4, zorder=4)
    k42 = [p for p in path if p["k"] == 42][0]
    for p in side:
        a.annotate("", (p["n"], p["T96_median"]), (k42["n"], k42["T96_median"]), zorder=3,
                   arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.0, ls=(0, (3, 2)), mutation_scale=8))
        a.scatter(p["n"], p["T96_median"], s=34, color="#c0392b", edgecolors="black", linewidths=0.4, zorder=4)
    note = dict(fontsize=9, color="#1a1a1a", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))
    a.annotate("hand-written, single regime\n(7-13 coef, at or near\ntheir class floor)", (8.2, 2.7e-3), ha="center", **note)
    a.annotate("hand-written split\n(inspiral | merger, 14 coef)", (14.0, 4.6e-4), (8.0, 3.2e-4), ha="center",
               arrowprops=dict(arrowstyle="-", color="#777", lw=0.7), **note)
    a.annotate("pass 5-6: second regime\n(ppBHPT-clock merger blend)", (17, 6.0e-4), (22.5, 8.2e-4), ha="left",
               arrowprops=dict(arrowstyle="-", color="#777", lw=0.7), **note)
    a.annotate("climb: +coefficients\n(passes 6-21)", (29, 4.6e-4), ha="center", **note)
    a.annotate("condense: 34 -> 18 coef\nat constant error\n(passes 21-40)", (27, 9.0e-5), ha="center", **note)
    a.annotate("prune: 16 -> 12\n(pass 49b)", (11.0, 1.55e-4), ha="center", **note)
    a.annotate("pass 43: 1PA theory clock,\nfalls back to the\nhand-written level", (16.8, 1.9e-3), ha="left", va="center", **note)
    a.set_yscale("log"); a.set_xlim(5, 37); a.set_ylim(8e-6, 4e-3)
    a.set_xlabel("fitted coefficients"); a.set_ylabel(r"training $\mathcal{E}$ (T96 median)")
    a.set_title("(a) the trajectory through model space", fontsize=11)
    a.grid(True, color="#ececec", lw=0.6, which="both"); a.set_axisbelow(True)
    # --- (b) against pass order
    for p in hand:
        fo = p["name"] == "flux_anc_spin_with_orbit"
        b.scatter(-3, p["T96_median"], marker="D" if fo else {"installed": "o", "published": "*", "split": "s"}[p["family"]],
                  s=70 if fo else 60, color=GOLD if fo else ("#e34948" if p["family"] == "published" else "#8c8c8c"),
                  edgecolors="black", linewidths=0.4, zorder=4 if fo else 3)
        if fo:
            b.axhline(p["T96_median"], color=GOLD, lw=1.2, ls=":", zorder=1)
            b.text(50, p["T96_median"] * 1.06, "flux_anc_spin_with_orbit (9 coef)", ha="right", va="bottom", fontsize=8.5,
                   color="#b07a10", fontweight="bold")
    for p in main_ + side:
        b.scatter(p["k"], p["T96_median"], s=20, color="#c0392b" if p in side else c(p["k"]), alpha=0.8, zorder=3,
                  edgecolors="none")
    seq = sorted(main_, key=lambda p: p["k"])
    run, rb = [], np.inf
    for p in seq:
        rb = min(rb, p["T96_median"]); run.append((p["k"], rb))
    hb = min(p["T96_median"] for p in hand if p["family"] != "published")
    b.step([-3] + [r[0] for r in run], [hb] + [r[1] for r in run], where="post", color="black", lw=1.2, label="running best")
    for y, lab, col in FLOORS:
        b.axhline(y, color=col, ls="--", lw=1.0)
    b.axvspan(-5, 0, color="#8c8c8c", alpha=0.07)
    b.text(-2.5, 3.2e-3, "hand-\nwritten", ha="center", va="top", fontsize=9, color="#555")
    b.set_yscale("log"); b.set_xlim(-5, 51); b.set_ylim(8e-6, 4e-3)
    b.set_xlabel("exploration pass"); b.set_ylabel(r"training $\mathcal{E}$ (T96 median)")
    b.set_title("(b) the same, in pass order", fontsize=11); b.legend(fontsize=9, frameon=False, loc="lower left")
    b.grid(True, color="#ececec", lw=0.6, which="both"); b.set_axisbelow(True)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(cm(np.linspace(0, 0.9, 64))),
                               norm=matplotlib.colors.Normalize(kmin, kmax))
    fig.colorbar(sm, ax=b, pad=0.01, fraction=0.04).set_label("pass")
    fig.suptitle("Spin calibration: hand-written models vs the exploration trajectory "
                 "(dashed = per-case floor of each model class)", fontsize=12)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(HERE / f"trajectory.{e}", dpi=160)


if __name__ == "__main__":
    main()
