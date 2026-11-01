"""Pareto front: fitted coefficients vs error, every spin model of the repo (user, 2026-10-26).

Error axes: T96 median and mean (training grid, available for every model) and the sealed-test R100 median (where
scored).  Sources: the exploration result jsons (pN/*.json), the installed models' coeffs jsons, the split models'
results, and the logged numbers for the early passes (EXPLORATION.md, passes 6-14) and the published Rink et al. model.
Coefficient counts are FITTED coefficients; derived / theory inputs and constants fitted offline to theory data are
not counted (user rule 2026-10-18).  The Ori–Thorne window start of split B is counted as 1 (13 + T0 -> 14).
    python pareto/build_pareto.py      -> pareto/pareto_data.json, pareto/pareto_front.{png,pdf}
"""
import json, glob
from pathlib import Path
import numpy as np
from labels import label_front

HERE = Path(__file__).resolve().parents[1]
P = []


def add(name, family, n, med, mean, r100=None, note=""):
    P.append(dict(name=name, family=family, n=int(n), T96_median=float(med), T96_mean=float(mean),
                  R100_median=None if r100 is None else float(r100), note=note))


SHOW_CLOSED = False   # user 2026-10-27: closed-form-input models (passes 46-48) off the front for now
# published and installed / older spin models (T96 from the coeffs / results jsons; R100 from exploration/t100)
t100 = lambda k: json.loads((HERE / "t100" / f"{k}.json").read_text())["R100"]["median"] if (HERE / "t100" / f"{k}.json").exists() else None
add("Rink et al. 2024", "published", 12, 2.35e-3, 3.47e-3, t100("rink"))
_z = HERE / "fdm" / "results" / "zhong_scores.json"   # Zhong et al. 2025 (arXiv:2503.04534) extracted (2,2) coefficients
if _z.exists():                                        # spin-blind: alpha'(q), beta'(q) only; their literal phase map
    _zz = json.loads(_z.read_text())["zhong"]["T96"]
    add("Zhong et al. 2025", "published", 8, _zz["median"], _zz["mean"], t100("zhong"), "spin-blind; non-spinning CfC fit")
add("pn_anchored_spin", "installed", 13, 8.649e-4, 1.172e-3, t100("pn13"))
add("pn_anc_spin_physical", "installed", 11, 9.642e-4, 1.143e-3)
add("flux_anc_spin_with_orbit", "installed", 9, 8.859e-4, 1.092e-3, t100("flux_orbit"))
add("fluxanchored_spin", "installed", 9, 1.300e-3, 2.110e-3)
add("energy_anchored_spin", "installed", 8, 1.574e-3, 2.359e-3)
add("junction_spin", "installed", 7, 9.64e-4, 1.335e-3)
add("split (slim, T = -1)", "split", 14, 5.42e-4, 9.44e-4)
add("split A (slim, T = -2)", "split", 14, 4.643e-4, 8.539e-4, json.loads((HERE / "t100" / "split_slim__win-2.0.json").read_text())["R100"]["median"])
add("split B (b = 0, T = -3)", "split", 14, 6.206e-4, 8.486e-4, json.loads((HERE / "t100" / "split_tline.json").read_text())["R100"]["median"], "13 fitted + window start")
# early exploration passes (logged; mean / median)
for nm, n, mean, med in (("pass 6 M2", 17, 9.01e-4, 6.01e-4), ("pass 7 V1", 19, 8.88e-4, 5.91e-4), ("pass 7b V2", 21, 8.46e-4, 5.29e-4),
                         ("pass 8 V3", 23, 6.44e-4, 3.93e-4), ("pass 11 V4", 25, 5.82e-4, 3.67e-4), ("pass 13 V5", 32, 2.89e-4, 2.32e-4),
                         ("pass 14 V6", 32, 2.74e-4, 1.74e-4)):
    add(nm, "exploration", n, med, mean)
# exploration result jsons
R100 = {"p23/p23_Z.json": "v7s3", "p36/p36_D1.json": "p36", "p38/p38_N2.json": "p38n2", "p39/p39_L4.json": "p39", "p42/p42_K1.json": "k1"}
skip = {"p43/p43_H3.json", "p43/p43_H4.json", "p43/p43_H5.json",   # passes 50b/c: not on the front (user 2026-10-27)
        "p26/p26_iscowin.json", "p26/p26_iscowin_forkbug.json", "p46/p46_backup.json", "p47/p46.json"}
for f in sorted(glob.glob(str(HERE / "p*/p*.json"))):
    rel = str(Path(f).relative_to(HERE))
    if rel in skip:
        continue
    try:
        d = json.loads(Path(f).read_text())
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    items = [("", d)] + [(k, v) for k, v in d.items() if isinstance(v, dict)]
    for tag, x in items:
        t = x.get("T96") if isinstance(x, dict) else None
        if not (isinstance(t, list) and len(t) == 3 and all(isinstance(v, (int, float)) for v in t)) or t[1] > 0.5:
            continue
        n = x.get("n_coef") or (len(x["free"]) if isinstance(x.get("free"), list) else None)
        if rel.startswith(("p40/", "p46/", "p47/")):
            n = 18                                         # pass-40 coefficients, inputs swapped
        if rel.startswith("p48/"):
            n = {"m48": 18, "lean48": 16}[tag]
        if rel.startswith("p49/") or (not SHOW_CLOSED and rel.startswith(("p46/", "p47/", "p48/"))):
            continue                                       # added below with their own labels
        if n is None:
            continue
        name = f"pass {rel.split('/')[0][1:]} {tag or Path(f).stem.split('_', 1)[-1]}".strip()
        r = R100.get(rel)
        if rel == "p40/p40.json" and tag == "O2n":
            r = "p40"; name = "pass 40"
        if rel == "p48/p48.json":
            r = {"m48": "m48", "lean48": "lean48"}[tag]; name = {"m48": "pass 48 (closed form)", "lean48": "pass 48 lean (closed form)"}[tag]
        if rel == "p32/p32_G2.json" and tag == "round2":
            r = "p32"
        add(name, "exploration", n, t[1], t[0], t100(r) if r else None)
# pass 49 pruned lean models (if present)
p49 = HERE / "p49" / "p49.json"
SHOW_P49 = False                                       # user 2026-10-27: closed-form pruned models off the front for now (49b kept)
if SHOW_P49 and p49.exists():
    for s in json.loads(p49.read_text())["steps"][1:]:
        add(f"pass 49 lean-{s['n']} (drop {s['dropped']})", "pruned", s["n"], s["T96"][1], s["T96"][0],
            t100(f"p49_n{s['n']}"))
p49b = HERE / "p49b" / "p49.json"                      # same pruning on pass 40 lean (module QNM / remnant, spline shapes)
if p49b.exists():
    for s in json.loads(p49b.read_text())["steps"][1:]:
        if s["n"] < 12:
            continue                                   # pass 49c (11..9) not on the front until reported (2026-10-28)
        add(f"pass 49b lean-{s['n']} (drop {s['dropped']})", "pruned", s["n"], s["T96"][1], s["T96"][0],
            t100(f"p49b_n{s['n']}"))


def _short(name):
    """Compact plot label: 'pass 49 lean-12 (drop ea0)' -> 'p49 lean-12', 'pass 48 lean (closed form)' -> 'p48 lean'."""
    import re
    name = re.sub(r"\s*\((drop \w+|closed form)\)", "", name)
    return re.sub(r"^pass (\d+)", r"p\1", name).replace(" round2", "").replace("flux_anc_spin_with_orbit", "flux+orbit").replace(" et al. ", " ")


def front(pts, key):
    """Non-dominated points: no other point has <= coefficients AND a lower error (strictly better in one)."""
    ok = [p for p in pts if p[key] is not None]
    return sorted([p for p in ok if not any((o["n"] <= p["n"] and o[key] < p[key]) and (o["n"] < p["n"] or o[key] < p[key]) for o in ok)],
                  key=lambda p: p["n"])


def main():
    out = dict(points=P, front_T96_median=[p["name"] for p in front(P, "T96_median")],
               front_T96_mean=[p["name"] for p in front(P, "T96_mean")], front_R100_median=[p["name"] for p in front(P, "R100_median")])
    (HERE / "pareto" / "pareto_data.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"published": "#e34948", "installed": "#8c8c8c", "split": "#eb6834", "exploration": "#2a78d6", "pruned": "#1baf7a"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, key, lab in zip(axs, ("T96_median", "T96_mean", "R100_median"),
                            ("training T96 median", "training T96 mean", "sealed test R100 median")):
        for fam, c in col.items():
            pp = [p for p in P if p["family"] == fam and p[key] is not None]
            ax.scatter([p["n"] for p in pp], [p[key] for p in pp], s=38, color=c, alpha=0.85, label=fam, zorder=3, edgecolors="white", linewidths=0.6)
        fr = front(P, key)
        ax.step([p["n"] for p in fr], [p[key] for p in fr], where="post", color="#000000", lw=1.2, zorder=2, label="Pareto front")
        ax.set_yscale("log"); ax.set_xlim(3, 36)
        label_front(ax, [(p["n"], p[key], _short(p["name"])) for p in fr])
        for p in P:                                    # published models: always labelled (rarely on the front)
            if p["family"] == "published" and p[key] is not None and p not in fr:
                ax.annotate(_short(p["name"]), (p["n"], p[key]), xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=8, color="#1a1a1a", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        ax.set_xlabel("fitted coefficients"); ax.set_ylabel(r"$\mathcal{E}$"); ax.set_title(lab, fontsize=11)
        ax.grid(True, color="#e6e6e6", lw=0.6, which="both"); ax.set_axisbelow(True)
    axs[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Spin calibration models: fitted coefficients vs full-window error (lower-left is better)", fontsize=12)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(HERE / "pareto" / f"pareto_front.{e}", dpi=170)
    for key in ("T96_median", "R100_median"):
        print(key, "front:", " -> ".join(f"{p['name']} ({p['n']}c, {p[key]:.2e})" for p in front(P, key)))
    print(len(P), "models")


if __name__ == "__main__":
    main()
