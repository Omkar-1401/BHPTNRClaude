"""Summary of the training-size campaign: results/*.json -> SUMMARY.md + summary.json.

Per (model, grid) the reported run is the start with the lowest TRAINING mean full-window E (training data only;
V300 never enters the choice).  Fresh-G96 reproduction: the G96 refits against the shipped model's stored T96 numbers.

    python summarize.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
RES = HERE / "results"
MODELS = {"pn13": "pn_anchored_spin (13c, 10 refit)", "flux_orbit": "flux_anc_spin_with_orbit (9c)",
          "splitA": "split A (14c)", "splitB": "split B (13c + window)", "p39L4": "pass 39 L4 (18c)",
          "p40": "pass 40 (18c; L4 + closed-form R2)", "lean": "lean, pass 42 K1 (16c; OT onset)",
          "lean12": "lean-12, pass 49b (12c; pruned on T96)", "lean9": "lean-9, pass 49c (9c; pruned on T96)",
          "rink": "Rink et al. form (12c)"}
GRIDS = (24, 48, 96, 192)


def st(v):
    v = np.asarray(v, float); return dict(n=int(v.size), median=float(np.median(v)), mean=float(v.mean()), max=float(v.max()))


def shipped_T96(model):
    """The shipped model's stored per-case T96 values, from the file each harness gates against."""
    j = lambda p: json.loads((REPO / p).read_text())
    if model == "pn13":
        return list(j("spin_dep/pn_anchored_spin_results/coeffs_spin.json")["mismatches"].values())
    if model == "flux_orbit":
        return list(j("spin_dep/flux_anc_spin_with_orbit_results/coeffs.json")["mismatches"].values())
    if model == "splitA":
        return [r["full_E_full"] for r in j("spin_dep_split/results/explore/slim__win-2.0.json")["cases"]["T96"]]
    if model == "splitB":
        return [r["full_E_full"] for r in j("spin_dep_split/results/cv/tline__foldall.json")["train_rows"]]
    if model == "p39L4":
        return list(j("exploration/p39/p39_L4.json")["T96_cases"].values())
    if model == "p40":
        return list(j("exploration/p40/p40.json")["O2n"]["T96_cases"].values())
    if model == "lean":
        return list(j("exploration/p42/p42_K1.json")["T96_cases"].values())
    if model in ("lean12", "lean9"):
        n = int(model[4:])
        return list([s for s in j("exploration/p49b/p49.json")["steps"] if s["n"] == n][0]["T96_cases"].values())
    if model == "rink":
        return [r["full_E_full"] for r in j("spin_dep_split/results/rink_scores.json")["T96"]]


def f3(s):
    return f"{s['median']:.2e} / {s['mean']:.2e} / {s['max']:.1e}"


def main():
    runs = {}
    for p in sorted(RES.glob("*__G*__*.json")):
        d = json.loads(p.read_text()); runs.setdefault((d["model"], d["grid"]), []).append(d)
    out, L = {}, []
    L += ["# Training-size study: summary", "",
          "median / mean / max of full-window E (full alignment).  Reported run per grid = lowest TRAINING mean over starts.",
          "V300 = 300 fresh uniform points (score only).  'shipped' = the model as installed (fit on T96).", ""]
    # 1. fresh-G96 reproduction
    L += ["## 1. Fresh G96 refit vs shipped (T96 per-case numbers)", "",
          "| model | shipped T96 | G96 refit T96 (chosen start) | ratio median / mean / max | starts: train mean | per-case refit/shipped |",
          "|:--|:--|:--|:--|:--|:--|"]
    for m in MODELS:
        if (m, 96) not in runs:
            continue
        rs = runs[(m, 96)]; best = min(rs, key=lambda d: d["stats"]["train"]["mean"])
        sh = st(shipped_T96(m)); rf = best["stats"]["T96"]
        sm = {k: v for k, v in best["per_case"]["T96"].items()}
        rat = [rf[x] / sh[x] for x in ("median", "mean", "max")]
        starts = ", ".join(f"{d['start']} {d['stats']['train']['mean']:.4e}" for d in sorted(rs, key=lambda d: d["start"]))
        ref = np.array(shipped_T96(m)); per = np.sort(np.array(list(sm.values())))
        L.append(f"| {MODELS[m]} | {f3(sh)} | {f3(rf)} | {rat[0]:.3f} / {rat[1]:.3f} / {rat[2]:.3f} | {starts} | "
                 f"sorted-value ratio {np.min(per / np.sort(ref)):.3f}..{np.max(per / np.sort(ref)):.3f} |")
        out.setdefault(m, {})["repro96"] = dict(shipped=sh, refit=rf, ratio=rat, chosen=best["start"])
    # 2. learning curve
    L += ["", "## 2. V300 vs training size (chosen start per grid)", "",
          "| model | shipped (T96) | G24 | G48 | G96 | G192 |", "|:--|:--|:--|:--|:--|:--|"]
    for m in MODELS:
        if not any((m, g) in runs for g in GRIDS):
            continue
        cells, row = [], {}
        for g in GRIDS:
            if (m, g) not in runs:
                cells.append("-"); continue
            rs = runs[(m, g)]; best = min(rs, key=lambda d: d["stats"]["train"]["mean"])
            row[g] = dict(chosen=best["start"], V300=best["stats"]["V300"], train=best["stats"]["train"],
                          T96=best["stats"]["T96"], starts={d["start"]: dict(train=d["stats"]["train"], V300=d["stats"]["V300"])
                                                          for d in rs})
            cells.append(f3(best["stats"]["V300"]) + ("" if best["start"] == "t96" else f" ({best['start']})"))
        shipv = next(iter(runs.get((m, 96), [{}])))["stats"].get("V300_shipped") or next(iter(runs.get((m, 96), [{}])))["stats"].get("V300_published")
        L.append(f"| {MODELS[m]} | {f3(shipv)} | " + " | ".join(cells) + " |")
        out.setdefault(m, {}).update(curve=row, V300_shipped=shipv)
    # 3. start dependence
    L += ["", "## 3. Start dependence: V300 median per start (train mean in brackets)", "",
          "| model | grid | t96 | pert1 | pert2 |", "|:--|:--|:--|:--|:--|"]
    for m in MODELS:
        for g in GRIDS:
            if (m, g) not in runs:
                continue
            by = {d["start"]: d for d in runs[(m, g)]}
            L.append(f"| {m} | {g} | " + " | ".join(
                f"{by[s]['stats']['V300']['median']:.3e} ({by[s]['stats']['train']['mean']:.3e})" if s in by else "-"
                for s in ("t96", "pert1", "pert2")) + " |")
    (HERE / "SUMMARY.md").write_text("\n".join(L) + "\n")
    (HERE / "summary.json").write_text(json.dumps(out, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
