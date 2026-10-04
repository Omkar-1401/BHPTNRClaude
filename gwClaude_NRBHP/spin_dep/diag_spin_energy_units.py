"""Did the 2026-09-24 switch to E per TOTAL mass hurt the SPIN energy/flux models at low q,
as it did the nonspinning ones (q=2: 1.20e-3 -> 5.89e-3)?  Diagnosis only.

Small grid (4 q x 4 chi1), both models (energy_anchored_spin 8c, fluxanchored_spin 9c),
both E units, identical starts (the neutral seed and the shipped M-unit spin theta) and
identical time budget.  Scored on the small grid and at q=2 (held out; note q=2 is outside
BHPTNRSur2dq1e3's q>=3 training range, so it is an extrapolation of the INPUT too).
Nothing shipped is written: output diag_spin_energy_units.json only.

E per m1 = E per M / X1; the flux_hat drive is unit-invariant (peak-normalised).
"""
from __future__ import annotations
import json, multiprocessing as mp, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_fluxanchored_spin as FS
import fit_energy_anchored_spin as EA
import fit_scaling_gwr_energy_flux as FL

QS = [3.0, 4.2, 5.9487179487, 8.0]
CHIS = [-0.6, -0.2, 0.2, 0.6]
Q2_CHIS = [-0.6, -0.2, 0.0, 0.2, 0.6]
BUDGET_S = 25 * 60
OUT = HERE / "diag_spin_energy_units.json"
CASES = {}


def get(q, c, units):
    k = (q, c)
    if k not in CASES:
        CASES[k] = FS.build_case(q, c)          # M-unit E (cache verified)
    case = CASES[k]
    if units == "M":
        return case
    x1 = q / (1 + q); d = dict(case)
    d["losses"] = dict(case["losses"]); d["fit_losses"] = dict(case["fit_losses"])
    d["losses"]["e_oft"] = case["losses"]["e_oft"] / x1
    d["fit_losses"]["e_oft"] = case["fit_losses"]["e_oft"] / x1
    return d


def err(model, th, q, c, units, t0, full):
    case = get(q, c, units)
    if model == "EA":
        return EA.mismatch(th, q, c, case, t0)
    p = FS.params_at(q, c, th)
    if t0 is None:
        g = np.linspace(-250.0, 150.0, 81)
        t0 = float(g[int(np.argmin([FL.fast_mismatch(p, case, x, "fit")[0] for x in g]))])
    e, t0 = FL.fast_mismatch(p, case, t0, "fit")
    return (FL.fast_mismatch(p, case, t0, "full")[0], t0) if full else (e, t0)


def run(arg):
    model, units, tag, th0 = arg
    nuis = {}; t_start = time.time(); best = {"f": np.inf, "th": np.array(th0, float)}

    class Done(Exception):
        pass

    def obj(th):
        if time.time() - t_start > BUDGET_S:
            raise Done
        es = []
        for q in QS:
            for c in CHIS:
                e, nuis[(q, c)] = err(model, th, q, c, units, nuis.get((q, c)), False)
                es.append(e)
        f = float(np.mean(es))
        if f < best["f"]:
            best.update(f=f, th=np.array(th, float))
        return f

    try:
        minimize(obj, best["th"], method="Powell",
                 options={"maxiter": 400, "xtol": 1e-4, "ftol": 1e-6})
        st = "converged"
    except Done:
        st = "budget"
    th = best["th"]
    grid = [err(model, th, q, c, units, None, True)[0] for q in QS for c in CHIS]
    q2 = {f"{c:+.1f}": float(err(model, th, 2.0, c, units, None, True)[0]) for c in Q2_CHIS}
    return dict(model=model, units=units, start=tag, status=st, obj=best["f"],
                grid_median=float(np.median(grid)), grid_max=float(np.max(grid)),
                q2=q2, q2_median=float(np.median(list(q2.values()))), theta=th.tolist())


def main():
    for q in QS:
        for c in CHIS:
            get(q, c, "M")
    for c in Q2_CHIS:
        get(2.0, c, "M")
    shipped = {m: np.array(json.loads((HERE / d / "coeffs_nu.json").read_text())["theta"])
               for m, d in (("EA", "energy_anchored_spin_results"), ("FA", "fluxanchored_spin_results"))}
    seeds = {"EA": EA.seed_theta(), "FA": FS.seed_theta()}
    jobs = [(m, u, tag, th) for m in ("EA", "FA") for u in ("M", "m1")
            for tag, th in (("seed", seeds[m]), ("shipped_M", shipped[m]))]
    print(f"{len(jobs)} runs x {BUDGET_S/60:.0f} min", flush=True)
    res = []
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        for r in pool.imap_unordered(run, jobs):
            res.append(r); OUT.write_text(json.dumps(res, indent=1))
            print(f"{r['model']} {r['units']:<2} {r['start']:<9} {r['status']:<9} grid median "
                  f"{r['grid_median']:.3e} max {r['grid_max']:.3e} | q=2 median {r['q2_median']:.3e} "
                  + " ".join(f"{k}:{v:.2e}" for k, v in r["q2"].items()), flush=True)
    print("\nbest per (model, units) by objective:")
    for m in ("EA", "FA"):
        for u in ("M", "m1"):
            b = min((r for r in res if r["model"] == m and r["units"] == u), key=lambda r: r["obj"])
            print(f"  {m} {u:<2}  grid median {b['grid_median']:.3e}  q=2 median {b['q2_median']:.3e}"
                  f"  (start {b['start']})")


if __name__ == "__main__":
    main()
