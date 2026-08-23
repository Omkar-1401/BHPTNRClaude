"""What does the INSPIRAL ALONE want beta's slope to be?

Every drift measurement so far (scan_bE, monotone_beta, beta_drive_pnfree) scored the
mismatch over the whole window or start->merger, and E(t) does 66-82% of its range in
the last 2% of the window.  So the recovered sign of `b_E` could be set entirely by the
merger-ringdown region, with the inspiral having no say.

User requirement (2026-08-11, refined): beta must be uniformly INCREASING at least
through the INSPIRAL.  So the question that matters is the inspiral's own preference.

Test: refit the 4 per-q parameters with the NR comparison window TRUNCATED at T_CUT
(default -200 M), i.e. the merger and ringdown are simply not scored.  Multistart from
both signs; seeds kept as candidates so the reported best can't be worse than a seed.

Read-out:
  b_E > 0  -> the inspiral WANTS a rising beta; the negative sign seen in the full-window
              fits is imposed by the merger region.  A two-term beta (one term for the
              inspiral, one to absorb the merger) then buys a rising inspiral cheaply.
  b_E < 0  -> the inspiral wants a falling beta too, so the requirement conflicts with
              the data on its own turf and the 2x-4.5x cost is irreducible.

Also reports, for each q, the full-window optimum for reference, and how much of the
inspiral-only mismatch is given up by forcing b_E to the opposite sign.
"""
import sys, json, warnings
import numpy as np
from scipy.optimize import minimize, minimize_scalar
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG

QS = [3.0, 5.0, 8.0]
T_CUTS = [-200.0, -500.0]


def truncate(case, t_cut):
    """Copy of the case with the NR arrays cut at t_cut (BHPT side untouched)."""
    c = dict(case)
    for tk, hk in (("t_nr", "h_nr"), ("fit_t_nr", "fit_h_nr")):
        t = np.asarray(case[tk], float); h = np.asarray(case[hk])
        m = t <= t_cut
        c[tk] = t[m]; c[hk] = h[m]
    return c


def mism(p, case, t0_ref, key="fit"):
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"], case["fit_t_nr"],
                               case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"], case["t_nr"],
                               case["h_nr"], case["losses"])
    a, tau, dmin = FL.model_shape(p, tb, los)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_ref
    f = lambda t0: GG.err_at_t0(t0, a, tau, hb, tn, hn)
    r = minimize_scalar(f, bounds=(t0_ref - 120.0, t0_ref + 120.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    return float(r.fun), float(r.x)


def fit(case, pref, t0_ref, be_seeds, fix_be=None):
    """Optimise the 4 params (or 3, with b_E fixed).  Seeds are candidates too."""
    cands, best_seed = [], np.inf
    for be in ([fix_be] if fix_be is not None else be_seeds):
        for s in (1.0, 0.98):
            if fix_be is None:
                p0 = np.array([pref[0] * s, pref[1], pref[2], be], float)
                obj = lambda p: mism(p, case, t0_ref)[0]
                x0 = p0
            else:
                x0 = np.array([pref[0] * s, pref[1], pref[2]], float)
                obj = lambda u: mism(np.r_[u, fix_be], case, t0_ref)[0]
            e0 = obj(x0)
            cands.append((e0, np.asarray(x0, float).copy()))
            best_seed = min(best_seed, e0)
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-12})
            cands.append((float(r.fun), np.asarray(r.x, float).copy()))
    e, x = min(cands, key=lambda c: c[0])
    if e > best_seed + 1e-15:
        print("      !! optimiser never beat its best seed", flush=True)
    p = np.r_[x, fix_be] if fix_be is not None else x
    return e, p


cache = json.load(open(str(HERE.parent) + "/gwr_energy_flux_results/per_q_cache_flux.json"))
res = {}
for q in QS:
    ref = min(cache.values(), key=lambda v: abs(v["q"] - q))
    pref = np.array(ref["params"], float); t0r = ref["t0"]
    case = FL.add_flux(G.load_case(q, 6, 10))
    e_full = ref["err"]; be_full = pref[3]
    E_tot = float(case["losses"]["e_oft"][-1] - case["losses"]["e_oft"][0])
    print(f"\n=== q={q:g}   full-window optimum: E={e_full:.4e}  b_E={be_full:+.4f}",
          flush=True)
    res[q] = {"full": {"err": e_full, "b_E": be_full}, "cuts": {}}
    for tc in T_CUTS:
        tcase = truncate(case, tc)
        n = len(tcase["t_nr"]); ntot = len(case["t_nr"])
        e, p = fit(tcase, pref, t0r, [be_full, -1.0, 0.0, 1.0, 3.0])
        # cost of forcing the opposite sign, inspiral-only
        e_opp, _ = fit(tcase, pref, t0r, [], fix_be=(-abs(p[3]) if p[3] > 0
                                                     else abs(p[3])))
        sign = "RISING" if p[3] > 0 else ("flat" if p[3] == 0 else "falling")
        # how much of E(t) has accumulated by the cut
        Eat = float(np.interp(tc, case["t_bhpt"], case["losses"]["e_oft"]))
        res[q]["cuts"][tc] = {"err": e, "b_E": float(p[3]), "err_opp_sign": e_opp,
                              "n_nr": n, "frac_window": n / ntot, "E_frac": Eat / E_tot}
        print(f"   cut t<={tc:+7.0f} M  ({n/ntot*100:5.1f}% of NR samples, "
              f"{Eat/E_tot*100:5.1f}% of E's range): "
              f"E={e:.4e}  b_E={p[3]:+.4f}  {sign}   "
              f"[opposite sign forced: {e_opp:.4e}, x{e_opp/e:.2f}]", flush=True)

print("\n\n==== SUMMARY: sign of beta's inspiral slope ====")
print(f"{'q':>4s} {'full-window b_E':>16s} " +
      "".join(f"{'b_E (t<=' + format(tc, '+.0f') + ')':>20s}" for tc in T_CUTS))
for q in QS:
    row = "".join(f"{res[q]['cuts'][tc]['b_E']:+20.4f}" for tc in T_CUTS)
    print(f"{q:4.1f} {res[q]['full']['b_E']:+16.4f} " + row)
json.dump({str(k): v for k, v in res.items()},
          open(str(HERE / "inspiral_only_sign.json"), "w"), indent=1, default=float)
