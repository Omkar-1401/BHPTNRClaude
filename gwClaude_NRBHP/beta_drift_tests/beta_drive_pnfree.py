"""Route 2, PN-FREE variant: give beta an inspiral-weighted drive built from the
ppBHPT waveform itself, instead of E(t).

Why: E(t) is a merger-ringdown clock -- only 7% (q=3) / 14% (q=8) of its range
accumulates over the first 83% of the window -- so a single monotone drive with that
shape cannot supply an inspiral rise, and `sign(dbeta/dt) = sign(b_D)` is then forced
negative above q~4 by the merger region.  Route 1 (phase gauge) is closed: it does not
make the drift rise (q=5 got MORE negative).  So change the drive's SHAPE.

`pn_anchored` achieves a rising beta at all q using x = (M Omega)^(2/3), and the old
gated models used p_loss -- both PN-flavoured.  This script asks whether a PN-FREE
inspiral-weighted drive does the same job, which would keep the property that
`gwr_energy_stiff` is held as evidence for.

alpha is UNCHANGED (flux_hat, the best coordinate).  Only beta's drive changes:

    beta(t) = beta_PP * (1 + b_D * D(t))

Every D is normalised D(window start) = 0, D(merger) = 1, so `b_D` is directly the
fractional drift from start to merger and is comparable across drives (and to b_E,
whose natural scale is b_E*E_tot).

Drives (all monotone non-decreasing, enforced by a running max where needed):
    e_oft   baseline, the shipped cumulative radiated energy       [merger-weighted]
    phi     accumulated GW phase of the (2,2) mode                 [inspiral-weighted]
    lnw     log of the GW frequency                                [inspiral-weighted]
    t       raw time                                               [maximally so]

All four come from the waveform/grid alone -- no PN expressions anywhere.
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
DRIVES = ["e_oft", "phi", "lnw", "t"]


def _norm(x, t, i_m):
    """0 at window start, 1 at merger, monotone non-decreasing."""
    x = np.maximum.accumulate(np.asarray(x, float))       # enforce monotone
    x0 = x[0]; xm = x[i_m]
    if not np.isfinite(xm - x0) or abs(xm - x0) < 1e-300:
        return None
    return (x - x0) / (xm - x0)


def add_drives(case):
    t = np.asarray(case["t_bhpt"], float)
    h = np.asarray(case["h_bhpt"])
    i_m = int(np.argmax(np.abs(h)))

    ph = np.unwrap(np.angle(h))
    if ph[-1] < ph[0]:
        ph = -ph                                          # make it increasing
    w = np.abs(np.gradient(ph, t))
    w = np.maximum(w, 1e-12)

    raw = {"e_oft": np.asarray(case["losses"]["e_oft"], float),
           "phi": ph, "lnw": np.log(w), "t": t}
    info = {}
    for k, v in raw.items():
        d = _norm(v, t, i_m)
        if d is None:
            raise RuntimeError(f"drive {k} degenerate")
        case["losses"][f"D_{k}"] = d
        case["fit_losses"][f"D_{k}"] = np.interp(case["fit_t_bhpt"], t, d)
        # inspiral weighting: fraction of the drive's range done in the first 83%
        t83 = t[0] + 0.83 * (t[-1] - t[0])
        info[k] = float(np.interp(t83, t, d))
    return case, info, i_m


def shape(p, t_bhpt, losses, drive):
    a_pp, a_f, b_pp, b_D = p
    alpha = a_pp * (1.0 + a_f * losses["flux_hat"])
    beta = b_pp * (1.0 + b_D * losses[f"D_{drive}"])
    if np.any(beta <= 0) or not np.all(np.isfinite(beta)):
        return alpha, None, -1.0
    bc = G.creative.cumulative_trapezoid(beta, t_bhpt)
    tau = bc - float(np.interp(G.T_ANCHOR, t_bhpt, bc))
    return alpha, tau, float(np.min(np.diff(tau)))


def mism(p, case, t0_ref, drive, key="fit"):
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"], case["fit_t_nr"],
                               case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"], case["t_nr"],
                               case["h_nr"], case["losses"])
    a, tau, dmin = shape(p, tb, los, drive)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_ref
    f = lambda t0: GG.err_at_t0(t0, a, tau, hb, tn, hn)
    r = minimize_scalar(f, bounds=(t0_ref - 120.0, t0_ref + 120.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    return float(r.fun), float(r.x)


def fit(case, pref, t0_ref, drive):
    """Seeds kept as candidates, so the best can never be worse than a seed."""
    cands, best_seed = [], np.inf
    for bD in (-0.05, -0.02, 0.0, 0.02, 0.05):
        for s in (1.0, 0.98):
            p0 = np.array([pref[0] * s, pref[1], pref[2], bD], float)
            e0, _ = mism(p0, case, t0_ref, drive)
            cands.append((e0, p0.copy())); best_seed = min(best_seed, e0)
            r = minimize(lambda p: mism(p, case, t0_ref, drive)[0], p0,
                         method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-12})
            cands.append((float(r.fun), np.asarray(r.x, float).copy()))
    e_fit, p = min(cands, key=lambda c: c[0])
    if e_fit > best_seed + 1e-15:
        print(f"      !! optimiser never beat its best seed for {drive}", flush=True)
    full, _ = mism(p, case, t0_ref, drive, "full")
    return full, p


cache = json.load(open(str(HERE.parent) + "/gwr_energy_flux_results/per_q_cache_flux.json"))
res = {}
for q in QS:
    ref = min(cache.values(), key=lambda v: abs(v["q"] - q))
    pref = np.array(ref["params"], float); t0r = ref["t0"]
    case = FL.add_flux(G.load_case(q, 6, 10))
    case, info, i_m = add_drives(case)
    E_tot = float(case["losses"]["e_oft"][-1] - case["losses"]["e_oft"][0])
    print(f"\n=== q={q:g}   (shipped: E={ref['err']:.4e}, b_E={pref[3]:+.4f}, "
          f"dbeta={pref[3]*E_tot*100:+.2f}%)", flush=True)
    print("   inspiral weighting -- fraction of the drive's range done in the first "
          "83% of the window:")
    for k in DRIVES:
        print(f"      {k:6s} {info[k]*100:6.2f}%", flush=True)
    res[q] = {"info": info, "fits": {}}
    print(f"   {'drive':6s} {'full E':>12s} {'x shipped':>10s} {'b_D':>9s} "
          f"{'dbeta(->merger)':>16s}  drift")
    for k in DRIVES:
        e, p = fit(case, pref, t0r, k)
        sign = "RISING" if p[3] > 0 else ("flat" if p[3] == 0 else "falling")
        res[q]["fits"][k] = {"err": e, "params": p.tolist()}
        print(f"   {k:6s} {e:12.4e} {e/ref['err']:10.2f} {p[3]:+9.4f} "
              f"{p[3]*100:+15.2f}%  {sign}", flush=True)

print("\n\n==== SUMMARY: sign of beta's drift per drive ====")
print(f"{'drive':6s} " + "".join(f"{'q='+format(q,'g'):>22s}" for q in QS))
for k in DRIVES:
    row = ""
    for q in QS:
        f = res[q]["fits"][k]
        row += f"{f['err']:.3e} {f['params'][3]:+.3f}".rjust(22)
    print(f"{k:6s} " + row)
json.dump({str(k): v for k, v in res.items()},
          open(str(HERE / "beta_drive_pnfree.json"), "w"), indent=1)
