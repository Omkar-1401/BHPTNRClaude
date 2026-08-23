"""Cost of the MERGER-ALIGNED phase gauge (dphi=0) vs the L2-optimal analytic phi0.

Route 1 of the "make beta rise at all q" pipeline (RESUME_gwremnant.md).

Model unchanged (flux alpha, E-driven beta, 4 per-q params, t0 inner nuisance).
Only the phase convention changes:
   L2      :  Z = |S|                            S = sum(h_NR conj(g))   <- shipped
   aligned :  Z = Re(S e^{-i phi0}),  phi0 = angle(h_NR at the NR merger)
                                            - angle(g    at the same time)
Question: in the aligned gauge, does the fit want b_E > 0 (rising beta), and what
does the gauge cost in mismatch?

ATTEMPT 1 (2026-08-11) WAS INVALID -- kept as gauge_aligned_attempt1_invalid.* .  Two
defects, both fixed here:

  1. phi0 was read off `argmax|g|`, a parameter-DEPENDENT integer index that jumps as
     the parameters vary, so the objective was discontinuous and Nelder-Mead stalled.
     Now both phases are evaluated at the FIXED NR merger time -- `argmax|h_NR|` is NR
     data and does not depend on the parameters, so phi0 is continuous.
  2. the inner t0 seed was carried across objective calls in a mutating dict, making
     the objective non-deterministic.  Now t0 is bracketed around a fixed reference.

Guard against a repeat: every seed is itself kept as a candidate, so the reported best
can never be worse than a seed (that inequality is exactly what exposed attempt 1).
The L2 branch is untouched and reproduces the shipped per-q numbers, so it doubles as a
regression check.
"""
import sys, json, warnings
import numpy as np
from scipy.optimize import minimize, minimize_scalar
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL

QS = [3.0, 5.0, 8.0]
T0_HALF_WIDTH = 120.0          # fixed bracket, since t0 is no longer carried


def err_at_t0(t0, alpha, tau_shape, h_bhpt, t_nr, h_nr, gauge):
    """gauge='L2' -> analytic phi0 (shipped); 'align' -> merger-aligned phi0."""
    tau = t0 + tau_shape
    tmin = max(t_nr[0], tau[0]); tmax = min(t_nr[-1], tau[-1])
    if tmax - tmin < 0:
        return 50.0
    m = (t_nr >= tmin) & (t_nr <= tmax)
    if m.sum() / len(t_nr) < G.MIN_COVERAGE:
        return 50.0
    tc = t_nr[m]; href = h_nr[m]
    g = np.interp(tc, tau, h_bhpt.real) + 1j * np.interp(tc, tau, h_bhpt.imag)
    g = np.interp(tc, tau, alpha) * g
    n1 = np.sum(np.abs(href) ** 2); n2 = np.sum(np.abs(g) ** 2)
    S = np.sum(href * np.conj(g))
    if gauge == "L2":
        Z = np.abs(S)
    else:
        # phi0 from the NR merger time.  argmax|h_NR| is NR data -> parameter
        # independent -> phi0 continuous in the parameters (this is the attempt-1 fix).
        ipk = int(np.argmax(np.abs(h_nr)))
        t_pk = float(t_nr[ipk])
        if not (tau[0] <= t_pk <= tau[-1]):
            return 50.0
        g_pk = float(np.interp(t_pk, tau, alpha)) * (
            float(np.interp(t_pk, tau, h_bhpt.real))
            + 1j * float(np.interp(t_pk, tau, h_bhpt.imag)))
        if abs(g_pk) <= 0.0:
            return 50.0
        phi0 = np.angle(h_nr[ipk]) - np.angle(g_pk)
        Z = np.real(S * np.exp(-1j * phi0))
    return float((n1 + n2 - 2.0 * Z) / (2.0 * n1))


def mism(p, case, t0_ref, gauge, key="fit"):
    """Deterministic: t0 is bracketed around the FIXED t0_ref, never carried."""
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"], case["fit_t_nr"],
                               case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"], case["t_nr"],
                               case["h_nr"], case["losses"])
    a, tau, dmin = FL.model_shape(p, tb, los)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_ref
    f = lambda t0: err_at_t0(t0, a, tau, hb, tn, hn, gauge)
    r = minimize_scalar(f, bounds=(t0_ref - T0_HALF_WIDTH, t0_ref + T0_HALF_WIDTH),
                        method="bounded", options={"xatol": 1e-3, "maxiter": 80})
    return float(r.fun), float(r.x)


def fit(case, pref, t0_ref, gauge, be_seeds):
    """Multistart Nelder-Mead.  Seeds are themselves candidates, so the returned best
    is <= every seed by construction."""
    cands, best_seed = [], np.inf
    for be in be_seeds:
        for s in (1.0, 0.98):
            p0 = np.array([pref[0] * s, pref[1], pref[2], be], float)
            e0, _ = mism(p0, case, t0_ref, gauge, "fit")
            cands.append((e0, p0.copy()))
            best_seed = min(best_seed, e0)
            r = minimize(lambda p: mism(p, case, t0_ref, gauge, "fit")[0], p0,
                         method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-12})
            cands.append((float(r.fun), np.asarray(r.x, float).copy()))
    e_fit, p = min(cands, key=lambda c: c[0])
    if e_fit > best_seed + 1e-15:
        print(f"      !! optimiser never beat its best seed ({e_fit:.4e} > "
              f"{best_seed:.4e}) -- objective still ill-behaved", flush=True)
    full, t0f = mism(p, case, t0_ref, gauge, "full")
    return full, p, t0f, e_fit, best_seed


cache = json.load(open(str(HERE.parent) + "/gwr_energy_flux_results/per_q_cache_flux.json"))
res = {}
for q in QS:
    ref = min(cache.values(), key=lambda v: abs(v["q"] - q))
    pref = np.array(ref["params"], float); t0r = ref["t0"]
    case = FL.add_flux(G.load_case(q, 6, 10))
    Etot = float(case["losses"]["e_oft"][-1] - case["losses"]["e_oft"][0])

    eL2, pL2, t0L2, _, _ = fit(case, pref, t0r, "L2", [pref[3], -1.0, 1.0])
    eAL, pAL, t0AL, _, seedAL = fit(case, pref, t0r, "align",
                                    [pref[3], -1.0, 0.0, 1.0, 3.0])
    # cross-scorings separate "the gauge costs" from "the fit found worse parameters"
    eAL_inL2, _ = mism(pAL, case, t0r, "L2", "full")
    eL2_inAL, _ = mism(pL2, case, t0r, "align", "full")

    res[q] = dict(eL2=eL2, beL2=pL2[3], eAL=eAL, beAL=pAL[3],
                  eAL_inL2=eAL_inL2, eL2_inAL=eL2_inAL, Etot=Etot,
                  pL2=pL2.tolist(), pAL=pAL.tolist())
    print(f"q={q}", flush=True)
    print(f"   L2 gauge   (shipped): E={eL2:.4e}  b_E={pL2[3]:+.4f} "
          f"-> dbeta={pL2[3]*Etot*100:+.2f}%")
    print(f"   aligned gauge       : E={eAL:.4e}  b_E={pAL[3]:+.4f} "
          f"-> dbeta={pAL[3]*Etot*100:+.2f}%   (gauge cost x{eAL/eL2:.2f})")
    print(f"   aligned params, rescored with optimal phi0: E={eAL_inL2:.4e} "
          f"(x{eAL_inL2/eL2:.2f} vs L2 best)")
    print(f"   L2 params,      rescored in aligned gauge : E={eL2_inAL:.4e}")
    # the attempt-1 tell: a legitimate fit must not be beaten by rescoring L2 params
    if eAL > eL2_inAL:
        print(f"      !! INVALID: aligned fit ({eAL:.4e}) is worse than the L2 "
              f"params rescored in the same gauge ({eL2_inAL:.4e})", flush=True)
    else:
        print(f"      ok: aligned fit beats L2-params-in-aligned-gauge", flush=True)

print(f"\n{'q':>4s} {'L2 E':>11s} {'b_E':>8s} | {'aligned E':>11s} {'b_E':>8s} "
      f"{'dbeta':>8s} {'x cost':>7s}")
for q in QS:
    r = res[q]
    print(f"{q:4.1f} {r['eL2']:11.4e} {r['beL2']:+8.3f} | {r['eAL']:11.4e} "
          f"{r['beAL']:+8.3f} {r['beAL']*r['Etot']*100:+7.2f}% {r['eAL']/r['eL2']:7.2f}")
signs = {q: ("RISING" if res[q]["beAL"] > 0 else
             "flat" if res[q]["beAL"] == 0 else "falling") for q in QS}
print("\naligned-gauge beta drift:", ", ".join(f"q={q:g} {s}" for q, s in signs.items()))
json.dump({str(k): v for k, v in res.items()},
          open(str(HERE / "gauge_aligned.json"), "w"), indent=1)
