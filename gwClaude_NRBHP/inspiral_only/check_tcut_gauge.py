"""How wrong are the `*_below_cut` diagnostics, which mix the NR and BHPT time axes?

`fit_scaling_inspiral_fluxanchored.fit_one_q` computes three diagnostics as

    e_at_cut   = np.interp(t_cut, case["t_bhpt"], los["e_oft"])
    m          = case["t_bhpt"] <= t_cut
    drive_span = ptp(los[drive][m])

but `t_cut` is an **NR-side** threshold -- `truncate()` applies it to `t_nr` -- while
`t_bhpt` is the BHPT time axis, and the two are related by the fitted time map
`tau = t0_nr + integral beta dt`, with beta ~ 0.72-0.87.  So `t_bhpt = -200` and
`t_nr = -200` are NOT the same instant, and `E_frac_at_cut`, `beta_rise_to_cut` and
`drive_span_below_cut` are evaluated at the wrong point.

THE FIT ITSELF IS UNAFFECTED: `truncate()` cuts `t_nr`/`fit_t_nr` correctly, so every
b_E, mismatch and cost ratio in the caches, logs and md files stands.  Only these three
reported numbers are mixed-gauge.  This script prints both versions so the size of the
error is on record.

    python check_tcut_gauge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG
import fit_scaling_inspiral_fluxanchored as IN

from scipy.optimize import minimize_scalar

T_CUT = IN.DEFAULT_T_CUT


def run(model, qs=(3.0, 5.0, 8.0)):
    drive = IN.MODELS[model]["drive"]
    cache = {round(float(r["q"]), 10): r
             for r in __import__("json").loads(IN.paths_for(model)[0].read_text()).values()
             if abs(r["t_cut"] - T_CUT) < 1e-9}
    t0s = {k: v[4] for k, v in IN.mult_seeds().items()}

    print(f"\n=== model=inspiral_{model}, drive={drive}, t_cut={T_CUT:g} M ===")
    print(f"{'q':>5} {'t_bhpt @ t_nr=cut':>18} {'E frac AS-IS':>13} {'E frac FIXED':>13} "
          f"{'dbeta AS-IS':>12} {'dbeta FIXED':>12} {'span AS-IS':>11} {'span FIXED':>11}")
    for q in qs:
        k = min(cache, key=lambda g: abs(g - q))
        row = cache[k]
        case = FL.add_flux(G.load_case(k, IN.SRC_STRIDE, IN.NR_STRIDE))
        p = np.asarray(row["inspiral"]["params"], float)
        los, tb = case["losses"], case["t_bhpt"]

        # the fitted time map for the inspiral solution, t0 re-derived as in the plots
        alpha, tau_shape, _ = IN.model_shape(p, tb, los, drive)
        tc = IN.truncate(case, T_CUT)
        r = minimize_scalar(
            lambda t0: GG.err_at_t0(t0, alpha, tau_shape, case["h_bhpt"],
                                    tc["t_nr"], tc["h_nr"]),
            bounds=(t0s[k] - 120.0, t0s[k] + 120.0), method="bounded",
            options={"xatol": 1e-3, "maxiter": 80})
        tau = float(r.x) + tau_shape

        # BHPT time whose IMAGE under the map is the NR-side cut
        t_bhpt_at_cut = float(np.interp(T_CUT, tau, tb))

        e_tot = float(los["e_oft"][-1] - los["e_oft"][0])
        e_asis = float(np.interp(T_CUT, tb, los["e_oft"]))          # as shipped
        e_fix = float(np.interp(t_bhpt_at_cut, tb, los["e_oft"]))    # correct gauge

        m_asis = tb <= T_CUT
        m_fix = tb <= t_bhpt_at_cut
        s_asis = float(np.ptp(los[drive][m_asis]))
        s_fix = float(np.ptp(los[drive][m_fix]))

        print(f"{k:5.2f} {t_bhpt_at_cut:18.2f} {e_asis/e_tot:13.4f} {e_fix/e_tot:13.4f} "
              f"{p[3]*e_asis*100:+11.3f}% {p[3]*e_fix*100:+11.3f}% "
              f"{s_asis:11.4f} {s_fix:11.4f}")


if __name__ == "__main__":
    for m in ("fluxanchored", "anchored"):
        run(m)
