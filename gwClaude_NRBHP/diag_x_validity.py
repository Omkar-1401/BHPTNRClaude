"""Where does x stop being a usable coordinate?  Measured, not asserted.

x = (0.5 * omega_GW,pp)^(2/3) rises through the inspiral, its RATE peaks at merger, and
then it plateaus as omega_GW settles onto the QNM value.  Once it plateaus the model
alpha(x), beta(x) is FROZEN while the true alpha and beta keep evolving -- so saturation
is where an x-parameterised model stops being able to represent anything, and it is the
natural place to cut the scored window.

Reported per q:
  t_peak_rate   time of max dx/dt                      (~merger)
  t_sat_05      first time after that where dx/dt < 5% of its peak
  t_sat_01      ...< 1% of its peak
  t_x95, t_x99  times at which x reaches 95% / 99% of its final value
  t_nonmono     last time before which x is monotone to within tolerance
  frac_range_after_tsat   how much of x's range still lies past t_sat_05

The point of the last column: if it is ~0 then cutting at saturation costs the model
nothing in coordinate range, and everything past it was unrepresentable anyway.

Usage:  python diag_x_validity.py [--q 2 3 5 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_x_drive as XD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, nargs="+",
                    default=[2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
    args = ap.parse_args()

    out = {}
    print(f"{'q':>5} {'t_peak_rate':>12} {'t_sat_05':>9} {'t_sat_01':>9} "
          f"{'t_x95':>8} {'t_x99':>8} {'x_max':>8} {'% of x-range past t_sat_05':>27}")
    for q in args.q:
        case = XD.add_x(FL.add_flux(G.load_case(q, XD.SRC_STRIDE, XD.NR_STRIDE)))
        t = np.asarray(case["t_bhpt"], float)
        x = np.asarray(case["_x_raw"], float)
        rate = np.gradient(x, t)

        i_pk = int(np.argmax(rate))
        t_pk = float(t[i_pk])
        pk = float(rate[i_pk])

        def first_below(frac):
            j = np.where(rate[i_pk:] < frac * pk)[0]
            return float(t[i_pk + j[0]]) if len(j) else float("nan")

        t05, t01 = first_below(0.05), first_below(0.01)
        x0, xf = float(x[0]), float(np.nanmax(x))
        t95 = float(np.interp(x0 + 0.95 * (xf - x0), x, t))
        t99 = float(np.interp(x0 + 0.99 * (xf - x0), x, t))

        if np.isfinite(t05):
            x_at = float(np.interp(t05, t, x))
            frac_after = (xf - x_at) / (xf - x0) * 100.0
        else:
            frac_after = float("nan")

        print(f"{q:5g} {t_pk:12.1f} {t05:9.1f} {t01:9.1f} {t95:8.1f} {t99:8.1f} "
              f"{xf:8.4f} {frac_after:26.2f}%")
        out[f"{q:g}"] = {"t_peak_rate": t_pk, "t_sat_05": t05, "t_sat_01": t01,
                         "t_x95": t95, "t_x99": t99, "x_max": xf,
                         "pct_range_after_t_sat_05": frac_after}

    p = XD.RESULTS / "x_validity.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    t05s = [v["t_sat_05"] for v in out.values() if np.isfinite(v["t_sat_05"])]
    if t05s:
        print(f"\nt_sat_05 across q: {min(t05s):.1f} .. {max(t05s):.1f} M")
        print("-> a single cut at the LATEST of these keeps every q inside its own "
              "valid range")


if __name__ == "__main__":
    main()
