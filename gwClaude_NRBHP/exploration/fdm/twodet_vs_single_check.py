"""One-off diagnostic: at iota = 0, how do the single-detector (fd_core, kappa-averaged) and two-detector (twodet_core)
matches differ, and how much do (3,2), (4,2) add?  python fdm/twodet_vs_single_check.py MODEL"""
import numpy as np
import hm_run as HR, fd_core as FC, twodet_core as TD
for k in [(5.0, 0.0), (3.0, -0.6), (8.0, 0.6)]:
    tc, nr, mod = HR.maps(k)
    two = lambda modes, fm: TD.match2(tc, {m: nr[m] for m in modes}, {m: mod[m] for m in modes}, 100.0, fmin=fm)["mm"]
    one = FC.match(tc, nr[(2, 2)], mod[(2, 2)], 100.0)
    print(f"{HR.MODEL} {k} M=100: single-det (2,2) 20Hz {one['mm']:.3e} (per kappa {np.round(one['mm_delta'],7).tolist()}) | "
          f"two-det (2,2) f20 {two([(2,2)], 20.0):.3e} rink {two([(2,2)], 'rink'):.3e} | "
          f"two-det all m=2 f20 {two(list(mod), 20.0):.3e} rink {two(list(mod), 'rink'):.3e}", flush=True)
