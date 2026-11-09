"""Per-mode phase consistency (diagnostic, no outputs): after the (2,2) fixes an orbital rotation phi = arg z22 / 2
(mod pi), a mode (l, m) should need a constant phase m * phi (mod 2 pi).  Prints the residual
d_lm = arg z_lm - m arg z22 / 2 (mod 2 pi; for odd m also + pi, the other rotation branch), the mode's power share and
its free-phase error.  z_lm = sum(nr_lm conj(model_lm)) on the common window (flat weight).
    python fdm/hm_phase_diag.py MODEL
"""
import sys
import numpy as np
import hm_run as HR
for k in [(3.0, 0.0), (5.0, 0.0), (8.0, 0.0), (5.0, 0.6), (5.0, -0.6)]:
    tc, nr, mod = HR.maps(k)
    z = {mo: np.sum(nr[mo] * np.conj(mod[mo])) for mo in nr}
    P = {mo: np.sum(np.abs(nr[mo]) ** 2) for mo in nr}; Pt = sum(P.values())
    t22 = np.angle(z[(2, 2)])
    w = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    line = []
    for (l, m) in nr:
        d = w(np.angle(z[(l, m)]) - m * t22 / 2)
        alt = w(d + np.pi) if m % 2 else d
        dd = d if abs(d) <= abs(alt) else alt
        n1, n2 = P[(l, m)], np.sum(np.abs(mod[(l, m)]) ** 2)
        e = (n1 + n2 - 2 * abs(z[(l, m)])) / (2 * n1)
        line.append(f"({l},{m}) d={dd:+.2f} P={P[(l, m)] / Pt:.1e} E={e:.1e}")
    print(HR.MODEL, k, " | ".join(line), flush=True)
