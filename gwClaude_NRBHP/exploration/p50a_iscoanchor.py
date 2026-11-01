"""EXPLORATION pass 50a (diagnostic, login-node arithmetic) -- B4: does an ISCO-frequency clock ratio predict the fitted
plunge-clock offsets r1, r3?  ln beta_M = ln beta_r + g (r1 chi1 + r3 chi1^2) (pass 49b lean-12: r1 +0.0663, r3 -0.0748).
Candidate anchors, beta_isco = Omega_isco,pp [per m1] / Omega_isco,NR [per M] (frequency matching at the ISCO):
    pp side: Kerr ISCO of the primary, M Omega_isco(chi1)
    NR side  A: Kerr ISCO of the remnant, Omega_isco(chi_f) / M_f
             B: Kerr ISCO at the binary's effective spin X1^2 chi1, total mass M
             C: B with the Schwarzschild conservative self-force shift (1 + 1.2513 nu) (Barack-Sago / Favata)
Compared as d(chi1) = ln beta_isco - ln beta_r, minus its chi1 = 0 value (the fitted offset vanishes at chi1 = 0).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physics_common as PC                      # exact Kerr w220 (light; the spin_dep fit module loads the surrogates)
import surfinBH
SF = surfinBH.LoadFits("NRSur3dq8Remnant")


def beta_r(q, c):                                 # = fit_pn_anchored_spin.beta_r_phys
    mf, _ = SF.mf(q, [0, 0, c], [0, 0, 0]); chif, _ = SF.chif(q, [0, 0, c], [0, 0, 0])
    return PC.w220_exact(c) * float(mf) / PC.w220_exact(float(chif[2])), float(mf), float(chif[2])


def om_isco(a):
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3)); z2 = np.sqrt(3 * a * a + z1 * z1)
    r = 3 + z2 - np.sign(a) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return 1 / (r ** 1.5 + a)


CH = np.round(np.arange(-0.6, 0.81, 0.2), 2)
fit = 0.0663 * CH - 0.0748 * CH ** 2
print("chi1        ", " ".join(f"{c:+7.2f}" for c in CH))
print("fitted r1,r3", " ".join(f"{v:+7.4f}" for v in fit))
for q in (3.0, 5.0, 8.0):
    nu = q / (1 + q) ** 2; X1 = q / (1 + q)
    rows = {"A remnant": [], "B X1^2chi": [], "C B+SF": []}
    for c in CH:
        br, mf, chif = beta_r(q, c); lbr = np.log(br); pp = om_isco(c)
        rows["A remnant"].append(np.log(pp * mf / om_isco(chif)) - lbr)
        rows["B X1^2chi"].append(np.log(pp / om_isco(X1 ** 2 * c)) - lbr)
        rows["C B+SF"].append(np.log(pp / (om_isco(X1 ** 2 * c) * (1 + 1.2513 * nu))) - lbr)
    for k, v in rows.items():
        v = np.array(v); d = v - v[CH == 0][0]
        b = np.polyfit(d, fit, 1) if np.ptp(d) > 0 else (np.nan, np.nan)
        print(f"q={q:.0f} {k:9s}", " ".join(f"{x:+7.4f}" for x in d), f"| level {v[CH == 0][0]:+.4f} | fit/cand slope {np.dot(d, fit) / np.dot(d, d):+.2f}, rms resid {np.sqrt(np.mean((fit - d * np.dot(d, fit) / np.dot(d, d)) ** 2)):.4f}")
