"""HISTORICAL DIAGNOSTIC (2026-08-30).  Both functions below keep the (1+q)/q ON PURPOSE:
they compare two variants of the OLD formula.  That factor was later shown to be wrong
(UNITS_AUDIT.md A1, 2026-09-24).  Do not reuse these as beta_r."""
"""Is beta_r's spin dependence right?  beta_r = omega_QNM(pp side) / omega_QNM(NR side).

In ppBHPT the 'remnant' IS the primary, spin chi1 (the secondary is negligible), so the
pp-side QNM must use chi1 -- NOT Schwarzschild.  The nonspinning code hardcodes 0.3683,
which is exactly QF1 + QF2*(1-0)^QF3, i.e. the chi=0 value of the same fitting function.
"""
import sys, numpy as np
sys.path.insert(0, "/work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP")
sys.path.insert(0, "/work2/11702/omkarsridhar1401/stampede3/BHPTutils")
import surfinBH
_S = surfinBH.LoadFits("NRSur3dq8Remnant")
QF1, QF2, QF3 = 1.5251, -1.1568, 0.1292
w220 = lambda chi: QF1 + QF2 * (1.0 - chi) ** QF3
print(f"w220(0) = {w220(0.0):.4f}   (the hardcoded 0.3683 -> confirms the same fit)\n")

def old(q, chi1):
    mf, _ = _S.mf(q, [0,0,chi1], [0,0,0]); chif, _ = _S.chif(q, [0,0,chi1], [0,0,0])
    return (0.3683*(1+q)/q) / (w220(float(chif[2]))/float(mf))

def new(q, chi1):
    mf, _ = _S.mf(q, [0,0,chi1], [0,0,0]); chif, _ = _S.chif(q, [0,0,chi1], [0,0,0])
    return (w220(chi1)*(1+q)/q) / (w220(float(chif[2]))/float(mf))

print(f"{'q':>4} {'chi1':>6} {'chi_f':>7} {'M_f':>6} {'beta_r OLD':>11} {'beta_r NEW':>11} "
      f"{'*r0 OLD':>9} {'*r0 NEW':>9} {'beta_insp~base':>14}")
r0 = 0.6869
for q in (3.0, 5.0, 8.0):
    base = (q/(1+q))**1.2
    for c in (-0.6, -0.3, 0.0, 0.4, 0.8):
        mf,_ = _S.mf(q,[0,0,c],[0,0,0]); chif,_ = _S.chif(q,[0,0,c],[0,0,0])
        o, n = old(q,c), new(q,c)
        print(f"{q:>4g} {c:>+6.2f} {float(chif[2]):>7.4f} {float(mf):>6.4f} "
              f"{o:>11.4f} {n:>11.4f} {o*r0:>9.4f} {n*r0:>9.4f} {base:>14.4f}")
