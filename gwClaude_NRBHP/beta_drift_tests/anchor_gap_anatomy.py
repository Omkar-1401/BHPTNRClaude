"""WHERE does the anchor gap come from?  Decompose B1/B0 - 1 into its ingredients.

  B0 = X1^(6/5)                                    inspiral: mass-ratio algebra only
  B1 = W_Schw * Mf / omega_220(chi_f)               ringdown: remnant mass x spin correction
     = Mf  *  [W_Schw / omega_220(chi_f)]
       ^mass    ^spin factor (=1 exactly when chi_f=0)

So   1 + g_b = B1/B0 = (Mf / X1^(6/5)) * S(chi_f),   S = W_Schw/omega_220(chi_f) < 1.

Both -> 1 as nu -> 0.  The question is the nu-scaling of the RESIDUAL: if the anchors agreed
to O(nu) and O(nu^2) and only parted at higher order, the gap is a genuinely small,
high-order effect rather than a leading-order disagreement.
"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import BHPTNRPNAnchored as PNA
import gwModels   # valid to q ~ 1000, unlike surfinBH (q <= 8)

W = 0.3737
QS = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 50.0, 200.0, 1000.0]
print(f"{'q':>7s} {'nu':>8s} {'B0':>8s} {'Mf':>7s} {'chi_f':>7s} {'S(chi_f)':>9s} "
      f"{'B1':>8s} {'g_b %':>8s} {'g_b/nu':>8s} {'g_b/nu^2':>9s} {'g_b/nu^3':>9s}")
rows = []
for q in QS:
    X1 = q/(1+q); nu = q/(1+q)**2
    B0 = X1**1.2
    r = gwModels.remnants.gwModelRemS(q, 0.0, 0.0)
    mf, chif = float(r[0]), float(r[1])
    qn = PNA._cfg()["qnm"]
    w220 = qn["F1"] + qn["F2"]*(1.0-chif)**qn["F3"]     # dimensionless, per remnant mass
    S = W/w220
    B1 = W*mf/w220
    g = B1/B0 - 1.0
    print(f"{q:7.1f} {nu:8.5f} {B0:8.5f} {mf:7.4f} {chif:7.4f} {S:9.5f} {B1:8.5f} "
          f"{g*100:8.3f} {g/nu:8.4f} {g/nu**2:9.4f} {g/nu**3:9.3f}")
    rows.append((q, nu, g, mf, chif, S, B0))

a = np.array([(r[1], r[2]) for r in rows if r[2] > 0], float)
print("\nlocal log-log slope  d ln g_b / d ln nu  (adjacent pairs):")
for i in range(len(a)-1):
    p = np.log(a[i,1]/a[i+1,1])/np.log(a[i,0]/a[i+1,0])
    print(f"   nu {a[i+1,0]:.5f} -> {a[i,0]:.5f}   p = {p:6.3f}")

print("\nthe two ingredients separately, as fractional deviations from 1:")
print(f"{'q':>7s} {'Mf-1 %':>9s} {'S-1 %':>9s} {'B0-1 %':>9s} {'sum %':>9s} {'g_b %':>9s}")
for q, nu, g, mf, chif, S, B0 in rows:
    print(f"{q:7.1f} {(mf-1)*100:9.3f} {(S-1)*100:9.3f} {(B0-1)*100:9.3f} "
          f"{((mf-1)+(S-1)-(B0-1))*100:9.3f} {g*100:9.3f}")
