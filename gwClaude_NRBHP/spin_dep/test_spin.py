import sys, numpy as np
sys.path.insert(0, "/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/surrogates")
import h5py
f = h5py.File("/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/data/BHPTNRSur2dq1e3.h5","r")
print("h5 OK, keys:", list(f.keys())[:8]); f.close()
import BHPTNRSur2dq1e3 as sur
for (q, chi) in [(6.0, 0.0), (6.0, 0.6), (6.0, -0.4)]:
    t, h = sur.generate_surrogate(q=q, spin1=chi, calibrated=False,
                                  modes=[(2,2),(3,3)], neg_modes=False)
    a = np.abs(h[(2,2)])
    print(f"q={q} chi1={chi:+.1f}: t [{t[0]:.0f},{t[-1]:.0f}] n={len(t)} "
          f"peak|h22|={a.max():.4f} at t={t[np.argmax(a)]:.1f}")
