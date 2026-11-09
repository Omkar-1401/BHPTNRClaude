"""Why does the 8-mode ppBHPT (2,2) differ from the cached (2,2) case?  (diagnostic, no outputs)"""
import numpy as np
import fd_route
S, FS = fd_route.S, fd_route.FS
import BHPTNRSur2dq1e3 as ppsur
from hm_cache import MODES
for k in [(5.0, -0.6), (5.0, 0.0), (3.0, 0.6)]:
    case = FS.build_case(*k); t0, h0 = case["t_bhpt"], case["h_bhpt"]
    for label, modes in (("22 only", [(2, 2)]), ("8 modes", MODES)):
        t, hd = ppsur.generate_surrogate(q=k[0], spin1=k[1], calibrated=False, modes=modes, neg_modes=False)
        t = np.asarray(t, float); h = np.asarray(hd[(2, 2)])
        same_t = t.shape == t0.shape and np.allclose(t, t0)
        if same_t:
            r = h / h0; m = np.abs(h0) > 1e-3 * np.abs(h0).max()
            print(k, label, "same grid; max|dh|/max|h| %.2e; |ratio| %.6f..%.6f; arg ratio %.4f..%.4f" % (
                np.abs(h - h0).max() / np.abs(h0).max(), np.abs(r[m]).min(), np.abs(r[m]).max(),
                np.angle(r[m]).min(), np.angle(r[m]).max()), flush=True)
        else:
            print(k, label, "grid differs", t.shape, t0.shape, t[:2], t0[:2], flush=True)
