"""Gate for ml_bench/common.py: pass 40 (stand-alone generator) scored through score_ab and score_wave must reproduce the
stored per-case T96 values (exploration/p40/p40.json O2n T96_cases) at ratio 1.0000.

    python gate.py [n_cases]
"""
import json, sys
import numpy as np
import common as C
sys.path.insert(0, str(C.REPO / "exploration"))
import bhpt_spin_calib_p40 as P40

REF = json.loads((C.REPO / "exploration" / "p40" / "p40.json").read_text())["O2n"]["T96_cases"]
CF = P40.load_exact_coefficients()


def ab_fun(k, c):
    def f(key):
        pre = "fit_" if key == "fit" else ""
        return P40.alpha_beta(k[0], k[1], c[pre + "t_bhpt"], c[pre + "h_bhpt"], c[pre + "x"],
                              c[pre + "losses"]["e_oft"], coef=CF, t_full=c["t_bhpt"], h_full=c["h_bhpt"])
    return f


def wave_fun(k, c):
    """the same model rendered as a waveform on NR time (tau = int beta dt), for the score_wave path"""
    f = ab_fun(k, c)
    def g(key):
        pre = "fit_" if key == "fit" else ""
        a, b = f(key); t, h = c[pre + "t_bhpt"], c[pre + "h_bhpt"]
        bc = C.creative.cumulative_trapezoid(b, t); ts = bc - np.interp(C.S.T_ANCHOR, t, bc)
        return ts, a * h
    return g


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    keys = [C.T96[i] for i in np.linspace(0, len(C.T96) - 1, n).astype(int)]
    worst = 0.0
    for k in keys:
        c = C.case(k)
        e1 = C.score_ab(ab_fun(k, c), c); e2 = C.score_wave(wave_fun(k, c), c)
        ref = [v for kk, v in REF.items() if abs(float(kk.split("_")[0]) - k[0]) < 1e-5 and abs(float(kk.split("_")[1]) - k[1]) < 1e-5][0]
        worst = max(worst, abs(e1 / ref - 1), abs(e2 / ref - 1))
        print(f"q={k[0]:.4f} chi1={k[1]:+.1f}: ref {ref:.6e}  score_ab {e1:.6e} ({e1/ref:.6f})  score_wave {e2:.6e} ({e2/ref:.6f})", flush=True)
    # repo convention: ratio 1.0000 (4 digits); score_ab differs from the chain by t0-search detail only
    print(f"gate worst |ratio-1| = {worst:.2e}", "PASS" if worst < 1e-4 else "FAIL")
    print("pool size", len(C.train_pool()), "cached", len(C.cached_keys()))
