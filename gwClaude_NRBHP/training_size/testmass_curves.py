"""alpha(t), beta(t) of p40 / lean12 at q = 1000 (test-mass limit), for ml_bench/plot_testmass.py (user 2026-11-03).
Same chain as fit_p40.py (p40: shipped L4 theta; lean12: G96 refit from the t96 start, as in the ml_bench figures).
alpha, beta need only the ppBHPT waveform, so cases are built into a private cache with a PLACEHOLDER NR part (h_nr = h_pp):
no mismatch is computed here, and these cache files must never be scored.  Cache routing replaces grids._route for these keys only.
One model per process (the p42_otonset import used by lean12 changes p25's onset guard for every model in the process).

    python testmass_curves.py p40|lean12      -> ml_bench/results/testmass_<model>.npz
"""
import json, sys
import numpy as np

MODEL = sys.argv[1]
sys.argv = [sys.argv[0], MODEL]
import fit_p40 as F                            # noqa: E402  (model setup at import)

G, S, FS = F.G, F.G.S, F.G.FS
ML = G.REPO / "ml_bench" / "results"
WF, LOSS = ML / "testmass_cache" / "wf", ML / "testmass_cache" / "loss"
WF.mkdir(parents=True, exist_ok=True); LOSS.mkdir(parents=True, exist_ok=True)
QS, CHIS = (1000.0,), (0.0, 0.6)
KEYS = [(q, c) for c in CHIS for q in QS]
_KS = {(round(q, 6), round(c, 3)) for q, c in KEYS}
_route0 = G._route


def _route(q, chi1):
    if (round(float(q), 6), round(float(chi1), 3)) in _KS:
        S.CACHE, FS.LOSSCACHE = WF, LOSS
    else:
        _route0(q, chi1)


G._route = _route


# Remnant above NRSur3dq8Remnant's hard limit (q = 10): O(nu) test-mass result for a plunge from the Kerr ISCO (user choice
# 2026-11-03): mf = 1 - nu (1 - E_isco(chi1)), chif = chi1 + nu (L_isco - 2 chi1 E_isco); E, L per unit m2, in units of m1.
def kerr_isco_EL(a):
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3)); z2 = np.sqrt(3 * a * a + z1 * z1)
    r = 3 + z2 - np.sign(a) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2)); v = np.sqrt(r)
    d = r ** 0.75 * np.sqrt(r ** 1.5 - 3 * v + 2 * a)
    return (r ** 1.5 - 2 * v + a) / d, (r * r - 2 * a * v + a * a) / d


def remnant_onu(q, chi1):
    nu = q / (1 + q) ** 2; E, Lz = kerr_isco_EL(float(chi1))
    return 1 - nu * (1 - E), chi1 + nu * (Lz - 2 * chi1 * E)


class _Remnant:
    def __init__(self, fit):
        self.fit = fit

    def mf(self, q, s1, s2, **k):
        return self.fit.mf(q, s1, s2, **k) if q <= 10 else (float(remnant_onu(q, s1[2])[0]), None)

    def chif(self, q, s1, s2, **k):
        return self.fit.chif(q, s1, s2, **k) if q <= 10 else (np.array([0.0, 0.0, remnant_onu(q, s1[2])[1]]), None)


S._SFBH = _Remnant(S._SFBH)


def make_wf(q, chi1):
    """ppBHPT (2,2) from the 2d surrogate as S.build_case makes it; NR slot filled with a placeholder copy."""
    p = WF / f"wf_q{q:.6f}_chi{chi1:+.3f}.npz"
    if p.exists():
        return
    import BHPTNRSur2dq1e3 as ppsur
    t, hd = ppsur.generate_surrogate(q=q, spin1=chi1, calibrated=False, modes=[(2, 2)], neg_modes=False)
    h = np.asarray(hd[(2, 2)]); t = np.asarray(t, float)
    np.savez_compressed(p, t_bhpt=t, h_bhpt_re=h.real, h_bhpt_im=h.imag, t_nr=t, h_nr_re=h.real, h_nr_im=h.imag,
                        placeholder_nr=True)


def main():
    P6, L = F.P6, F.L
    P6.get = L.get; P6.ab = L.ab; L._NC["v"] = L.NUC; F.R2C.use("O2n")     # as fit_p40.main
    if F.K is not None:
        F.K.M.S1FUN = F.K.onset
    if MODEL == "lean12":
        th = np.array(json.loads((G.HERE / "results" / "lean12__G96__t96.json").read_text())["theta"], float)
    else:
        th = np.array(F.SHIP["theta"], float)
    out = {}
    for q, c in KEYS:
        make_wf(q, c)
        cs = P6.get((q, c))["case"]; a, b = P6.ab(th, (q, c), "full")
        if a is None:
            print(f"q={q:g} chi1={c:+.1f}: form invalid", flush=True); continue
        t, h = cs["t_bhpt"], cs["h_bhpt"]; s = t - np.asarray(t)[int(np.argmax(np.abs(h)))]
        tag = f"q{q:g}_chi{c:+.1f}"
        out[tag + "_s"], out[tag + "_alpha"], out[tag + "_beta"] = s, np.asarray(a), np.asarray(b)
        m = (s > -3000) & (s < -100)
        print(f"{MODEL} q={q:g} chi1={c:+.1f}: inspiral alpha {np.min(a[m]):.4f}..{np.max(a[m]):.4f}  "
              f"beta/X1 {np.min(b[m]) * (1 + q) / q:.4f}..{np.max(b[m]) * (1 + q) / q:.4f}", flush=True)
    np.savez(ML / f"testmass_{MODEL}.npz", **out)
    print("wrote", ML / f"testmass_{MODEL}.npz")


if __name__ == "__main__":
    main()
