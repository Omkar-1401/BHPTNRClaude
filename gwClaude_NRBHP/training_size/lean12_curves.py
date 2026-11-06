"""alpha(t), beta(t) of lean12 (pass 49b, G96 refit from the t96 start) on the ml_bench figE3 V300 cases, for plot_efit.py.
Same chain as fit_p40.py lean12; each case is re-scored and must reproduce the stored per-case V300 E (gate).
Run in its own process: the p42_otonset import widens p25's onset guard for every model in the process.

    TS_FRESH=1 python lean12_curves.py
"""
import json, sys
import numpy as np

sys.argv = [sys.argv[0], "lean12"]
import fit_p40 as F                            # noqa: E402  (model setup at import: lean12 chain, p49b free set)

ML = F.G.REPO / "ml_bench" / "results"
R = json.loads((F.G.REPO / "training_size" / "results" / "lean12__G96__t96.json").read_text())
TH = np.array(R["theta"], float)
CASES = json.loads((ML / "efit_summary.json").read_text())["_cases_E3"]


def _tpeak(t, h):                              # as ml_bench/direct_targets._tpeak
    a = np.abs(h); i = int(np.argmax(a)); y0, y1, y2 = a[i - 1], a[i], a[i + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return t[i] + d * (t[i + 1] - t[i])


def main():
    P6, L = F.P6, F.L
    P6.get = L.get; P6.ab = L.ab; L._NC["v"] = L.NUC; F.R2C.use("O2n")     # as fit_p40.main
    F.K.M.S1FUN = F.K.onset
    out = {}
    for name, (q, c) in CASES.items():
        k = (float(q), float(c))
        e = P6._full((list(TH), k)); ref = R["per_case"]["V300"][F.C.kstr(k)]
        print(f"{name:10s} q={q:.4f} chi1={c:+.3f}: E {e:.6e}  stored {ref:.6e}  ratio {e / ref:.6f}", flush=True)
        if abs(e / ref - 1) > 2e-3:
            sys.exit("gate failed: lean12 chain does not reproduce the stored V300 case")
        cs = P6.get(k)["case"]; a, b = P6.ab(TH, k, "full")
        t, h = cs["t_bhpt"], cs["h_bhpt"]
        out[f"{name}_s"] = t - _tpeak(t, h); out[f"{name}_alpha"] = np.asarray(a); out[f"{name}_beta"] = np.asarray(b)
    np.savez(ML / "lean12_E3_curves.npz", **out)
    print("wrote", ML / "lean12_E3_curves.npz")


if __name__ == "__main__":
    main()
