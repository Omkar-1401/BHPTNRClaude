"""Score both spin pn models below the training grid (q = 2, 2.5), plus q = 3 as a check.

q < 3 is EXTRAPOLATION twice over: below TRAIN_Q (q >= 3) and below BHPTNRSur2dq1e3's own q_min = 3.
    pn_anc_spin_physical            pn_anc_spin_physical_results/coeffs_physical.json
    pn_anchored_spin switch+merger  test_free_switch_results/merger_full_stored.json (test)
Each model is scored exactly as its fit reports: physical = mismatch() with a cold t0 search
(as score_all does for the final table); pn_anchored_spin = fit_pn_anchored_spin.full_mm.
The q = 3 row must reproduce the stored values.  Writes test_free_switch_results/lowq_grid.json.
"""
from __future__ import annotations
import json, multiprocessing as mp, sys, warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S
import fit_pn_anc_spin_physical as P

QS = [2.0, 2.5, 3.0]
CHIS = S.CHIS
PHYS = json.loads(P.COEFFS.read_text())
PNS = json.loads((HERE / "test_free_switch_results" / "merger_full_stored.json").read_text())
for d, n in ((PHYS, "physical"), (PNS, "pn_anchored_spin")):
    if d.get("model_form") != S.MODEL_FORM:
        sys.exit(f"{n} coefficients stamped {d.get('model_form')!r}, not {S.MODEL_FORM!r}")
TH_P = np.array(PHYS["theta"], float)
TH9, THS = np.array(PNS["theta9"], float), np.array(PNS["theta_spin"], float)


def work(k):
    q, c = k
    case = S.build_case(q, c)              # generates + caches if missing
    ep, _ = P.mismatch(TH_P, q, c, case, None)
    en = S.full_mm(TH9, THS, q, c, case)
    return k, ep, en


if __name__ == "__main__":
    keys = [(q, c) for q in QS for c in CHIS]
    with mp.get_context("fork").Pool(len(keys)) as pool:
        res = pool.map(work, keys)
    out = {"physical": {}, "pn_anchored_spin_switch_merger": {}}
    for (q, c), ep, en in res:
        k = f"{q:.4f}_{c:+.1f}"
        out["physical"][k] = float(ep); out["pn_anchored_spin_switch_merger"][k] = float(en)
    ok = True
    for name, st, key in (("physical", PHYS["mismatches"], "physical"),
                          ("pn_anchored_spin", PNS["mismatches"], "pn_anchored_spin_switch_merger")):
        r = np.array([out[key][f"3.0000_{c:+.1f}"] / st[f"3.0000_{c:+.1f}"] for c in CHIS])
        print(f"q=3 check {name}: rescored/stored in [{r.min():.5f}, {r.max():.5f}]")
        ok &= bool(np.all(np.abs(r - 1) < 1e-3))
    for key in out:
        print(f"\n{key}  mathcalE x 1e-3")
        for q in QS:
            print(f"{q:5.2f} " + " ".join(f"{out[key][f'{q:.4f}_{c:+.1f}'] * 1e3:6.2f}" for c in CHIS))
    out["q3_check_passed"] = ok
    (HERE / "test_free_switch_results" / "lowq_grid.json").write_text(json.dumps(out, indent=2))
    print("\nq=3 check", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
