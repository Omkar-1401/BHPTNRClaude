"""Low-q extrapolation check (2026-10-04, user request): q = 2 and 2.5, chi1 = -0.6 .. +0.8 (8 spins), no refitting.

CAVEAT: the ppBHPT surrogate BHPTNRSur2dq1e3 is trained for q >= 3, so its INPUT waveform is itself extrapolated here
(EXPLORATION.md: never VALIDATE at q = 2).  Absolute errors mix surrogate and model extrapolation; every model gets the
same ppBHPT waveform, so the comparison BETWEEN models is fair.

Models (all T96-fitted, full-window E, 'full' alignment, plus the split errors):
  slim14    fluxanchored_spin_split, slim MR branch (aF, bPP, S)        results/slim_confirm_aF_bPP_S.json
  split18   fluxanchored_spin_split, full MR copy                       results/flux_spin_split_T96_T-1.json
  fluxSO9   original flux form + derived SO term, one form              spin_dep/flux_anc_spin_with_orbit_results/coeffs.json
  flux9     installed fluxanchored_spin;  pn13 installed pn_anchored_spin
Caches: waveforms read from spin_dep/spin_wf_cache; energy/flux drives from spin_dep/flux_loss_cache when present, else
computed into spin_dep_split/loss_cache_lowq (q = 2.5 had none).  -> results/extrap_lowq.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slim_split as SL                     # installs the masked-MR ab into fit_flux_spin_split
FF, M = SL.FF, SL.M
D = M.D
CASES_LIST = [(q, c) for q in (2.0, 2.5) for c in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)]
EXTRA = HERE / "loss_cache_lowq"; EXTRA.mkdir(exist_ok=True)


def get_case(q, chi, where="grid"):
    S, FS = D.S, D.FS
    S.CACHE = HERE.parent / "spin_dep" / "spin_wf_cache"
    w = S.CACHE / f"wf_q{q:.6f}_chi{chi:+.3f}.npz"
    if not w.exists():
        sys.exit(f"missing waveform cache {w}")
    shared = HERE.parent / "spin_dep" / "flux_loss_cache" / f"loss_q{q:.6f}_chi{chi:+.3f}.npz"
    FS.LOSSCACHE = shared.parent if shared.exists() else EXTRA     # never write into spin_dep
    return FS.build_case(q, chi)


D.get_case = get_case


def main():
    for k in CASES_LIST:
        M.CASES[k] = M.prepare(*k)
    FF.KEYS = CASES_LIST
    R = json.loads((HERE / "results" / "slim_confirm_aF_bPP_S.json").read_text())
    ti_slim = np.array(R["theta_insp"]); tm_slim = SL.build_tm(ti_slim, R["groups"], np.array(R["x"]))
    s18 = json.loads((HERE / "results" / "flux_spin_split_T96_T-1.json").read_text())
    so9 = np.array(json.loads((HERE.parent / "spin_dep" / "flux_anc_spin_with_orbit_results" / "coeffs.json").read_text())["theta"])
    out = {}
    with get_context("fork").Pool(len(CASES_LIST)) as pool:
        out["slim14"] = FF.final(pool, "split", ti_slim, tm_slim)
        out["split18"] = FF.final(pool, "split", np.array(s18["theta_insp"]), np.array(s18["theta_mr"]))
        for kind in ("flux9", "pn13"):
            out[kind] = FF.final(pool, kind, None, None)
    for k in CASES_LIST:                                    # one-form model: no blend
        c = M.CASES[k]; c["S"] = 0 * c["S"]; c["fit_S"] = 0 * c["fit_S"]
    FF.SEEDS.clear()
    with get_context("fork").Pool(len(CASES_LIST)) as pool:
        out["fluxSO9"] = FF.final(pool, "split", so9, so9)
    names = ["slim14", "split18", "fluxSO9", "flux9", "pn13"]
    print("E_full (full-aligned) per case:")
    print("  q    chi1   " + "  ".join(f"{n:>9s}" for n in names))
    for i, k in enumerate(CASES_LIST):
        print(f"  {k[0]:.1f} {k[1]:+.1f}   " + "  ".join(f"{out[n][i]['full_E_full']:9.2e}" for n in names))
    for q in (2.0, 2.5):
        print(f"q={q}: median E_full  " + "  ".join(
            f"{n} {np.median([r['full_E_full'] for r in out[n] if r['q'] == q]):.2e}" for n in names))
    (HERE / "results" / "extrap_lowq.json").write_text(json.dumps(out, indent=1))
    print("wrote results/extrap_lowq.json")


if __name__ == "__main__":
    main()
