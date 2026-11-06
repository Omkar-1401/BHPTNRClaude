"""Pipeline item 5 (prepared 2026-10-05; run when the user says so): overfitting checks for the slim split (14c:
9 inspiral + derived SO, 5 MR: A0' A1' b' s1' s2'; Ori–Thorne window).

Each refit repeats the slim model's protocol on a subset of the training points:
  stage 1  inspiral copy theta_i on mean E_insp (MR copy = inspiral copy while fitting, as the original split started)
  stage 2  the 5 MR coefficients (groups aF, bPP, S) on mean E_MR, inspiral frozen   (slim_split.stage2)
(1) Random halves: 4 random 48 + 48 splits of T96 (seed 1, the same splits as spin_dep/overfit_checks_flux_orbit.py);
    fit on the first half, score full-window E (full alignment) on the OTHER half and on H270; stability of all 14
    coefficients over the 4 half-fits.
(2) Learning curve: H270 error when fitted on 18 (S18), 48 (half 0) and 96 points (the confirmed slim fit).
Starts from the confirmed slim coefficients (results/slim_confirm_aF_bPP_S.json) every time; nothing is installed.
-> results/overfit_checks_slim_split.json

    python overfit_checks_slim_split.py            (compute node; ~60 min on one development node)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slim_split as SL                      # installs the masked-MR ab; stage2(); build_tm()
FF, M = SL.FF, SL.M
import score_h270 as H

GROUPS = ["aF", "bPP", "S"]
S18 = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]
N_SPLITS, SEED = 4, 1


def refit(pool, keys, ti0, x0):
    FF.KEYS = keys; FF.SEEDS.clear(); log = []
    ti, _ = FF.stage(pool, "insp", np.array(ti0), np.array(ti0), log)             # MR copy = inspiral copy in stage 1
    FF.SEEDS.clear()
    x, tm = SL.stage2(pool, ti, ("groups", GROUPS), np.array(x0))
    return ti, x, tm


def score(pool, ti, tm, keys):
    FF.KEYS = keys; FF.SEEDS.clear()
    v = np.array([r["full_E_full"] for r in FF.final(pool, "split", ti, tm)])
    return dict(median=float(np.median(v)), mean=float(v.mean()), max=float(v.max()))


def main():
    T96 = [(q, c) for q in M.S.TRAIN_Q for c in M.S.CHIS]
    for k in T96 + H.H270:
        M.CASES[k] = M.prepare(*k)
    R = json.loads((HERE / "results" / "slim_confirm_aF_bPP_S.json").read_text())
    ti_conf, x_conf = np.array(R["theta_insp"]), np.array(R["x"])
    rng = np.random.default_rng(SEED)
    splits = []
    for i in range(N_SPLITS):
        p = rng.permutation(len(T96)); splits.append(([T96[j] for j in p[:48]], [T96[j] for j in p[48:]]))
    out = dict(note=__doc__.split("\n\n")[0], halves=[], learning={})
    with get_context("fork").Pool(128) as pool:
        for i, (a, b) in enumerate(splits):
            t = time.time(); ti, x, tm = refit(pool, a, ti_conf, x_conf)
            r = dict(split=i, theta_insp=list(map(float, ti)), x=list(map(float, x)),
                     other_half=score(pool, ti, tm, b), H270=score(pool, ti, tm, H.H270))
            out["halves"].append(r)
            print(f"half {i}: other half {r['other_half']['median']:.3e}/{r['other_half']['mean']:.3e}  "
                  f"H270 {r['H270']['median']:.3e}/{r['H270']['mean']:.3e}  ({time.time() - t:.0f}s)", flush=True)
        ti18, x18, tm18 = refit(pool, S18, ti_conf, x_conf)
        h0 = out["halves"][0]
        for name, ti, x in (("18 (S18)", ti18, x18), ("48 (half 0)", np.array(h0["theta_insp"]), np.array(h0["x"])),
                            ("96 (confirmed)", ti_conf, x_conf)):
            tm = SL.build_tm(ti, GROUPS, x)
            out["learning"][name] = dict(H270=score(pool, ti, tm, H.H270))
            print(f"learning {name:15s}: H270 {out['learning'][name]['H270']['median']:.3e} / "
                  f"{out['learning'][name]['H270']['mean']:.3e} / {out['learning'][name]['H270']['max']:.2e}", flush=True)
    names = list(FF.FS.NAMES) + ["A0'", "A1'", "b'", "s1'", "s2'"]
    T = np.array([h["theta_insp"] + h["x"] for h in out["halves"]]); conf = np.r_[ti_conf, x_conf]
    out["coefficient_stability"] = {n: dict(confirmed=float(conf[j]), mean=float(T[:, j].mean()), std=float(T[:, j].std()),
                                            rel_std=float(T[:, j].std() / (abs(T[:, j].mean()) + 1e-12)))
                                    for j, n in enumerate(names)}
    for n, d in out["coefficient_stability"].items():
        print(f"  {n:4s} confirmed {d['confirmed']:+.4f}  halves {d['mean']:+.4f} +- {d['std']:.4f}  (rel {d['rel_std']:.2f})")
    (HERE / "results" / "overfit_checks_slim_split.json").write_text(json.dumps(out, indent=1))
    print("wrote results/overfit_checks_slim_split.json")


if __name__ == "__main__":
    main()
