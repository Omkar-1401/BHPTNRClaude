"""Overnight step 3 (2026-10-03/04): does the derived PN spin-orbit term fix the global inspiral?

    delta ln beta_SO = k (3/5) chi1 x^(3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1]      (OVERNIGHT_LOG step 1)
x = the note's x (= PC.x_scale(q) min(x_pp, XCLIP)); k = 1 is the derived (zero-parameter) term.
beta -> beta * (1 + delta).  Inspiral only ('insp' alignment), switch off, S18, mean E_insp objective.
Variants (coefficient counts):
  pn_SOd   pn x-polynomial (7) + derived SO           7     pn_SOk   + fitted k     8
  pn3_SOd  b1 b2 a2, no fitted spin terms + derived SO 3    pn3_SOk  + fitted k     4
  fl_SOd   fluxanchored form (9) + derived SO          9     fl_SOk   + fitted k    10
  fl0_SOd  fluxanchored with s1 = s2 = 0 + derived SO  7
References (same protocol): installed pn_anchored_spin 2.12e-4 (median); pn form 1.14e-3; flux form 6.62e-4.
-> results/diag_so_term.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_pn_anchored_spin_split as M
import diag_inspiral_form as DI
import diag_inspiral_forms_all as DA

KEYS = DI.KEYS
FS = M.F.FS


def so_coef(q, corrected=False):
    """Step-1 value used 3/5 (constant-beta matching).  Correct for an x^(3/2) term: 3/2 (OVERNIGHT_LOG)."""
    X1 = q / (1 + q)
    return (1.5 if corrected else 0.6) * (113 / 12 * X1 ** 1.2 - 19 / 6 * X1 ** 2 - 25 / 4 * X1)


def xnote(q, c, pre):
    return M.PC.x_scale(q) * np.minimum(c[pre + "x"], M.S.XCLIP)


def ab(model, p, q, chi, c, key):
    pre = "fit_" if key == "fit" else ""
    xc = xnote(q, c, pre)
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    if model.startswith("pn"):
        if model.startswith("pn3"):
            b1, b2, a2 = p[:3]; s11 = s12 = s21 = s22 = 0.0; rest = p[3:]
        else:
            b1, b2, a2, s11, s12, s21, s22 = p[:7]; rest = p[7:]
        beta = base * (1 + nu * ((b1 + s11 * chi + s12 * chi ** 2) * xc + (b2 + s21 * chi + s22 * chi ** 2) * xc ** 2))
        alpha = base * (1 + M.S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
    else:
        th = np.r_[p[:7], 0.0, 0.0] if model.startswith("fl0") else p[:9]
        rest = p[7:] if model.startswith("fl0") else p[9:]
        a_pp, a_f, b_pp, b_e = FS.params_at(q, chi, th)
        los = c[pre + "losses"]
        alpha = a_pp * (1 + a_f * los["flux_hat"]); beta = b_pp * (1 + b_e * los["e_oft"])
    k = rest[0] if model.endswith("SOk") else 1.0
    return alpha, beta * (1 + k * so_coef(q, corrected=model.endswith("SOD")) * chi * xc ** 1.5)


def main():
    for k in KEYS:
        M.CASES[k] = M.prepare(*k)
        M.CASES[k]["S"] = 0 * M.CASES[k]["S"]; M.CASES[k]["fit_S"] = 0 * M.CASES[k]["fit_S"]
    DA.ab = ab
    pn = np.array(json.loads((HERE / "results" / "diag_inspiral_form.json").read_text())["best_base"]["theta"])
    fl = np.array(json.loads((HERE / "results" / "diag_inspiral_forms_all.json").read_text())["models"]["flux9"]["theta_best"])
    fl_inst = np.array(json.loads((FS.RESULTS / "coeffs_nu.json").read_text())["theta"])
    if "--corrected" in sys.argv:
        flk = np.array(json.loads((HERE / "results" / "diag_so_term.json").read_text())["variants"]["fl_SOk"]["theta_best"])
        starts = {"fl_SOD": [flk[:9], fl], "pn3_SOD": [pn[:3]]}
        OUTNAME = "diag_so_term_corrected.json"
    else:
        OUTNAME = "diag_so_term.json"
    if "--corrected" not in sys.argv:
      starts = {"pn_SOd": [pn], "pn_SOk": [np.r_[pn, 1.0], np.r_[pn, 0.0]],
              "pn3_SOd": [pn[:3]], "pn3_SOk": [np.r_[pn[:3], 1.0], np.r_[pn[:3], 2.0]],
              "fl_SOd": [fl, fl_inst], "fl_SOk": [np.r_[fl, 1.0], np.r_[fl, 0.0]],
              "fl0_SOd": [fl[:7], fl_inst[:7]]}
    out = dict(note=__doc__.split("\n\n")[0], variants={})
    with get_context("fork").Pool(len(KEYS)) as pool:
        for model, sts in starts.items():
            best = None; runs = []
            for i, p0 in enumerate(sts):
                DA.SEEDS.clear(); t = time.time()
                x, f0, f = DA.fit(pool, model, p0)
                print(f"{model:8s} start {i}  mean E_insp (fit grid) {f0:.3e} -> {f:.3e}  ({time.time() - t:.0f}s)", flush=True)
                runs.append(dict(start=i, start_mean=f0, end_mean=f, theta=list(map(float, x))))
                if best is None or f < best[0]:
                    best = (f, x)
            DA.SEEDS.clear()
            E, B = DA.full_res(pool, model, best[1])
            v = np.array(list(E.values()))
            kk = f"  k = {best[1][-1]:.3f}" if model.endswith("SOk") else ""
            print(f"BEST {model}: full-res E_insp mean {v.mean():.3e} median {np.median(v):.3e} max {v.max():.2e}{kk};"
                  " median by band: " + ", ".join(f"{lab} {np.median(B[:, j]):.1e}" for j, lab in
                                                  enumerate(["< -1000", "-1000..-300", "-300..-100", "-100..cut"])), flush=True)
            out["variants"][model] = dict(runs=runs, theta_best=list(map(float, best[1])),
                                          E_insp_full={f"{k[0]}_{k[1]}": e for k, e in E.items()},
                                          bands_median=list(np.median(B, axis=0)))
    print("references (median E_insp): installed pn_anchored_spin 2.12e-4; pn form 1.14e-3; flux form 6.62e-4")
    (HERE / "results" / OUTNAME).write_text(json.dumps(out, indent=1))
    print("wrote results/" + OUTNAME)


if __name__ == "__main__":
    main()
