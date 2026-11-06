"""Diagnostic (2026-10-03): does the installed model's inspiral rely on its beta_r admixture?

Installed pn_anchored_spin in the inspiral: beta ~ (1 - S) beta_insp + S r0 beta_r(q, chi1) with S ~ 0.04-0.10
(its switch never turns off).  Switching it off costs ~4x in E_insp (2.1e-4 -> 1.1e-3, S18 median).  That
removed three things at once: (1) the beta_r(q, chi1) admixture, (2) the nu-dependence of the switch,
(3) the p_loss time profile.  Variants, all fitted to mean E_insp only ('insp' alignment, S18, no MR branch):
  admix_beta   : beta = (1 - s) beta_insp + s r0 beta_r;  alpha = alpha_insp          (7 + s, r0 = 9)
  admix_both   : + alpha = (1 - s) alpha_insp + s X1^(6/5) (m0 + m1 nu)                  (11)
  installed    : the installed form itself (p_loss sigmoid switch), refit on E_insp only  (13) -- the ceiling
User's prior (2026-10-03): a ringdown quantity should make the inspiral beta worse, not better.
-> results/diag_admixture.json
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
import diag_inspiral_forms_all as DA        # fit(), run(), full_res() harness, generic over `ab`

KEYS = DI.KEYS


def ab(model, p, q, chi, c, key):
    pre = "fit_" if key == "fit" else ""
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    if model == "installed":                                   # fit_pn_anchored_spin.model_ab, all 13 free
        return M.S.model_ab(p[:9], p[9:], q, chi, c[pre + "x"], c[pre + "p_loss"])
    b1, b2, a2, s11, s12, s21, s22 = p[:7]
    xc = M.PC.x_scale(q) * np.minimum(c[pre + "x"], M.S.XCLIP)
    bi = base * (1 + nu * ((b1 + s11 * chi + s12 * chi ** 2) * xc + (b2 + s21 * chi + s22 * chi ** 2) * xc ** 2))
    ai = base * (1 + M.S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
    s, r0 = p[7], p[8]
    beta = (1 - s) * bi + s * r0 * M.S.beta_r_phys(q, chi)
    if model == "admix_beta":
        return ai, beta
    m0, m1 = p[9], p[10]
    return (1 - s) * ai + s * base * (m0 + m1 * nu), beta


def main():
    for k in KEYS:
        M.CASES[k] = M.prepare(*k)
    DA.ab = ab                                                  # reuse the harness with these forms
    d = json.loads(M.S.COEFFS.read_text())
    th9, ths = np.array(d["theta9"]), np.array(d["theta_spin"])
    inst_in = np.r_[th9[:3], ths]                               # b1 b2 a2 s11 s12 s21 s22
    best_pn = np.array(json.loads((HERE / "results" / "diag_inspiral_form.json").read_text())["best_base"]["theta"])
    starts = {
        "admix_beta": {"installed": np.r_[inst_in, 0.07, th9[6]], "pn_best_s0.05": np.r_[best_pn, 0.05, 1.0]},
        "admix_both": {"installed": np.r_[inst_in, 0.07, th9[6], th9[7], th9[8]],
                       "pn_best_s0.05": np.r_[best_pn, 0.05, 1.0, 1.1, -1.4]},
        "installed": {"installed": np.r_[th9, ths]}}
    # check: the 'installed' variant with the shipped coefficients is the installed model
    k = KEYS[7]; c = M.CASES[k]
    a1, b1 = ab("installed", np.r_[th9, ths], *k, c, "full")
    a2, b2 = M.F.ab("pn13", M.F.installed("pn13", *k), *k, c, "full")
    assert np.allclose(a1, a2, rtol=1e-12) and np.allclose(b1, b2, rtol=1e-12)
    print("check: 'installed' variant with shipped coefficients == the installed model", flush=True)
    out = dict(note=__doc__.split("\n\n")[0], variants={})
    with get_context("fork").Pool(len(KEYS)) as pool:
        for model, st in starts.items():
            best = None; runs = []
            for name, p0 in st.items():
                DA.SEEDS.clear(); t = time.time()
                x, f0, f = DA.fit(pool, model, p0)
                print(f"{model:11s} start={name:14s} mean E_insp (fit grid) {f0:.3e} -> {f:.3e}  ({time.time() - t:.0f}s)", flush=True)
                runs.append(dict(start=name, start_mean=f0, end_mean=f, theta=list(map(float, x))))
                if best is None or f < best[0]:
                    best = (f, x)
            DA.SEEDS.clear()
            E, B = DA.full_res(pool, model, best[1])
            v = np.array(list(E.values()))
            extra = f"  s = {best[1][7]:.4f}, r0 = {best[1][8]:.4f}" if model != "installed" else ""
            print(f"BEST {model}: full-res E_insp mean {v.mean():.3e} median {np.median(v):.3e} max {v.max():.2e}{extra};"
                  " median by band: " + ", ".join(f"{lab} {np.median(B[:, j]):.1e}" for j, lab in
                                                  enumerate(["< -1000", "-1000..-300", "-300..-100", "-100..cut"])), flush=True)
            out["variants"][model] = dict(runs=runs, theta_best=list(map(float, best[1])),
                                          E_insp_full={f"{k[0]}_{k[1]}": e for k, e in E.items()},
                                          bands_median=list(np.median(B, axis=0)))
    print("reference (S18, insp-aligned, full res): installed pn_anchored_spin as shipped median 2.12e-4; "
          "pure pn inspiral form best 1.14e-3; energy/flux forms 6.8e-4 / 6.6e-4")
    (HERE / "results" / "diag_admixture.json").write_text(json.dumps(out, indent=1))
    print("wrote results/diag_admixture.json")


if __name__ == "__main__":
    main()
