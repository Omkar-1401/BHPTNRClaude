"""Overnight step 2 (2026-10-03/04): per-case inspiral fits on S18 -- how does beta's spin dependence scale with q?

Each (q, chi1) case is fitted on its own to E_insp ('insp' alignment, window NR [-5000.1, t_isco), switch off):
  flux form : a_pp, a_F, b_pp, b_e   (alpha = a_pp (1 + a_F F), beta = b_pp (1 + b_e E))      4 per case
  pn form   : b1c, b2c, a2           (beta = X1^(6/5) (1 + nu (b1c x + b2c x^2)), alpha as pn)  3 per case
Reported: per-case E_insp (the per-case floor of each form) and the early-inspiral beta prefactor
b_pp / X1^(6/5) vs chi1 at each q, compared with the derived PN spin-orbit term
  delta ln beta_SO = (3/5) chi1 x^(3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1]
averaged over the case's inspiral (|h_NR|^2-weighted x^(3/2) on the NR window).
-> results/diag_percase.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_pn_anchored_spin_split as M
import diag_inspiral_form as DI
import diag_so_term as SO

KEYS = DI.KEYS
FS = M.F.FS


def ab(form, p, q, chi, c, key):
    pre = "fit_" if key == "fit" else ""
    if form == "flux":
        a_pp, a_f, b_pp, b_e = p; los = c[pre + "losses"]
        return a_pp * (1 + a_f * los["flux_hat"]), b_pp * (1 + b_e * los["e_oft"])
    b1c, b2c, a2 = p
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    xc = SO.xnote(q, c, pre)
    return (base * (1 + M.S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2),
            base * (1 + nu * (b1c * xc + b2c * xc ** 2)))


def fit_case(a):
    form, k, p0 = a
    c = M.CASES[k]; st = {"t0": -50.0}
    def obj(p, key="fit", wide=False):
        al, be = ab(form, p, *k, c, key)
        (ei, _, _), t0 = M.score(al, be, c, key, "insp", st["t0"], wide)
        if ei < 1.0:
            st["t0"] = t0
        return ei
    obj(p0, wide=True)
    x = np.array(p0, float); best = obj(x)
    for rep in range(4):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-14, "adaptive": True})
        obj(r.x, wide=True)
        fr = obj(r.x)
        done = fr > best * (1 - 1e-3)
        if fr < best:
            x, best = r.x, fr
        if done:
            break
    obj(x, wide=True)
    e_full = obj(x, key="full", wide=True)
    # |h_NR|^2-weighted mean of x^(3/2) over the NR inspiral window, mapped through this case's beta
    al, be = ab(form, x, *k, c, "full")
    t, tn, hn = c["t_bhpt"], c["t_nr"], c["h_nr"]
    tau = st["t0"] + M.F.tau_shape(be, t)
    m = (tn >= tau[0]) & (tn < c["lm"]["t_cut_nr"])
    xc = np.interp(tn[m], tau, SO.xnote(k[0], c, ""))
    w = np.abs(hn[m]) ** 2
    return k, form, list(map(float, x)), float(e_full), float(np.sum(w * xc ** 1.5) / np.sum(w))


def main():
    for k in KEYS:
        M.CASES[k] = M.prepare(*k)
        M.CASES[k]["S"] = 0 * M.CASES[k]["S"]; M.CASES[k]["fit_S"] = 0 * M.CASES[k]["fit_S"]
    fl = np.array(json.loads((HERE / "results" / "diag_inspiral_forms_all.json").read_text())["models"]["flux9"]["theta_best"])
    pn = np.array(json.loads((HERE / "results" / "diag_inspiral_form.json").read_text())["best_base"]["theta"])
    tasks = []
    for k in KEYS:
        q, chi = k
        tasks.append(("flux", k, list(FS.params_at(q, chi, fl))))
        b1, b2, a2, s11, s12, s21, s22 = pn
        tasks.append(("pn", k, [b1 + s11 * chi + s12 * chi ** 2, b2 + s21 * chi + s22 * chi ** 2, a2]))
    t = time.time()
    with get_context("fork").Pool(len(tasks)) as pool:
        res = pool.map(fit_case, tasks, chunksize=1)
    print(f"{len(res)} per-case fits in {time.time() - t:.0f}s", flush=True)
    R = {(r[1], r[0]): r for r in res}
    for form in ("flux", "pn"):
        e = np.array([R[(form, k)][3] for k in KEYS])
        print(f"per-case floor, {form} form: E_insp median {np.median(e):.2e} max {e.max():.2e}")
    print("\nflux form, per-case b_pp / X1^(6/5) vs chi1, and the derived SO prediction exp(coef chi1 <x^1.5>) x const:")
    out = dict(note=__doc__.split("\n\n")[0], cases=[dict(form=f, q=k[0], chi1=k[1], theta=th, E_insp=e, x15=x15)
                                                    for k, f, th, e, x15 in res], slopes={})
    for q in (3.0, 5.0, 8.0):
        rows = sorted([R[("flux", k)] for k in KEYS if k[0] == q], key=lambda r: r[0][1])
        chis = np.array([r[0][1] for r in rows]); bpp = np.array([r[2][2] for r in rows]) / (q / (1 + q)) ** 1.2
        pred = SO.so_coef(q) * chis * np.array([r[4] for r in rows])          # delta ln beta at the mean x^1.5
        s_meas = np.polyfit(chis, np.log(bpp), 1)[0]; s_pred = np.polyfit(chis, pred, 1)[0]
        out["slopes"][str(q)] = dict(measured=float(s_meas), predicted=float(s_pred))
        print(f"  q={q:.0f}: " + "  ".join(f"{c:+.1f}:{b:.5f}" for c, b in zip(chis, bpp))
              + f"   | d ln b_pp/d chi1 measured {100 * s_meas:+.3f}%  derived {100 * s_pred:+.3f}%  ratio {s_meas / s_pred:.2f}")
    (HERE / "results" / "diag_percase.json").write_text(json.dumps(out, indent=1))
    print("wrote results/diag_percase.json")


if __name__ == "__main__":
    main()
