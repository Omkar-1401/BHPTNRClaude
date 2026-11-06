"""Diagnostic (2026-10-03): which inspiral form should carry the split line?

Same test as diag_inspiral_form.py (pn_anchored_spin's x-polynomial, switch off: best median E_insp
1.14e-3 vs the installed model's 2.12e-4), now for the other three models' forms.  Each form is fitted
to mean E_insp ONLY ('insp' alignment, S18, several starts), with no switch / no MR branch:
  phys11 : X1^(6/5) [1 + nu (b1 x + b2 x^2 + sSO chi1 x^(3/2) + sSS chi1^2 x^2)],  alpha as pn  (5 coef)
           = fit_pn_anc_spin_physical.model_ab with S = 0
  en8    : alpha = a_pp (1 + a_e E(t)),  beta = b_pp (1 + b_e E(t))   (fit_energy_anchored_spin.params_at, 8)
  flux9  : alpha = a_pp (1 + a_F F(t)),  beta = b_pp (1 + b_e E(t))   (fit_fluxanchored_spin.params_at, 9)
E = cumulative radiated energy (per M, the models' own despiked drive), F = Edot / max Edot.
The flux / energy models have no switch, so this is their whole form, fitted to the inspiral only.
Not a model; nothing is installed.  -> results/diag_inspiral_forms_all.json
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
import diag_inspiral_form as DI                     # KEYS, BANDS
PH, FS, EA = M.F.PH, M.F.FS, M.F.EA

KEYS, BANDS = DI.KEYS, DI.BANDS
SEEDS = {}
NAMES = {"phys11": ["b1", "b2", "a2", "sSO", "sSS"],
         "en8": list(EA.NAMES), "flux9": list(FS.NAMES)}


def installed(model):
    if model == "phys11":
        th, _ = PH.shipped11()                        # b1 b2 a2 p00 p01 w0 r0 m0 m1 sSO sSS
        return np.array([th[0], th[1], th[2], th[9], th[10]])
    p = (EA if model == "en8" else FS).RESULTS / "coeffs_nu.json"
    return np.array(json.loads(p.read_text())["theta"], float)


SPIN_IDX = {"phys11": [3, 4], "en8": [6, 7], "flux9": [7, 8]}


def ab(model, p, q, chi, c, key):
    pre = "fit_" if key == "fit" else ""
    if model == "phys11":
        b1, b2, a2, sSO, sSS = p
        th = [b1, b2, a2, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, sSO, sSS]           # switch parameters unused
        X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
        xc = M.PC.x_scale(q) * np.minimum(c[pre + "x"], M.S.XCLIP)          # as PH.model_ab, S = 0
        beta = base * (1 + nu * (b1 * xc + b2 * xc ** 2 + sSO * chi * xc ** 1.5 + sSS * chi ** 2 * xc ** 2))
        alpha = base * (1 + M.S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
        return alpha, beta
    los = c[pre + "losses"]
    if model == "en8":
        a_pp, a_e, b_pp, b_e = EA.params_at(q, chi, p)
        return a_pp * (1 + a_e * los["e_oft"]), b_pp * (1 + b_e * los["e_oft"])
    a_pp, a_f, b_pp, b_e = FS.params_at(q, chi, p)
    return a_pp * (1 + a_f * los["flux_hat"]), b_pp * (1 + b_e * los["e_oft"])


def _check_phys_matches_module():
    """Our S = 0 phys11 form must equal PH.model_ab with its switch pushed fully off."""
    k = KEYS[4]; c = M.CASES[k]; p = installed("phys11")
    th = [p[0], p[1], p[2], 1e6, 0.0, 1.0, 1.0, 1.0, 0.0, p[3], p[4]]       # p00 huge -> S = 0
    a1, b1 = PH.model_ab(th, *k, c["x"], c["p_loss"]); a2, b2 = ab("phys11", p, *k, c, "full")
    assert np.allclose(a1, a2, rtol=1e-12) and np.allclose(b1, b2, rtol=1e-12)


def _task(a):
    model, k, p, seed, key, wide = a
    al, be = ab(model, p, *k, M.CASES[k], key)
    errs, t0 = M.score(al, be, M.CASES[k], key, "insp", seed, wide)
    return k, errs[0], t0


def run(pool, model, p, key="fit", wide=False):
    out = pool.map(_task, [(model, k, p, SEEDS.get(k, -50.0), key, wide) for k in KEYS])
    for k, e, t0 in out:
        if e < 1.0:
            SEEDS[k] = t0
    return {k: e for k, e, _ in out}


def fit(pool, model, p0):
    run(pool, model, p0, wide=True)
    obj = lambda p: float(np.mean(list(run(pool, model, p).values())))
    x, best = np.array(p0, float), obj(p0); f0 = best
    for rep in range(6):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 800 * len(x), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
        run(pool, model, r.x, wide=True)
        fr = obj(r.x)
        improved = fr < best * (1 - 1e-3)
        if fr < best:
            x, best = r.x, fr
        if not improved:
            break
    r = minimize(obj, x, method="Powell", options={"xtol": 1e-6, "ftol": 1e-10, "maxiter": 20000})
    if r.fun < best:
        x, best = r.x, float(r.fun)
    return x, f0, best


def full_res(pool, model, p):
    run(pool, model, p, wide=True)
    E = run(pool, model, p, key="full", wide=True)
    bands = []
    for k in KEYS:
        c = M.CASES[k]; al, be = ab(model, p, *k, c, "full")
        t, h, tn, hn = c["t_bhpt"], c["h_bhpt"], c["t_nr"], c["h_nr"]
        tau = SEEDS[k] + M.F.tau_shape(be, t)
        m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1])); tc, r = tn[m], hn[m]
        g = np.interp(tc, tau, al) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
        ins = tc < c["lm"]["t_cut_nr"]; z = np.vdot(g[ins], r[ins]); g = g * z / abs(z)
        res = np.abs(r - g) ** 2; n = (np.abs(r[ins]) ** 2).sum()
        bands.append([float(res[ins & (tc >= lo) & (tc < hi)].sum() / (2 * n)) for lo, hi in BANDS])
    return E, np.array(bands)


def main():
    for k in KEYS:
        M.CASES[k] = M.prepare(*k)
    _check_phys_matches_module()
    print("check: phys11 S=0 form == fit_pn_anc_spin_physical.model_ab with switch off", flush=True)
    out = dict(note=__doc__.split("\n\n")[0], models={})
    with get_context("fork").Pool(len(KEYS)) as pool:
        for model in ("phys11", "en8", "flux9"):
            p_inst = installed(model)
            nospin = p_inst.copy(); nospin[SPIN_IDX[model]] = 0.0
            starts = {"installed": p_inst, "installed_no_spin": nospin,
                      "installed_x1.05": p_inst * 1.05}
            SEEDS.clear()
            E0 = run(pool, model, p_inst, key="full", wide=True)            # installed coefficients, as is
            best, runs = None, []
            for name, s in starts.items():
                SEEDS.clear(); t = time.time()
                x, f0, f = fit(pool, model, s)
                print(f"{model:6s} start={name:18s} mean E_insp (fit grid) {f0:.3e} -> {f:.3e}  ({time.time() - t:.0f}s)", flush=True)
                runs.append(dict(start=name, start_mean=f0, end_mean=f, theta=list(map(float, x))))
                if best is None or f < best[0]:
                    best = (f, x)
            SEEDS.clear()
            E, B = full_res(pool, model, best[1])
            v = np.array(list(E.values())); v0 = np.array(list(E0.values()))
            print(f"BEST {model}: full-res E_insp mean {v.mean():.3e} median {np.median(v):.3e} max {v.max():.2e}"
                  f"  [installed coefficients, insp-aligned: median {np.median(v0):.3e}]; median by band: "
                  + ", ".join(f"{lab} {np.median(B[:, j]):.1e}" for j, lab in
                              enumerate(["< -1000", "-1000..-300", "-300..-100", "-100..cut"])), flush=True)
            out["models"][model] = dict(names=NAMES[model], runs=runs, theta_best=list(map(float, best[1])),
                                        E_insp_full={f"{k[0]}_{k[1]}": e for k, e in E.items()},
                                        E_insp_installed={f"{k[0]}_{k[1]}": e for k, e in E0.items()},
                                        bands_median=list(np.median(B, axis=0)))
    print("reference: pn_anchored_spin x-polynomial form, switch off, best: median 1.14e-3 (b0 variant 9.4e-4); "
          "installed pn_anchored_spin: median 2.12e-4, by band 1.0e-4, 3.9e-5, 2.3e-5, 7.9e-5")
    (HERE / "results" / "diag_inspiral_forms_all.json").write_text(json.dumps(out, indent=1))
    print("wrote results/diag_inspiral_forms_all.json")


if __name__ == "__main__":
    main()
