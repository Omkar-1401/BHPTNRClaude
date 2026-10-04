"""Full refit of the shipped spin models with the CORRECTED beta_r.

WHY.  `fit_pn_anchored_spin.beta_r_phys` multiplies the pp-side QNM frequency by
(1+q)/q, converting the primary's QNM to total-mass units.  The ppBHPT surrogate's time
axis is ALREADY in primary-mass units (`calibrated=False` returns the raw training grid,
common_utils/utils.py:176), so the conversion is applied twice and beta_r is too large
by 1/X1 -- 33% at q=3, 12.5% at q=8.  Measured: the surrogate's late-ringdown frequency
is 0.4954 at q=5, chi1=+0.6 vs w220(0.6)=0.4975 (0.4%) and w220(0.6)*(1+q)/q=0.5969
(20%).  The DENOMINATOR (w220(chi_f)/M_f) is correct -- verified to 0.8-1.4% against the
measured NR ringdown frequency -- because the NR waveform IS in total-mass units.

Job 3533460 showed the shipped coefficients cannot simply be carried over: dropping the
correction in costs 14x, and refitting `r0` ALONE still leaves 2.2x.  All coefficients
had jointly adapted to the wrong anchor.  So this script refits EVERY free coefficient.

MODELS
  physical : fit_pn_anc_spin_physical, 11 coef [b1 b2 a2 p00 p01 w0 r0 m0 m1 sSO sSS]
  spin13   : fit_pn_anchored_spin, 13 coef = its 9 nonspinning + [s11 s12 s21 s22].
             The shipped protocol FREEZES the 9 and fits only the 4, but those 9 were
             themselves fitted against the wrong beta_r, so freezing them here would
             just carry the contamination in.  All 13 are freed; this is therefore NOT
             the shipped protocol and the number is not directly comparable to
             7.6858e-04 -- it answers "what can this form do with a correct anchor".

Grid and objective identical to the shipped fits: S.TRAIN_Q x S.CHIS = 96 points,
mean per-case mismatch, phi0 analytic, t0 a per-case nuisance.
"""
import argparse, json, multiprocessing as mp, os, sys, time as _time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S
import fit_pn_anc_spin_physical as PH

# The (1+q)/q fix now lives in the SOURCE (UNITS_AUDIT.md A1, 2026-09-24), so no
# monkeypatch: `fix=False` here means "as the source now stands" = corrected.  The
# `fix` argument is kept only so the OLD behaviour can be reproduced for comparison.
_ORIG = S.beta_r_phys
def _old(q, chi1):
    return _ORIG(q, chi1) / (q / (1.0 + q))      # put the spurious factor BACK

RESULTS = HERE / "corrected_betar_results"; RESULTS.mkdir(exist_ok=True)
_CASES = {}


def _score(arg):
    model, th, q, c, old = arg
    S.beta_r_phys = _old if old else _ORIG
    PH.S.beta_r_phys = S.beta_r_phys
    th = np.asarray(th, float)
    if model == "physical":
        e, _ = PH.mismatch(th, q, c, _CASES[(q, c)], None)
        return e
    return S.full_mm(th[:9], th[9:], q, c, _CASES[(q, c)])


def main():
    import fit_pn_anchored_spin as _S0; _S0.refuse_frozen9_protocol(__file__)   # 2026-09-27
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["physical", "spin13"])
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--budget", type=float, default=105.0)
    ap.add_argument("--nproc", type=int, default=0)
    a = ap.parse_args()

    if a.model == "physical":
        th0 = PH.shipped11()[0]; nm = PH.NAMES
    else:
        th0 = np.r_[S.shipped9(),
                    np.array(json.loads(
                        (HERE / "pn_anchored_spin_results" / "coeffs_spin.json"
                         ).read_text())["theta_spin"], float)]
        nm = ["b1", "b2", "a2", "p00", "p01", "w0", "r0", "m0", "m1",
              "s11", "s12", "s21", "s22"]
    print(f"model {a.model}: {len(th0)} coefficients", flush=True)

    QS, CS = S.TRAIN_Q, S.CHIS
    for q in QS:
        for c in CS:
            _CASES[(q, c)] = S.build_case(q, c)
    keys = [(q, c) for q in QS for c in CS]
    np_ = a.nproc or max(1, min(24, (os.cpu_count() or 4) - 2))
    pool = mp.get_context("fork").Pool(np_)
    print(f"{len(keys)} grid points on {np_} workers", flush=True)

    def grid(th, fix):
        return np.array(pool.map(_score, [(a.model, th, q, c, fix) for q, c in keys]))

    base = grid(th0, True)      # old beta_r  -> reproduces the published number
    start = grid(th0, False)    # corrected beta_r, shipped coefficients
    print(f"shipped coeffs, OLD beta_r     : mean {base.mean():.4e}  "
          f"median {np.median(base):.4e}   <-- published")
    print(f"shipped coeffs, CORRECTED      : mean {start.mean():.4e}  "
          f"median {np.median(start):.4e}   <-- pre-refit\n", flush=True)

    t0 = _time.time(); st = {"n": 0, "best": np.inf, "bx": th0.copy()}

    class _B(Exception):
        pass

    def obj(th):
        if (_time.time() - t0) > a.budget * 60.0:
            raise _B
        v = float(grid(th, False).mean()); st["n"] += 1
        if v < st["best"]:
            st["best"] = v; st["bx"] = np.asarray(th, float).copy()
        if st["n"] % 25 == 0:
            print(f"  [{st['n']:>4d}] {_time.time()-t0:>6.0f}s mean={v:.5e} "
                  f"best={st['best']:.5e}", flush=True)
        return v

    try:
        r = minimize(obj, th0, method="Powell",
                     options={"maxiter": a.maxiter, "xtol": 1e-3, "ftol": 1e-5})
        th, note = r.x, "converged"
    except _B:
        th, note = st["bx"], f"budget {a.budget:.0f} min"
    fin = grid(th, False)
    print(f"\nrefit ({note}, {st['n']} evals): mean {fin.mean():.4e}  "
          f"median {np.median(fin):.4e}  max {fin.max():.4e}")
    print("  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(nm, th)))
    print(f"\n{'q':>8} " + " ".join(f"{c:>+9.1f}" for c in CS))
    M = fin.reshape(len(QS), len(CS))
    for i, q in enumerate(QS):
        print(f"{q:>8.4f} " + " ".join(f"{v:>9.3e}" for v in M[i]))
    pool.close(); pool.join()
    (RESULTS / f"{a.model}.json").write_text(json.dumps({
        "model": a.model, "names": nm, "theta": th.tolist(), "note": note,
        "n_eval": st["n"],
        "RESCALE_QNM": bool(S.RESCALE_QNM), "W_SCHW": float(S.W_SCHW),
        "old_betar_shipped_coeffs": {"mean": float(base.mean()),
                                         "median": float(np.median(base))},
        "corrected_betar_shipped_coeffs": {"mean": float(start.mean()),
                                           "median": float(np.median(start))},
        "corrected_betar_refit": {"mean": float(fin.mean()),
                                  "median": float(np.median(fin)),
                                  "max": float(fin.max())},
        "per_case": {f"{q:.4f}_{c:+.1f}": float(v) for (q, c), v in zip(keys, fin)},
    }, indent=1))
    print(f"wrote {RESULTS / (a.model + '.json')}")


if __name__ == "__main__":
    main()
