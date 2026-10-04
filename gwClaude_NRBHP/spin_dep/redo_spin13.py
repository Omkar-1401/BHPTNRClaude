"""pn_anchored_spin (13 coef) with the corrected beta_r -- SHIPPED PROTOCOL PRESERVED.

The published 7.6858e-04 is a MEDIAN over the 96-point grid, obtained by FREEZING the
nine nonspinning coefficients (transplanted from the 1d fit) and fitting only the four
spin terms s11, s12, s21, s22.  An earlier attempt here freed all 13, which is a
different model and not comparable -- and it was contaminated besides: `shipped9()` reads
pn_anchored_results/coeffs.json, which a concurrent job was overwriting at the time, so
its "baseline" was the bad refit rather than the shipped coefficients.  Both problems are
fixed here: the protocol is kept, and STEP 0 halts unless the published per-case values
reproduce exactly.

The frozen nine now come from the CORRECTED nonspinning multistart
(pn_anchored_results/coeffs_corrected_betar_multistart.json), not from the shipped file,
because the shipped nine were themselves fitted against the wrong beta_r.

Steps, as in redo_properly.py:
  0  original config (A1 off, A2 off), shipped 13 -> must reproduce the published values
  1  clock fix only (A1)
  2  clock + constant (A1 + A2)
Reports MEDIAN first (the published statistic) and mean alongside.
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S

OUT = HERE / "redo_properly_results"; OUT.mkdir(exist_ok=True)
MULTI = HERE.parent / "pn_anchored_results" / "coeffs_corrected_betar_multistart.json"
_FIXED = S.beta_r_phys
_CASES: dict = {}
LO = np.array([-20.0, -20.0, -40.0, -40.0])       # s11, s12, s21, s22
HI = np.array([ 20.0,  20.0,  40.0,  40.0])


def _set_mode(mode):
    S.RESCALE_QNM = (mode == 2)
    S.beta_r_phys = (lambda q, c: _FIXED(q, c) / (q / (1.0 + q))) if mode == 0 else _FIXED


def _score(arg):
    th9, ths, q, c, mode = arg
    _set_mode(mode)
    return S.full_mm(np.asarray(th9, float), np.asarray(ths, float), q, c, _CASES[(q, c)])


def main():
    import fit_pn_anchored_spin as _S0; _S0.refuse_frozen9_protocol(__file__)   # 2026-09-27
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--nproc", type=int, default=0)
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 2],
                    help="1 = clock fix only, 2 = clock + constant")
    ap.add_argument("--budget", type=float, default=95.0,
                    help="wall-clock minutes; stop cleanly, keep best so far")
    a = ap.parse_args()

    d = json.load(open(HERE / "PREFIX_COEFFS" /
                       "pn_anchored_spin_results_coeffs_spin.json"))
    pub = d["mismatches"]
    th9_pub = np.array(d["theta9_frozen"], float)
    ths_pub = np.array(d["theta_spin"], float)
    PUB_MED = float(d["median"])

    for k in pub:
        q, c = k.split("_"); _CASES[(float(q), float(c))] = S.build_case(float(q), float(c))
    keys = list(_CASES)
    nproc = a.nproc or max(1, min(24, (os.cpu_count() or 4) - 2))
    _T0 = time.time()
    pool = mp.get_context("fork").Pool(nproc)
    print(f"{len(keys)} cases on {nproc} workers\n", flush=True)

    e0 = np.array(pool.map(_score, [(th9_pub, ths_pub, q, c, 0) for q, c in keys]))
    ratio = np.array([e / pub[f"{q:.4f}_{c:+.1f}"] for e, (q, c) in zip(e0, keys)])
    print(f"STEP 0  published reproduced: median {np.median(e0):.4e} "
          f"(stored {PUB_MED:.4e})   worst per-case ratio {np.abs(ratio-1).max()+1:.5f}")
    if np.abs(ratio - 1).max() > 2e-3:
        print("        *** BASELINE DOES NOT REPRODUCE -- STOPPING ***"); sys.exit(1)
    print("        baseline verified\n", flush=True)

    if not MULTI.exists():
        print(f"*** need the corrected nonspinning fit first: {MULTI} ***"); sys.exit(2)
    rows9 = json.load(open(MULTI))["rows"]
    th9 = np.array(min(rows9, key=lambda r: r["obj"])["theta"], float)
    print(f"frozen 9 (corrected nonspinning, best start): {np.round(th9, 4)}\n", flush=True)

    rng = np.random.default_rng(20260924)
    starts = [ths_pub] + [np.clip(ths_pub * rng.uniform(0.4, 1.8, 4), LO, HI)
                          for _ in range(a.starts - 1)]

    res = {0: {"median": float(np.median(e0)), "mean": float(e0.mean())}}
    MODES = {1: "clock fix only", 2: "clock + constant"}
    for mode in a.modes:
        name = MODES[mode]
        def fit(s0):
            f = lambda t: float(np.mean(pool.map(
                _score, [(th9, np.clip(t, LO, HI), q, c, mode) for q, c in keys])))
            r = minimize(f, np.clip(s0, LO, HI), method="Powell",
                         bounds=list(zip(LO, HI)),
                         options={"maxiter": a.maxiter, "xtol": 1e-3, "ftol": 1e-5})
            t = np.clip(r.x, LO, HI)
            e = np.array(pool.map(_score, [(th9, t, q, c, mode) for q, c in keys]))
            return dict(median=float(np.median(e)), mean=float(e.mean()),
                        max=float(e.max()), theta_spin=t.tolist())
        # Save after EVERY start and respect a wall-clock budget: the first version of
        # this script printed only after all 8 starts finished and had no budget guard,
        # so a 2 h wall would have killed it mid-step and written nothing.
        t0 = time.time(); rr = []
        for i, s0 in enumerate(starts):
            if (time.time() - _T0) / 60.0 > a.budget:
                print(f"        budget reached after {i} starts", flush=True); break
            rr.append(fit(s0))
            rr.sort(key=lambda r: r["median"])
            print(f"        start {i}: median {rr[-1]['median']:.4e}   "
                  f"best so far {rr[0]['median']:.4e}   "
                  f"({time.time()-t0:.0f}s)", flush=True)
            (OUT / "spin13_partial.json").write_text(json.dumps(
                {"mode": mode, "n_done": len(rr), "rows": rr}, indent=1, default=float))
        if not rr:
            print("        no starts completed"); break
        b = rr[0]
        print(f"STEP {mode}  {name}  ({time.time()-t0:.0f}s, {len(rr)} starts)")
        print(f"        best  median {b['median']:.4e}  mean {b['mean']:.4e}  "
              f"max {b['max']:.4e}")
        print(f"        spread: median {rr[0]['median']:.4e} .. {rr[-1]['median']:.4e}")
        print(f"        vs published median {PUB_MED:.4e}  -> "
              f"{100*(b['median']/PUB_MED-1):+.1f}%\n", flush=True)
        res[mode] = {"best": b, "all": rr}

    pool.close(); pool.join()
    (OUT / "spin13.json").write_text(json.dumps(
        {"published_median": PUB_MED, "frozen9_corrected": th9.tolist(), "steps": res},
        indent=1, default=float))
    print(f"wrote {OUT/'spin13.json'}")


if __name__ == "__main__":
    main()
