"""TEST, NOT A MODEL: pn_anchored_spin with the SWITCH freed (p00, p01, w0 + 4 spin = 7 free).

Why (2026-09-27).  On the fixed code, pn_anchored_spin with the new nonspinning 9 frozen is
+16% median / +31% mean against published and had nearly stalled (job 3540979).  The
2026-09-24 subset test (scaling_junction_spin.md sec 15, PRE-fix code) found that of the 9,
only the switch matters: freeing p00, p01, w0 cut the chi1=+0.8 column 2.4x and the mean 44%,
while the merger amplitudes r0, m0, m1 barely moved anything.  The physical fit, with every
coefficient free, moved the switch much later (p00 0.89 -> 4.08).  So free the switch only;
b1, b2, a2, r0, m0, m1 stay at the nonspinning pn_anchored values.  No chi1 term in the switch
(user, 2026-09-27: not yet).

Two arms on the SAME subset, so the subset's own overfitting is measured, not credited:
    --free spin     control: the 4 spin terms only (the shipped protocol)
    --free switch   p00, p01, w0 + the 4 spin terms
    --free merger   p00, p01, w0, r0, m0, m1 + the 4 spin terms (only b1, b2, a2 frozen)
Both start from pn_anchored_spin_results/coeffs_spin.json (stamped, current model_form).
Scoring is fit_pn_anchored_spin.full_mm -- the model's own objective, not a copy.
Writes test_free_switch_results/<arm>.json only; never touches coeffs_spin.json.

    python test_spin_free_switch.py --free switch --budget 90
    python test_spin_free_switch.py --free spin --budget 90
Default subset = sec 15's 18 points (q 3, 5, 8 x chi1 -0.6, -0.2, 0, +0.4, +0.6, +0.8),
which over-samples the bad chi1=+0.8 column; --q/--chi override, no args after --q/--chi
means the full grid.
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S

OUT = HERE / "test_free_switch_results"; OUT.mkdir(exist_ok=True)
NAMES9 = ["b1", "b2", "a2", "p00", "p01", "w0", "r0", "m0", "m1"]
FREE = {"spin": [], "switch": [3, 4, 5],          # indices into the 9 that are freed
        "merger": [3, 4, 5, 6, 7, 8]}               # switch + merger branch (r0, m0, m1)
SUB_Q = [3.0, 5.0, 8.0]
SUB_CHI = [-0.6, -0.2, 0.0, 0.4, 0.6, 0.8]
_CASES: dict = {}
_TH9 = None


def key(q, c):
    return f"{q:.4f}_{c:+.1f}"


def _work(a):
    th9, ths, q, c = a
    return S.full_mm(th9, ths, q, c, _CASES[(q, c)])


def main():
    global _TH9
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", choices=sorted(FREE), required=True)
    ap.add_argument("--q", type=float, nargs="*", default=None)
    ap.add_argument("--chi", type=float, nargs="*", default=None)
    ap.add_argument("--budget", type=float, default=90.0, help="optimiser wall-clock minutes")
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--nproc", type=int, default=18)
    ap.add_argument("--start-switch", type=float, nargs=3, default=None,
                    metavar=("P00", "P01", "W0"),
                    help="--free switch: start the switch here instead of the stored values "
                         "(the stored start is still scored first as the reproduction check)")
    ap.add_argument("--start-merger", type=float, nargs=3, default=None,
                    metavar=("R0", "M0", "M1"),
                    help="--free merger: start r0, m0, m1 here instead of the stored values")
    ap.add_argument("--start-json", type=Path, default=None,
                    help="start the free coefficients from a previous test output json "
                         "(its theta9 / theta_spin)")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="multiply every free start value by 1 + U(-f, f) (multistart)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --perturb")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args()
    QS = SUB_Q if a.q is None else (a.q or S.TRAIN_Q)
    CS = SUB_CHI if a.chi is None else (a.chi or S.CHIS)
    keys = [(q, c) for q in QS for c in CS]

    ship = json.loads(S.COEFFS.read_text())
    if ship.get("model_form") != S.MODEL_FORM:
        sys.exit(f"{S.COEFFS.name} is stamped {ship.get('model_form')!r}, not {S.MODEL_FORM!r}")
    th9 = S.model9()           # installed 9 (2026-09-27); refuses stale / mismatched files
    ths = np.array(ship["theta_spin"], float)
    idx = FREE[a.free]
    free_names = [NAMES9[i] for i in idx] + S.SNAMES
    print(f"TEST pn_anchored_spin, free = {free_names} ({len(free_names)}); "
          f"{len(QS)} q x {len(CS)} chi1 = {len(keys)} points", flush=True)
    print("frozen 9 start: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES9, th9)))

    for k in keys:
        _CASES[k] = S.build_case(*k)
    pool = mp.get_context("fork").Pool(min(a.nproc, len(keys)))

    def unpack(x):
        t9 = th9.copy(); t9[idx] = x[:len(idx)]
        return t9, np.asarray(x[len(idx):], float)

    def score(x):
        t9, ts = unpack(x)
        return np.array(pool.map(_work, [(t9, ts, q, c) for q, c in keys]))

    x0 = np.r_[th9[idx], ths]
    e0 = score(x0)
    stored = np.array([ship["mismatches"][key(*k)] for k in keys])
    print(f"start (stored coeffs on this subset): mean {e0.mean():.4e} median "
          f"{np.median(e0):.4e} max {e0.max():.4e}; stored values give mean "
          f"{stored.mean():.4e} (ratio {e0.mean()/stored.mean():.5f})", flush=True)
    if abs(e0.mean() / stored.mean() - 1) > 1e-3:
        sys.exit("start does not reproduce the stored per-case values -- stop")
    moved = []
    if a.start_json is not None:
        d = json.loads(a.start_json.read_text())
        if d.get("model_form") != S.MODEL_FORM:
            sys.exit(f"{a.start_json} is stamped {d.get('model_form')!r}")
        x0 = np.r_[np.asarray(d["theta9"], float)[idx], d["theta_spin"]]
        moved.append(f"from {a.start_json.name}")
    if a.perturb:
        rng = np.random.default_rng(a.seed)
        x0 = x0 * (1 + rng.uniform(-a.perturb, a.perturb, x0.size))
        moved.append(f"perturbed +-{a.perturb:.0%} (seed {a.seed})")
    if moved:
        e1 = score(x0)
        print(f"start moved ({'; '.join(moved)}): " + ", ".join(
            f"{n}={v:+.4f}" for n, v in zip(free_names, x0)), flush=True)
        print(f"  moved start: mean {e1.mean():.4e} median {np.median(e1):.4e} "
              f"max {e1.max():.4e}", flush=True)
    if a.start_switch is not None or a.start_merger is not None:
        x0 = x0.copy()
        if a.start_switch is not None:
            if 3 not in idx:
                sys.exit("--start-switch needs --free switch or merger")
            x0[:3] = a.start_switch
        if a.start_merger is not None:
            if 6 not in idx:
                sys.exit("--start-merger needs --free merger")
            x0[3:6] = a.start_merger
        e1 = score(x0)
        print(f"start moved to switch {a.start_switch} merger {a.start_merger}: mean "
              f"{e1.mean():.4e} median {np.median(e1):.4e} max {e1.max():.4e}", flush=True)

    st = {"n": 0, "best": np.inf, "bx": x0.copy(), "t0": time.time()}
    part = OUT / f"{a.free}{a.tag}_partial.json"

    class _Budget(Exception):
        pass

    def obj(x):
        e = score(x); m = float(e.mean()); st["n"] += 1; tag = ""
        el = (time.time() - st["t0"]) / 60
        if m < st["best"]:
            st["best"], st["bx"] = m, np.array(x, float); tag = "  <-- best"
            part.write_text(json.dumps({"n_eval": st["n"], "names": free_names,
                                        "x": st["bx"].tolist(), "mean": m}, indent=2))
        if tag or st["n"] % 25 == 0:
            print(f"  [{st['n']:4d}] {el:6.1f}m mean={m:.4e} median={np.median(e):.4e} "
                  f"max={e.max():.4e}{tag}", flush=True)
        if el > a.budget:
            raise _Budget
        return m

    try:
        r = minimize(obj, x0, method="Powell",
                     options={"maxiter": a.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        note = f"CONVERGED / {st['n']} evals: {r.message}"
    except _Budget:
        note = f"STOPPED ON BUDGET / {st['n']} evals"
    print(f"  {note}", flush=True)

    t9, ts = unpack(st["bx"]); e = score(st["bx"])
    print("\nfitted: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(free_names, st["bx"])))
    print(f"\n{'chi1':>6} " + " ".join(f"{'q=%g' % q:>19}" for q in QS) + "   (start -> fitted)")
    per = dict(zip(keys, e)); per0 = dict(zip(keys, e0))
    for c in CS:
        print(f"{c:+6.1f} " + " ".join(f"{per0[(q, c)]:9.2e}->{per[(q, c)]:8.2e}" for q in QS))
    print(f"\nSUBSET {len(keys)}: start mean {e0.mean():.4e} median {np.median(e0):.4e} "
          f"max {e0.max():.4e}\n           fitted mean {e.mean():.4e} median "
          f"{np.median(e):.4e} max {e.max():.4e}")
    out = OUT / f"{a.free}{a.tag}.json"
    out.write_text(json.dumps({
        "test": "pn_anchored_spin with a subset of the nonspinning 9 freed (NOT a model)",
        "start_switch": a.start_switch, "start_merger": a.start_merger,
        "start_json": str(a.start_json) if a.start_json else None,
        "perturb": a.perturb, "seed": a.seed, "x_start": x0.tolist(),
        "model_form": S.MODEL_FORM, "free": free_names, "note": note,
        "grid": {"q": QS, "chi1": CS}, "theta9": t9.tolist(), "theta_spin": ts.tolist(),
        "start": {"theta9": th9.tolist(), "theta_spin": ths.tolist(),
                  "mismatches": {key(*k): float(v) for k, v in per0.items()}},
        "mismatches": {key(*k): float(v) for k, v in per.items()},
        "mean": float(e.mean()), "median": float(np.median(e)), "max": float(e.max()),
    }, indent=2))
    print(f"wrote {out}")
    pool.close()


if __name__ == "__main__":
    main()
