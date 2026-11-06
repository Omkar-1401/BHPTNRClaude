"""4-fold cross-validation on T96 for the split variants (2026-10-13; user: never select on H270, it is held out).

Selection between variants, and every tuned setting (window start, T_start line), is made on T96 only:
    fold of T96 point (iq, ic) = (iq + ic) % 4      (24 points per fold; each q twice, each chi1 three times)
Each variant is refit from scratch on the other 72 points (stage 1 always refit, then stage 2; split protocol of
explore_split.py) and scored on its fold with the full-window E (full alignment).  The 4 folds pool to 96 held-out
scores per variant.  This script loads and scores T96 points ONLY: no H270, no F100, no low q.
Starts: the optimisers start from the full-T96 coefficients, as in round 5 (a start, not data; noted in CV_LOG.md).

    python cv_split.py --fold K --variant NAME [--variant ...]     fixed-spec variants of explore_split.VARIANTS
    python cv_split.py --fold K|all --tline                         the T_start(chi1) line, below

T_start line ("tline", 14c): inspiral b pinned to 0 (8c), slim MR branch (5c), window start
    T_start(chi1) = min(T0 + T1 chi1, T_peak - 0.55)        (T_peak: Ori–Thorne time of the ppBHPT peak, per case)
T0 replaces the constant start; T1 is the added coefficient.  (T0, T1) is chosen from the grid GRID on the TRAINING points
only, by mean full-window E after the stage-2 fit at each grid point (stage 1 does not see the window).  T1 = 0 rows are
the constant-start model with T0 tuned on training.  --fold all = the full-T96 fit (the candidate to report).
-> results/cv/<name>__fold<K>.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explore_split as X
FF, M = X.FF, X.M
OT = M.OT
T96 = X.T96
NC = len(M.S.CHIS)
FOLD = {k: (i // NC + i % NC) % 4 for i, k in enumerate(T96)}
OUT = HERE / "results" / "cv"; OUT.mkdir(parents=True, exist_ok=True)
GRID = [(T0, T1) for T0 in (-1.0, -2.0, -3.0, -4.0) for T1 in (0.0, 1.5, 3.0, 4.5)]
TLINE = dict(insp=("pin0", ["b"]), mr="slim")
for _w in (-0.5, -2.0):                                         # made on the fly by explore_split.auto_round2
    X.VARIANTS.setdefault(f"slim__win{_w}", dict(insp=None, mr="slim", window=_w))


def split(fold):
    if fold == "all":
        return list(T96), []
    f = int(fold); return [k for k in T96 if FOLD[k] != f], [k for k in T96 if FOLD[k] == f]


def set_tline(T0, T1):
    """Per-case window start T0 + T1 chi1, guarded below the ppBHPT peak; rebuilds S before the fork."""
    starts = {}
    for k, c in M.CASES.items():
        starts[k] = OT.WINDOW_START = float(min(T0 + T1 * k[1], c["lm"]["T_peak"] - 0.55))
        Sb, _ = OT.blend(c["t_bhpt"], c["h_bhpt"], *k)
        c["S"] = Sb; c["fit_S"] = np.interp(c["fit_t_bhpt"], c["t_bhpt"], Sb)
    return starts


def fit(pool, spec, train, ti=None):
    """Two-stage fit on `train`; stage 1 is skipped when ti is given (it does not depend on the window)."""
    log = []
    if ti is None:
        FF.KEYS = train; FF.SEEDS.clear()
        ti, e1 = X.stage1(pool, X.TI_T96, spec["insp"][1] if spec["insp"] else [])
        log.append(dict(stage="insp", mean_E_insp=e1))
    FF.KEYS = train; FF.SEEDS.clear()
    x, e2 = X.stage2(pool, ti, spec["mr"])
    log.append(dict(stage="mr", mean_E_MR=e2))
    return ti, x, log


def score(pool, ti, mr, x, keys):
    if not keys:
        return []
    FF.KEYS = keys; FF.SEEDS.clear()
    return FF.final(pool, "split", ti, ("form", mr, list(map(float, x))))


def save(name, fold, d):
    d.update(fold=fold, folds_rule="(iq + ic) % 4 over T96 in (TRAIN_Q x CHIS) order")
    (OUT / f"{name}__fold{fold}.json").write_text(json.dumps(d, indent=1))


def line(name, fold, d):
    s = lambda rows: (f"{X.stats(rows)['median']:.3e}/{X.stats(rows)['mean']:.3e}/{X.stats(rows)['max']:.2e}" if rows else "-")
    print(f"CV {name:18s} fold {fold}: train {s(d['train_rows'])} | held-out {s(d['test_rows'])} | {d['seconds']}s", flush=True)


def run_fixed(name, fold):
    spec = X.VARIANTS[name]
    assert not spec.get("from") and not spec.get("joint"), f"{name}: joint/from variants are not the split protocol"
    train, test = split(fold); t = time.time()
    X.set_window(spec.get("window", -1.0))
    with get_context("fork").Pool(128) as pool:
        ti, x, log = fit(pool, spec, train)
        tr, te = score(pool, ti, spec["mr"], x, train), score(pool, ti, spec["mr"], x, test)
    n = 9 - len(spec["insp"][1] if spec["insp"] else []) + len(X.MR_FORMS[spec["mr"]][0])
    assert n <= 14, (name, n)
    d = dict(variant=name, spec=spec, ncoef=n, theta_insp=list(map(float, ti)), x_mr=list(map(float, x)), log=log,
             train_rows=tr, test_rows=te, seconds=round(time.time() - t))
    save(name, fold, d); line(name, fold, d)


def run_tline(fold):
    train, test = split(fold); t = time.time(); spec = TLINE
    n = 8 + len(X.MR_FORMS[spec["mr"]][0]) + 1                 # + T1 (T0 replaces the constant start)
    assert n <= 14, n
    with get_context("fork").Pool(128) as pool:                 # stage 1: window-independent, once
        FF.KEYS = train; FF.SEEDS.clear()
        ti, e1 = X.stage1(pool, X.TI_T96, spec["insp"][1])
    print(f"  tline fold {fold}: stage 1 mean E_insp {e1:.4e} ({time.time() - t:.0f}s)", flush=True)
    grid = []
    for T0, T1 in GRID:
        set_tline(T0, T1)
        with get_context("fork").Pool(128) as pool:
            _, x, log = fit(pool, spec, train, ti)
            tr = score(pool, ti, spec["mr"], x, train)
        g = dict(T0=T0, T1=T1, x_mr=list(map(float, x)), train_mean=X.stats(tr)["mean"], train_median=X.stats(tr)["median"],
                 mean_E_MR=log[-1]["mean_E_MR"], train_rows=tr)
        grid.append(g)
        print(f"  tline fold {fold}: T0 {T0:+.1f} T1 {T1:.1f} -> train mean {g['train_mean']:.4e} median "
              f"{g['train_median']:.4e} ({time.time() - t:.0f}s)", flush=True)
    best = min(grid, key=lambda g: g["train_mean"])
    starts = set_tline(best["T0"], best["T1"])
    with get_context("fork").Pool(128) as pool:
        te = score(pool, ti, spec["mr"], best["x_mr"], test)
    d = dict(variant="tline", spec=dict(spec, rule="T_start = min(T0 + T1 chi1, T_peak - 0.55)"), ncoef=n,
             chosen=dict(T0=best["T0"], T1=best["T1"]), theta_insp=list(map(float, ti)), x_mr=best["x_mr"],
             grid=[{k: v for k, v in g.items() if k != "train_rows"} for g in grid], window_starts={f"{k[0]:.6f},{k[1]:+.3f}": v for k, v in starts.items()},
             train_rows=best["train_rows"], test_rows=te, seconds=round(time.time() - t))
    save("tline", fold, d); line("tline", fold, d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", required=True, choices=["0", "1", "2", "3", "all"])
    ap.add_argument("--variant", action="append", default=[]); ap.add_argument("--tline", action="store_true")
    a = ap.parse_args()
    assert a.tline or (a.variant and a.fold != "all")
    for n in a.variant:
        assert n in X.VARIANTS, n
    for k in T96:                                               # T96 only
        M.CASES[k] = M.prepare(*k); X.rem(*k)
    assert sorted(np.bincount(list(FOLD.values()))) == [24] * 4
    for n in a.variant:
        run_fixed(n, a.fold)
    if a.tline:
        run_tline(a.fold)


if __name__ == "__main__":
    main()
