"""Score the other session's two final SPLIT candidates (spin_dep_split/, 14c each; CV_LOG.md 2026-10-14) on the sealed
TEST set (t100_set.py).  Read-only on spin_dep_split/: its code is imported and its result jsons read, nothing is written
there.  No fitting.
    slim__win-2.0   best pooled CV median (results/explore/slim__win-2.0.json)
    tline           best pooled CV mean / max (results/cv/tline__foldall.json: inspiral b = 0, T_start = -3)
Protocol = spin_dep_split/score_f100.py (FF.final, full_E_full = full-window E, full alignment), with the TEST keys served
from exploration/t100/cache.  Stored T96 per-case scores are reproduced on T96[::8] first (ratio 1 within 2e-3).
tline: H270 and q = 2.5 / 2 are scored too (validation / fixed low-q points; the explore variant already carries them).
    python t100_split.py      -> exploration/t100/split_<name>.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent; SPL = HERE.parent / "spin_dep_split"
sys.path.insert(0, str(HERE))
import t100_set as TS                          # noqa: E402  (opens the guard; FS.build_case serves TEST keys from t100/cache)
sys.path.insert(0, str(SPL))
import extrap_lowq as XL                       # noqa: E402  (read-only get_case for every non-TEST key)
import cv_split as C                           # noqa: E402
X, FF, M = C.X, C.FF, C.M
D = M.D
assert D.FS is TS.FS, "split code must share the patched fit_fluxanchored_spin module"
_xl = XL.get_case


def get_case(q, chi, where="grid"):
    if (round(float(q), 6), round(float(chi), 3)) in TS._T:
        return TS.FS.build_case(q, chi)
    return _xl(q, chi, where)


D.get_case = get_case
T96 = X.T96; CHK = T96[::8]
OUT = HERE / "t100"


def stats(v):
    v = np.asarray(v); return dict(median=float(np.median(v)), mean=float(v.mean()), max=float(v.max()), n=len(v))


def main():
    H270 = X.H.H270; LQ = XL.CASES_LIST
    keys = list(dict.fromkeys(CHK + TS.TEST + H270 + LQ))
    for k in keys:
        M.CASES[k] = M.prepare(*k); X.rem(*k)
    cands = {}
    d = json.loads((SPL / "results" / "explore" / "slim__win-2.0.json").read_text())
    cands["slim__win-2.0"] = (d, lambda: X.set_window(d["spec"].get("window", -1.0)), d["spec"]["mr"],
                              {(r["q"], r["chi1"]): r["full_E_full"] for r in d["cases"]["T96"]}, False)
    e = json.loads((SPL / "results" / "cv" / "tline__foldall.json").read_text())
    cands["tline"] = (e, lambda: C.set_tline(e["chosen"]["T0"], e["chosen"]["T1"]), e["spec"]["mr"],
                      {(r["q"], r["chi1"]): r["full_E_full"] for r in e["train_rows"]}, True)
    for name, (d, setup, mr, ref, extra) in cands.items():
        t0 = time.time(); setup()
        ti, x = np.array(d["theta_insp"]), d["x_mr"]
        with get_context("fork").Pool(128) as pool:
            rc = C.score(pool, ti, mr, x, CHK)
            r = np.array([row["full_E_full"] / ref[(row["q"], row["chi1"])] for row in rc])
            print(f"{name} ({d['ncoef']}c): reproduce stored T96 on {len(CHK)} points, ratio {r.min():.4f}..{r.max():.4f}", flush=True)
            if not np.allclose(r, 1, atol=2e-3):
                print(f"{name}: reproduction FAILED {np.round(r, 4)}; test set NOT scored", flush=True); continue
            rt = C.score(pool, ti, mr, x, TS.TEST)
            more = {g: C.score(pool, ti, mr, x, ks) for g, ks in (("H270", H270), ("lowq", LQ))} if extra else {}
        et = {(row["q"], row["chi1"]): row["full_E_full"] for row in rt}
        er = [et[k] for k in TS.R100]; el = [et[k] for k in TS.L20]
        out = dict(model=f"split_{name}", n_coef=d["ncoef"], seed=TS.SEED, reproduce_ratio=[float(r.min()), float(r.max())],
                   R100=stats(er), L20=stats(el), cases={f"{q:.6f}_{c:+.4f}": float(v) for (q, c), v in et.items()},
                   T96=stats(list(ref.values())))
        if extra:
            out["H270"] = stats([row["full_E_full"] for row in more["H270"]])
            for qq in (2.5, 2.0):
                out[f"q{qq}"] = stats([row["full_E_full"] for row in more["lowq"] if row["q"] == qq])
        else:
            out["H270"] = d["stats"]["H270"]["full"]; out["lowq_median"] = d["lowq_median"]
        (OUT / f"split_{name}.json").write_text(json.dumps(out, indent=1))
        s, l = out["R100"], out["L20"]
        print(f"split_{name} ({d['ncoef']}c): R100 {s['median']:.3e} / {s['mean']:.3e} / {s['max']:.2e} | "
              f"L20 {l['median']:.3e} / {l['mean']:.3e} / {l['max']:.2e} | H270 median {out['H270']['median']:.3e}"
              f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
