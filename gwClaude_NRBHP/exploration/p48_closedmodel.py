"""EXPLORATION pass 48 -- the fully closed-form model: pass 40 (and its lean variant) with every numerical input replaced
(R2: pass 40; QNM, remnant, ringdown decay: pass 46; shapes: pass 47b).  Coefficients frozen (pass 47b: a refit buys nothing).
    m48      pass 40's 18 coefficients                      (stand-alone: bhpt_spin_calib_p48.py)
    lean48   pass 42 K1's 16 coefficients + the Ori–Thorne onset
Scores T96, H270, the fixed low-q points and the sealed test set (R100, L20; one look per model, logged).  The m48 T96 must
reproduce pass 47b "Sboth" at ratio 1 (machinery gate).
    python p48_closedmodel.py      -> exploration/p48/p48.json (+ t100/m48.json, t100/lean48.json)
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t100_set as TS                           # noqa: E402  (sealed-set caches; opens the guard for this scoring only)
sys.argv = [sys.argv[0], "K1"]
import p42_otonset as K42                       # noqa: E402  (widened onset guard; OT onset hook)
sys.argv = [sys.argv[0], "40", "v2"]
import p47b_shapescf as B47                     # noqa: E402  (pass-46 closed-form inputs + shape installer)
B = B47.B; P6 = B.P6; L = B.L; PR = B.PR; M = K42.M
OUT = HERE / "p48"; OUT.mkdir(exist_ok=True)
k6 = lambda q, c: f"{q:.6f}_{c:+.4f}"


def _full(a):
    return P6._full(a)


def score(th, tag):
    pool = mp.get_context("fork").Pool(96)
    sc = PR.score(pool, th)
    et = np.array(pool.map(_full, [(list(th), k) for k in TS.TEST])); pool.close()
    st = lambda v: dict(median=float(np.median(v)), mean=float(np.mean(v)), max=float(np.max(v)), n=len(v))
    t = dict(model=tag, R100=st(et[:100]), L20=st(et[100:]), cases={k6(*k): float(v) for k, v in zip(TS.TEST, et)})
    (HERE / "t100" / f"{tag}.json").write_text(json.dumps(t, indent=1))
    lq = [v for kk, v in sc["lowq"].items() if kk.startswith("2.5")]
    print(f"{tag}: T96 {sc['T96'][1]:.4e} / {sc['T96'][0]:.4e} / {sc['T96'][2]:.3e} | H270 median {sc['H270'][1]:.4e} | "
          f"q2.5 max {max(lq):.3e} | R100 {t['R100']['median']:.4e} / {t['R100']['mean']:.4e} / {t['R100']['max']:.3e} | "
          f"L20 median {t['L20']['median']:.3e}", flush=True)
    return sc, t


def main():
    B47.SH = json.loads((HERE / "shapes_closedform_v2.json").read_text())
    P6.get = B.get; P6.ab = L.ab; B.C.use("O2n")
    B.CF.install(B.S, B.PC, qnm=True, remnant=True); B._USE_G["v"] = True; B47.install_shapes(True, True)
    out = {}
    th40 = np.array(json.loads((HERE / "p39" / "p39_L4.json").read_text())["theta"])
    ref = json.loads((HERE / "p47" / "p47b.json").read_text())["Sboth"]["T96_cases"]
    pool = mp.get_context("fork").Pool(96)
    chk = P6.T96[::8]; r = np.array(pool.map(_full, [(list(th40), k) for k in chk])) / np.array([ref[k6(*k)] for k in chk])
    pool.close(); print(f"m48 reproduces pass 47b Sboth: {r.min():.4f}..{r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    out["m48"], _ = score(th40, "m48")
    M.S1FUN = K42.onset                          # lean: the OT onset
    thk = np.array(json.loads((HERE / "p42" / "p42_K1.json").read_text())["theta"])
    out["lean48"], _ = score(thk, "lean48")
    (OUT / "p48.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
