"""EXPLORATION pass 49 -- one look each at the sealed test set (R100, L20) for the pruned lean models (15..12 coefficients).
Selection was done on T96 fit-mean only; this is scoring, not selection.  Writes t100/p49_n{n}.json.
    python p49_test.py [base=p48lean]      (p40lean: pass 49b, writes t100/p49b_n{n}.json)
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t100_set as TS                           # noqa: E402  (sealed-set caches; opens the guard for this scoring only)
import modelsetup                               # noqa: E402
BASE = sys.argv[1] if len(sys.argv) > 1 else "p48lean"
_M = modelsetup.load(BASE)
DIR, TAG = {"p48lean": ("p49", "p49"), "p40lean": ("p49b", "p49b")}[BASE]
P6 = _M["P6"]
k6 = lambda q, c: f"{q:.6f}_{c:+.4f}"


def main():
    st = json.loads((HERE / DIR / "p49.json").read_text())
    # machinery gate: the 16c base must reproduce pass 48 lean on the sealed set
    ref = json.loads((HERE / "t100" / f"{_M['t100']}.json").read_text())["cases"]
    pool = mp.get_context("fork").Pool(96)
    chk = TS.TEST[::10]
    r = np.array(pool.map(P6._full, [(st["steps"][0]["theta"], k) for k in chk])) / np.array([ref[k6(*k)] for k in chk])
    print(f"base reproduces {_M['t100']}: {r.min():.4f}..{r.max():.4f}", flush=True); assert np.allclose(r, 1, atol=2e-3)
    s = lambda v: dict(median=float(np.median(v)), mean=float(np.mean(v)), max=float(np.max(v)), n=len(v))
    for x in st["steps"][1:]:
        if (HERE / "t100" / f"{TAG}_n{x['n']}.json").exists(): continue   # one look per model: never rescore
        e = np.array(pool.map(P6._full, [(x["theta"], k) for k in TS.TEST]))
        t = dict(model=f"{TAG}_n{x['n']}", R100=s(e[:100]), L20=s(e[100:]), cases={k6(*k): float(v) for k, v in zip(TS.TEST, e)})
        (HERE / "t100" / f"{TAG}_n{x['n']}.json").write_text(json.dumps(t, indent=1))
        print(f"n={x['n']}: R100 {t['R100']['median']:.4e} / {t['R100']['mean']:.4e} / {t['R100']['max']:.3e} | "
              f"L20 median {t['L20']['median']:.3e}", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
