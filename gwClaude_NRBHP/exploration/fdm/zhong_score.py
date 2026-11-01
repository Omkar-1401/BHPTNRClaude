"""Score the Zhong et al. 2025 (arXiv:2503.04534) extracted (2,2) coefficients on our grids (fd_models.py has the map).

Gate first: the published Rink model through this harness must reproduce spin_dep_split/results/rink_scores.json on
T96[::8] (ratio 1 within 2e-3), else nothing is scored.  Then 'zhong' (their literal phase map, the Pareto entry) and
'zhong_t' (beta' as a time scaling, diagnostic) on T96 and V300 (training_size/V300.txt).  No fitting.
    python fdm/zhong_score.py      -> fdm/results/zhong_scores.json
"""
from __future__ import annotations
import json, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

import fd_models as FM
S = FM.S
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
T96 = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
V300 = [tuple(map(float, l.split()[:2])) for l in (REPO / "training_size" / "V300.txt").read_text().splitlines()
        if l.strip() and not l.startswith("#")]
k6 = lambda q, c: f"{q:.6f}_{c:+.4f}"


def _job(a):
    m, k = a
    try:
        return m, k, FM.score(m, *k)
    except Exception as exc:
        print(f"  [fail] {m} {k}: {exc!r}", flush=True); return m, k, float("nan")


def stats(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(median=float(np.median(v)), mean=float(np.mean(v)), max=float(v.max()), n=int(len(v)))


def main():
    t0 = time.time()
    assert len(V300) == 300, len(V300)
    R = json.loads((REPO / "spin_dep_split" / "results" / "rink_scores.json").read_text())
    ref = {(r["q"], r["chi1"]): r["full_E_full"] for r in R["T96"]}
    chk = T96[::8]
    with mp.get_context("fork").Pool(96) as pool:
        e = np.array([x[2] for x in pool.map(_job, [("rink", k) for k in chk])])
        r = e / np.array([ref[k] for k in chk])
        print(f"gate: published Rink through this harness / rink_scores.json on T96[::8]: {r.min():.5f}..{r.max():.5f}", flush=True)
        assert np.allclose(r, 1, atol=2e-3), r
        res = pool.map(_job, [(m, k) for m in ("zhong", "zhong_t") for k in T96 + V300], chunksize=2)
    out = dict(coefficients=dict(alpha=FM.Z_A, beta=FM.Z_B, source="arXiv:2503.04534 Tables II (l=2), III"),
               gate_rink_ratio=[float(r.min()), float(r.max())])
    for m in ("zhong", "zhong_t"):
        d = {k: v for mm, k, v in res if mm == m}
        t96 = [d[k] for k in T96]; v3 = [d[k] for k in V300]
        out[m] = dict(T96=stats(t96), V300=stats(v3),
                      T96_chi0=stats([d[k] for k in T96 if k[1] == 0.0]),
                      V300_lowspin=stats([d[k] for k in V300 if abs(k[1]) < 0.1]),
                      T96_cases={k6(*k): d[k] for k in T96}, V300_cases={k6(*k): d[k] for k in V300})
        s = out[m]
        print(f"{m:8s} T96 {s['T96']['median']:.3e} / {s['T96']['mean']:.3e} / {s['T96']['max']:.2e} | "
              f"V300 {s['V300']['median']:.3e} / {s['V300']['mean']:.3e} | chi1=0 T96 median {s['T96_chi0']['median']:.3e}",
              flush=True)
    (OUT / "zhong_scores.json").write_text(json.dumps(out, indent=1))
    print(f"done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
