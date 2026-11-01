"""EXPLORATION pass 22 -- prune the 27c leader (pass 18: V6r + 1 beta / 1 alpha pass-17 shape) toward <= 20 coefficients.

Round 1: each candidate free coefficient fixed at 0, the rest refitted from the leader for STEP_MIN minutes (T96 fit-stride
mean).  Skipped: t1a, t2a (window base times) and the six pass-15 essentials (flux_b2, r1, pa2, flux_s3, r3, pb1).
Round 2: joint drop of every candidate costing < CUT, longer refit; T96, H270, low q at full resolution.
Writes exploration/p22/p22_prune.json.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "V6", "p15", "1", "1"]
import p18_shapes2 as L                         # noqa: E402
import fit_pn_anchored_spin as S                # noqa: E402
P6 = L.P6; A = L.A
OUT = HERE / "p22"; OUT.mkdir(exist_ok=True)
LEAD = json.loads((HERE / "p18" / "p18_p15_11.json").read_text())
TH0 = np.array(LEAD["theta"]); FREE0 = list(LEAD["free"]); NL0 = len(A.NAMES)
NAMES = A.NAMES + ["eb0", "eb1", "eb2", "ea0", "ea1", "ea2"]          # the pass-18 shape amplitudes
assert len(NAMES) == len(TH0)
SKIP = {"t1a", "t2a", "flux_b2", "r1", "pa2", "flux_s3", "r3", "pb1"}
CUT, STEP_MIN = 0.02, 2.5
T_START = time.time()


def score(pool, th):
    e96 = np.array(pool.map(P6._full, [(list(th), k) for k in P6.T96]))
    import plot_spin_paper_figs as PF
    H = PF.heldout_points()
    eh = np.array(pool.map(P6._full, [(list(th), k) for k in H]))
    LQ = [(q, c) for q in (2.5, 2.0) for c in S.CHIS]
    el = np.array(pool.map(P6._full, [(list(th), k) for k in LQ]))
    col = {f"{ch:+.1f}": float(np.median([v for (q, c), v in zip(P6.T96, e96) if c == ch])) for ch in S.CHIS}
    row = {f"{q:.3f}": float(np.median([v for (qq, c), v in zip(P6.T96, e96) if qq == q])) for q in S.TRAIN_Q}
    return dict(T96=[float(e96.mean()), float(np.median(e96)), float(e96.max())],
                H270=[float(eh.mean()), float(np.median(eh)), float(eh.max())], col_median=col, row_median=row,
                lowq={f"{q:.1f}_{c:+.1f}": float(v) for (q, c), v in zip(LQ, el)},
                T96_cases={f"{q:.6f}_{c:+.4f}": float(v) for (q, c), v in zip(P6.T96, e96)})


def main():
    pool = mp.get_context("fork").Pool(96)
    e_base = float(np.mean(pool.map(P6._fit, [(list(TH0), k) for k in P6.T96])))
    print(f"leader 27c fit-stride T96 mean {e_base:.4e}", flush=True)
    out = dict(base_fit_mean=e_base, names=NAMES, round1={}, CUT=CUT)
    cands = [i for i in FREE0 if NAMES[i] not in SKIP]
    for i in cands:
        f0 = TH0.copy(); f0[i] = 0.0
        th, e = A.refit(pool, f0, [j for j in FREE0 if j != i], STEP_MIN, NAMES[i])
        out["round1"][NAMES[i]] = dict(fit_mean=e, cost=e / e_base - 1)
        print(f"{NAMES[i]:10s} cost {100 * (e / e_base - 1):+6.2f} %", flush=True)
        (OUT / "p22_prune.json").write_text(json.dumps(out, indent=1))
    zero = sorted(n for n, r in out["round1"].items() if r["cost"] < CUT)
    f0 = TH0.copy()
    for n in zero:
        f0[NAMES.index(n)] = 0.0
    free = [j for j in FREE0 if NAMES[j] not in zero]
    left = 110 - (time.time() - T_START) / 60 - 7
    th, e = A.refit(pool, f0, free, max(left, 8), "joint")
    sc = score(pool, th)
    out["round2"] = dict(zeroed=zero, n_coef=len(free), free=free, theta=list(map(float, th)), **sc)
    (OUT / "p22_prune.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"pruned {len(free)}c (zeroed {zero}): T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   column medians: " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    print("   row medians: " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for k, x in sc["lowq"].items() if k.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
