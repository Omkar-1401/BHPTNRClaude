"""TEST, NOT A MODEL: fluxanchored_spin with beta driven by J(t) + k*Jdot(t) instead of E(t).

Why (2026-09-27).  beta_shape_projection.py (no mismatch fit, measured beta(t) on 8 spin
cases, window to +10 M) found that the shipped beta shape {1, E} leaves 3.5% rms in the
merger band, and that {1, J + k*Jdot} with ONE k shared by all cases cuts it to 2.5%
(overall 1.11% -> 0.95%, best k = 0.70); a rate term on top of E does nothing (best k -> 0).
Jdot carries no new spin information (2 Edot/(omega Jdot) = 1 to 1e-5 in the inspiral); what it
adds is a merger-peaked profile next to the cumulative one.  This asks whether that survives
with the coefficients shared across the grid and scored by the model's own mismatch.

Arms, on the SAME subset (so the subset's own gain is measured, not credited):
    --arm control   shipped fluxanchored_spin: beta = b_pp*(1 + b_E*E(t)), 9 coefficients
    --arm jdrive    beta = b_pp*(1 + b_E*D(t)),  D = E_rad*[J(t)/J_rad + k*Jdot(t)/max Jdot]
                    9 + k = 10 coefficients
Both start from fluxanchored_spin_results/coeffs_nu.json; alpha is identical in both.
jdrive START (smoke test, 4 cases): the shipped P is tuned to E, and J/J_rad already sits at
0.16-0.26 by t = -5000 M and 0.88-0.92 at the peak (E/E_rad: 0.02-0.09 and 0.63-0.73), so the
shipped theta put through the J drive gives mismatch 0.14-0.7.  A coarse scan of
(P x lam, k, b + db) found lam = 0.1, k = 2.0, db = 0 at mean 2.76e-3 against the control's
3.86e-3 on the same 4 cases, so jdrive starts there (--p-scale, --k0).  The projection's k = 0.70
came from a -1000..+10 M window that missed the long inspiral, where J's extra drift costs most.

D's normalisation (a choice, stated): J and Jdot are ppBHPT quantities in the same units as the
shipped E (t, h scaled by X1 first: per unit total mass), each profile normalised per case,
and the sum carried at E's total radiated energy E_rad, so at k = 0 D is "E's amplitude with J's
time profile" and P keeps its meaning and scale.  Jdot is the signed (2,2)+(2,-2) flux
(1/16pi) sum m Im[h conj(hdot)], positive in the inspiral, despiked in the 2d join window like
E_dot.  Before trusting it, the same finite-difference code must reproduce gw_remnant's
E_dot to 1e-3 on every case.

Writes test_beta_jdrive_results/<arm>.json only; never touches coeffs_nu.json.
    python test_beta_jdrive.py --arm control --budget 95
    python test_beta_jdrive.py --arm jdrive  --budget 95
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import physics_common as PC
import fit_fluxanchored_spin as FS
FL = FS.FL

OUT = HERE / "test_beta_jdrive_results"; OUT.mkdir(exist_ok=True)
SUB_Q = [3.0, 5.0, 8.0]
SUB_CHI = [-0.6, -0.2, 0.0, 0.4, 0.6, 0.8]
_CASES: dict = {}


def key(q, c):
    return f"{q:.4f}_{c:+.1f}"


def j_profiles(t, h, q, e_dot_gwr):
    """Signed J_z flux and cumulative J on the per-M clock; checks E_dot vs gw_remnant."""
    x1 = q / (1.0 + q)
    tc, hc = t * x1, h * x1                      # same scaling as PC.energy_drive(e_units="M")
    hd = np.gradient(hc, tc)
    ed_mine = 2.0 * np.abs(hd) ** 2 / (16 * np.pi)
    ed_ref = PC.despike(np.asarray(e_dot_gwr, float), t, "2d")[0]
    ed_mine = PC.despike(ed_mine, t, "2d")[0]
    err = float(np.max(np.abs(ed_mine - ed_ref)) / np.max(ed_ref))
    jd = 4.0 * np.imag(hc * np.conj(hd)) / (16 * np.pi)
    ipk = int(np.argmax(np.abs(h)))
    if np.median(jd[:ipk]) < 0:
        jd = -jd
    jd = PC.despike(jd, t, "2d")[0]
    J = cumulative_trapezoid(jd, tc, initial=0.0)
    return J, jd, err


def build(q, c):
    case = FS.build_case(q, c)
    t = np.asarray(case["t_bhpt"], float)
    co = FS.G.gwr_energy_coordinates(t, case["h_bhpt"], q, surrogate="2d", e_units="M")
    J, jd, err = j_profiles(t, case["h_bhpt"], q, co["e_dot"])
    if err > 1e-3:
        sys.exit(f"q={q} chi1={c}: E_dot from this code differs from gw_remnant's by {err:.1e}")
    E = case["losses"]["e_oft"]
    case["jprof"] = {"E_rad": float(E[-1]), "Jn": J / J[-1], "Jdn": jd / np.max(jd)}
    case["edot_check"] = err
    return case


def with_drive(case, k):
    """Case copy whose e_oft is the J drive D(t) at this k (alpha's flux_hat untouched)."""
    jp = case["jprof"]
    D = jp["E_rad"] * (jp["Jn"] + k * jp["Jdn"])
    c2 = dict(case)
    c2["losses"] = {"e_oft": D, "flux_hat": case["losses"]["flux_hat"]}
    tf = case["fit_t_bhpt"]
    c2["fit_losses"] = {"e_oft": np.interp(tf, case["t_bhpt"], D),
                        "flux_hat": case["fit_losses"]["flux_hat"]}
    return c2


def _work(a):
    th, arm, qc, t0 = a
    q, c = qc
    case = _CASES[qc]
    if arm == "jdrive":
        case = with_drive(case, th[9])
    p = FS.params_at(q, c, th[:9])
    if t0 is None:
        g = np.linspace(-250.0, 150.0, 81)
        e0 = [FL.fast_mismatch(p, case, s, "fit")[0] for s in g]
        t0 = float(g[int(np.argmin(e0))])
    _, t0 = FL.fast_mismatch(p, case, t0, "fit")
    return FL.fast_mismatch(p, case, t0, "full")[0], t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "jdrive"], required=True)
    ap.add_argument("--budget", type=float, default=95.0, help="optimiser wall-clock minutes")
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--nproc", type=int, default=18)
    ap.add_argument("--p-scale", type=float, default=0.1, help="jdrive: start P0, P1, s1, s2 x this")
    ap.add_argument("--k0", type=float, default=2.0, help="jdrive: start k")
    ap.add_argument("--tag", default="", help="suffix for the output files (extra starts)")
    a = ap.parse_args()
    keys = [(q, c) for q in SUB_Q for c in SUB_CHI]
    ship = json.loads((FS.RESULTS / "coeffs_nu.json").read_text())
    assert ship["weight"] == "nu" and FS.WEIGHT == "nu"
    th = np.array(ship["theta"], float)
    names = list(FS.NAMES)
    if a.arm == "jdrive":
        th[5:9] *= a.p_scale
        th = np.r_[th, a.k0]; names.append("k")
    print(f"TEST fluxanchored_spin beta drive, arm = {a.arm} ({len(th)} coefficients); "
          f"{len(SUB_Q)} q x {len(SUB_CHI)} chi1 = {len(keys)} points", flush=True)

    t0 = time.time()
    for k in keys:
        _CASES[k] = build(*k)
    print(f"cases ready in {time.time()-t0:.0f}s; E_dot check vs gw_remnant: max "
          f"{max(c['edot_check'] for c in _CASES.values()):.1e}", flush=True)
    pool = mp.get_context("fork").Pool(min(a.nproc, len(keys)))
    nuis = {k: None for k in keys}

    def score(x):
        r = pool.map(_work, [(np.asarray(x, float), a.arm, k, nuis[k]) for k in keys])
        for k, (_, t0k) in zip(keys, r):
            nuis[k] = t0k
        return np.array([e for e, _ in r])

    # reproduction: the installed coefficients through the control path vs their stored values
    e_ship = np.array(pool.map(_work, [(np.array(ship["theta"]), "control", k, None) for k in keys]))[:, 0]
    stored = np.array([ship["mismatches"][key(*k)] for k in keys])
    ratio = e_ship.mean() / stored.mean()
    print(f"installed coeffs on this subset: mean {e_ship.mean():.4e} vs stored "
          f"{stored.mean():.4e} (ratio {ratio:.5f})", flush=True)
    if abs(ratio - 1) > 1e-2:
        print("  NOTE: stored values not reproduced to 1% -- stored file predates the current "
              "drive; both arms still start from the same theta, so the comparison holds", flush=True)
    e0 = score(th)
    print(f"start ({a.arm}): mean {e0.mean():.4e} median {np.median(e0):.4e} "
          f"max {e0.max():.4e}", flush=True)

    st = {"n": 0, "best": np.inf, "bx": th.copy(), "t0": time.time()}
    part = OUT / f"{a.arm}{a.tag}_partial.json"

    class _Budget(Exception):
        pass

    def obj(x):
        e = score(x); m = float(e.mean()); st["n"] += 1; tag = ""
        el = (time.time() - st["t0"]) / 60
        if m < st["best"]:
            st["best"], st["bx"] = m, np.array(x, float); tag = "  <-- best"
            part.write_text(json.dumps({"n_eval": st["n"], "names": names,
                                        "theta": st["bx"].tolist(), "mean": m}, indent=2))
        if tag or st["n"] % 25 == 0:
            print(f"  [{st['n']:4d}] {el:6.1f}m mean={m:.4e} median={np.median(e):.4e} "
                  f"max={e.max():.4e}{tag}", flush=True)
        if el > a.budget:
            raise _Budget
        return m

    try:
        r = minimize(obj, th, method="Powell",
                     options={"maxiter": a.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        note = f"CONVERGED / {st['n']} evals: {r.message}"
    except _Budget:
        note = f"STOPPED ON BUDGET / {st['n']} evals"
    print(f"  {note}", flush=True)
    e = score(st["bx"])
    print("\nfitted: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(names, st["bx"])))
    per = dict(zip(keys, e)); per0 = dict(zip(keys, e0))
    print(f"\n{'chi1':>6} " + " ".join(f"{'q=%g' % q:>19}" for q in SUB_Q) + "   (start -> fitted)")
    for c in SUB_CHI:
        print(f"{c:+6.1f} " + " ".join(f"{per0[(q, c)]:9.2e}->{per[(q, c)]:8.2e}" for q in SUB_Q))
    print(f"\nSUBSET {len(keys)}: start mean {e0.mean():.4e} median {np.median(e0):.4e} "
          f"max {e0.max():.4e}\n           fitted mean {e.mean():.4e} median "
          f"{np.median(e):.4e} max {e.max():.4e}")
    out = OUT / f"{a.arm}{a.tag}.json"
    out.write_text(json.dumps({
        "test": "fluxanchored_spin beta drive E vs J + k*Jdot (NOT a model)",
        "arm": a.arm, "names": names, "p_scale": a.p_scale, "k0": a.k0, "theta": st["bx"].tolist(), "note": note,
        "energy_drive": PC.ENERGY_DRIVE_2D, "grid": {"q": SUB_Q, "chi1": SUB_CHI},
        "start": {"theta": th.tolist(), "mismatches": {key(*k): float(v) for k, v in per0.items()}},
        "installed_on_subset": {"mean": float(e_ship.mean()), "stored_mean": float(stored.mean())},
        "mismatches": {key(*k): float(v) for k, v in per.items()},
        "mean": float(e.mean()), "median": float(np.median(e)), "max": float(e.max()),
    }, indent=2))
    print(f"wrote {out}")
    pool.close()


if __name__ == "__main__":
    main()
