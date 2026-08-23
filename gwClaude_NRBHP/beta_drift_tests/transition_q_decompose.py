"""WHY does beta's drift change sign near q=4?

The transition q is a zero crossing of a FITTED quantity, so it is a RATIO, not a scale.
`inspiral_only_sign.py` established that the two ends of the window disagree about the sign:
the inspiral wants b_E > 0 at every q, the merger-ringdown wants b_E < 0.  The full-window
optimum is their weighted compromise, so the crossing sits where the two pulls cancel.

This script measures both pulls on the SAME footing so the crossing can be predicted rather
than asserted.  At fixed t0 and phi0 the mismatch numerator is exactly additive in time, so

    err_full(b) = N_i(b)/(2 n1) + N_m(b)/(2 n1)        split at T_SPLIT

with each N_r approximately quadratic near its own minimum:

    N_r(b)/(2 n1) ~ e_r + k_r (b - b_r*)^2

Then the full-window optimum is b* = (k_i b_i* + k_m b_m*)/(k_i + k_m), and the transition q
is where k_i b_i* = -k_m b_m*, i.e. where the two TORQUES k_r*b_r* balance.  Reading off
b_r*, k_r vs q says which factor actually moves and therefore what sets ~4.

Usage:  python transition_q_decompose.py [--drive flux|e] [--split -200]
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG

# QS is taken from the per-q cache itself (full precision -- the waveform cache filename
# is built from q, so a rounded value misses the file), so the 3 held parameters are the
# per-q optimum at exactly that q.


def split_err(p, case, t0_seed, t_split, shape):
    """(err_total, err_inspiral, err_merger, t0) on the FULL window, the last two summing
    to the first.  t0 optimised on the full window; phi0 analytic on the full window; the
    residual is then split in NR time at t_split."""
    tb, hb = case["t_bhpt"], case["h_bhpt"]
    tn, hn, los = case["t_nr"], case["h_nr"], case["losses"]
    alpha, tau_shape, dmin = shape(p, tb, los)
    if dmin <= 0:
        return 50.0, 25.0, 25.0, t0_seed
    f = lambda t0: GG.err_at_t0(t0, alpha, tau_shape, hb, tn, hn)
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 50})
    t0 = float(r.x)
    tau = t0 + tau_shape
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    tc, href = tn[m], hn[m]
    g = np.interp(tc, tau, hb.real) + 1j * np.interp(tc, tau, hb.imag)
    g = np.interp(tc, tau, alpha) * g
    n1 = np.sum(np.abs(href) ** 2)
    phi0 = np.angle(np.sum(href * np.conj(g)))          # analytic optimum, held fixed
    res = np.abs(href - np.exp(1j * phi0) * g) ** 2      # additive in time
    ins = float(np.sum(res[tc <= t_split]) / (2.0 * n1))
    mrg = float(np.sum(res[tc > t_split]) / (2.0 * n1))
    return ins + mrg, ins, mrg, t0


def parab(b, y):
    """Vertex and curvature of the least-squares parabola through (b, y)."""
    c = np.polyfit(b, y, 2)
    if c[0] <= 0:
        return np.nan, np.nan, np.nan
    bstar = -c[1] / (2.0 * c[0])
    return float(bstar), float(c[0]), float(np.polyval(c, bstar))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="flux", choices=("flux", "e"))
    ap.add_argument("--split", type=float, default=-200.0)
    args = ap.parse_args()

    if args.drive == "flux":
        cache = json.loads((HERE.parent / "gwr_energy_flux_results" /
                            "per_q_cache_flux.json").read_text())
        shape = FL.model_shape
    else:
        cache = json.loads((HERE.parent / "gw_remnant_energy_results" /
                            "per_q_cache_mult.json").read_text())
        shape = lambda p, tb, los: G.model_shape_mult(p, tb, los) \
            if hasattr(G, "model_shape_mult") else None

    recs = sorted(cache.values(), key=lambda v: v["q"])
    qs_all = [float(v["q"]) for v in recs]
    print(f"drive = {args.drive}   split at t_NR = {args.split:+.0f} M   "
          f"{len(qs_all)} q from the per-q cache\n")
    hdr = (f"{'q':>5s} {'b*(full)':>9s} {'b_i*':>8s} {'b_m*':>8s} "
           f"{'k_i':>10s} {'k_m':>10s} {'k_i b_i*':>10s} {'k_m b_m*':>10s} "
           f"{'b* pred':>9s} {'w_i%':>6s}")
    print(hdr); print("-" * len(hdr))
    out = {}
    for ref in recs:
        q = float(ref["q"])
        p0 = np.array(ref["params"][:4], float); t0 = float(ref["t0"])
        case = FL.add_flux(G.load_case(q, 6, 10))
        b_opt = p0[3]
        bs = b_opt + np.linspace(-2.5, 2.5, 11)
        Ei, Em, Et = [], [], []
        for b in bs:
            p = p0.copy(); p[3] = b
            tot, i_, m_, t0 = split_err(p, case, t0, args.split, shape)
            Ei.append(i_); Em.append(m_); Et.append(tot)
        Ei, Em, Et = map(np.asarray, (Ei, Em, Et))
        bi, ki, ei = parab(bs, Ei)
        bm, km, em = parab(bs, Em)
        bt, kt, et = parab(bs, Et)
        pred = (ki * bi + km * bm) / (ki + km)
        w_i = ei / (ei + em) * 100.0
        print(f"{q:5.2f} {bt:9.3f} {bi:8.3f} {bm:8.3f} {ki:10.3e} {km:10.3e} "
              f"{ki*bi:+10.3e} {km*bm:+10.3e} {pred:9.3f} {w_i:6.1f}", flush=True)
        out[f"{q:g}"] = dict(b_full=bt, b_ins=bi, b_mrg=bm, k_ins=ki, k_mrg=km,
                             e_ins=ei, e_mrg=em, b_pred=pred, b_cache=float(b_opt))

    # where does the predicted compromise cross zero?
    qs = np.array([float(k) for k in out]); pr = np.array([out[k]["b_pred"] for k in out])
    s = np.where(np.diff(np.sign(pr)))[0]
    if len(s):
        i = s[0]
        qc = qs[i] + (qs[i+1]-qs[i]) * (-pr[i]) / (pr[i+1]-pr[i])
        print(f"\npredicted transition q (torques balance): {qc:.3f}")
        out["q_transition_predicted"] = float(qc)
    (HERE / f"transition_q_decompose_{args.drive}_{int(args.split)}.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("wrote", HERE / f"transition_q_decompose_{args.drive}_{int(args.split)}.json")


if __name__ == "__main__":
    main()
