"""Split-model exploration (2026-10-12; user: explore with the split, < 15 coefficients, validate on H270).

Every variant keeps the split protocol of this directory (README "Decisions"): Kerr-ISCO cut, E_insp / E_MR each with
its own NR norm, one shared t0/phi0, Ori–Thorne window blend S (T = -1 to the ppBHPT peak, eta = 1/q), two-stage fit
(stage 1: inspiral coefficients on mean E_insp; stage 2: MR coefficients on mean E_MR, inspiral frozen), all on T96.
Scored on T96, H270 (validation) and q = 2, 2.5 (8 spins each; ppBHPT extrapolated there) with full-window E (full
alignment, the published comparison) plus the split errors.  Variants only: nothing installed, spin_dep/ not edited.

    alpha = (1 - S) alpha_i + S alpha_m,      beta = (1 - S) beta_i + S beta_m
    inspiral copy (flux form + derived SO):   alpha_i = a_pp (1 + a_F F),  beta_i = b_pp (1 + b_e E)(1 + kappa_SO chi1 x^(3/2))
    (a_pp, a_F, b_pp, b_e) = fit_fluxanchored_spin.params_at(q, chi1, theta), theta = c0 c1 A0 A1 b P0 P1 s1 s2

Variant spec (VARIANTS below):
    insp_free : indices of theta_i refitted in stage 1 (None = keep the T96 stage-1 theta_i of flux_spin_split_T96_T-1.json);
                frozen indices sit at FROZEN values (default: the stored theta_i)
    mr        : MR-branch form name (MR_FORMS) -> its free coefficients and start
Coefficient count = len(free or stored inspiral, i.e. 9 minus indices pinned to 0) + MR count; asserted <= 14.

    python explore_split.py --variant NAME [--variant NAME ...]
-> results/explore/<NAME>.json ; summary lines in the log; EXPLORE_LOG.md is written by hand from these.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extrap_lowq as XL                    # installs the cache-safe D.get_case (writes only into spin_dep_split/)
import slim_split as SL                     # installs the masked-MR ab into fit_flux_spin_split
import score_h270 as H
FF, M = SL.FF, SL.M
SO, FS = FF.SO, FF.FS
OUT = HERE / "results" / "explore"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = list(FS.NAMES)                       # c0 c1 A0 A1 b P0 P1 s1 s2
T96 = [(q, c) for q in M.S.TRAIN_Q for c in M.S.CHIS]
LOWQ = XL.CASES_LIST
TI_T96 = np.array(json.loads((HERE / "results" / "flux_spin_split_T96_T-1.json").read_text())["theta_insp"])
SLIM = json.loads((HERE / "results" / "slim_confirm_aF_bPP_S.json").read_text())
SLIM_X = dict(zip(["A0", "A1", "b", "s1", "s2"], SLIM["x"]))


# ---------------------------------------------------------------- MR branch forms
# Each: (names of free coefficients, start(ti) -> x0, fn(x, ti, q, chi, c, pre) -> (alpha_m, beta_m)).
def _copy(th, q, chi, c, pre):
    los = c[pre + "losses"]
    a_pp, a_f, b_pp, b_e = FS.params_at(q, chi, th)
    return a_pp, a_f, b_pp, b_e, los["flux_hat"], los["e_oft"]


def _mask(names):
    idx = [NAMES.index(n) for n in names]
    def fn(x, ti, q, chi, c, pre):
        th = np.array(ti, float).copy(); th[idx] = x
        a_pp, a_f, b_pp, b_e, F, E = _copy(th, q, chi, c, pre)
        return a_pp * (1 + a_f * F), b_pp * (1 + b_e * E)
    start = lambda ti: np.array([SLIM_X.get(n, ti[NAMES.index(n)]) for n in names])
    return names, start, fn


def _with_extra(base_names, extra_names, extra_fn):
    """Masked copy on base_names plus extra coefficients applied by extra_fn(alpha, beta, xe, q, chi, F, E, nu)."""
    idx = [NAMES.index(n) for n in base_names]
    def fn(x, ti, q, chi, c, pre):
        th = np.array(ti, float).copy(); th[idx] = x[:len(idx)]
        a_pp, a_f, b_pp, b_e, F, E = _copy(th, q, chi, c, pre)
        nu = q / (1 + q) ** 2
        return extra_fn(a_pp * (1 + a_f * F), b_pp * (1 + b_e * E), x[len(idx):], q, chi, F, E, nu, a_pp, a_f, b_pp, b_e)
    start = lambda ti: np.r_[[SLIM_X.get(n, ti[NAMES.index(n)]) for n in base_names], np.zeros(len(extra_names))]
    return base_names + extra_names, start, fn


# ---------------------------------------------------------------- derived remnant spin functions (round 3)
# D_chi(q, chi1) = chi_f(q, chi1) - chi_f(q, 0)            remnant-spin shift, NRSur3dq8Remnant
# D_br(q, chi1)  = beta_r(q, chi1) / beta_r(q, 0) - 1      ringdown clock-ratio shift (exact Kerr QNMs + remnant fit)
# Both are 0 at chi1 = 0 and contain no fitted parameter.  They replace the MR copy's s1 chi1 + s2 chi1^2 in P.
_REM = {}


def rem(q, chi):
    k = (round(float(q), 10), round(float(chi), 10))
    if k not in _REM:
        F = M.S._SFBH
        cf = float(F.chif(q, [0, 0, chi], [0, 0, 0])[0][2]); c0 = float(F.chif(q, [0, 0, 0], [0, 0, 0])[0][2])
        _REM[k] = dict(D_chi=cf - c0, D_br=M.S.beta_r_phys(q, chi) / M.S.beta_r_phys(q, 0.0) - 1.0)
    return _REM[k]


def _remform(dname, base_names, n_r):
    """MR copy: free base_names; its s1, s2 set to 0 unless freed; extra P term nu (r1 D + r2 D^2) on beta's E coupling."""
    idx = [NAMES.index(n) for n in base_names]
    zero = [NAMES.index(n) for n in ("s1", "s2") if n not in base_names]
    def fn(x, ti, q, chi, c, pre):
        th = np.array(ti, float).copy(); th[zero] = 0.0; th[idx] = x[:len(idx)]
        a_pp, a_f, b_pp, b_e, F, E = _copy(th, q, chi, c, pre)
        nu = q / (1 + q) ** 2; D = rem(q, chi)[dname]
        dP = nu * sum(x[len(idx) + j] * D ** (j + 1) for j in range(n_r))
        return a_pp * (1 + a_f * F), b_pp * (1 + b_e * E) + dP * E
    def start(ti):
        # least-squares projection of the slim MR spin term nu (s1 chi + s2 chi^2) (minus any freed s1 chi) onto D, D^2 over T96
        rows, rhs = [], []
        for q, chi in T96:
            D = rem(q, chi)[dname]; tgt = SLIM_X["s1"] * chi + SLIM_X["s2"] * chi ** 2
            if "s1" in base_names:
                tgt -= SLIM_X["s1"] * chi
            rows.append([D ** (j + 1) for j in range(n_r)]); rhs.append(tgt)
        r = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        return np.r_[[SLIM_X.get(n, ti[NAMES.index(n)]) for n in base_names], r]
    return base_names + [f"r{j + 1}[{dname}]" for j in range(n_r)], start, fn


MR_FORMS = {
    # the slim split (gate: reproduces results/slim_confirm_aF_bPP_S.json)
    "slim": _mask(["A0", "A1", "b", "s1", "s2"]),
    # other 5-coefficient subsets of the copy
    "aF_P_S": _mask(["A0", "A1", "P0", "s1", "s2"]),
    "aF_bPP_P": _mask(["A0", "A1", "b", "P0", "P1"]),
    "aF_bPP_P0_s1": _mask(["A0", "A1", "b", "P0", "s1"]),
    # merger bump in beta driven by the flux: beta_m *= 1 + nu (k0 + k1 chi1) F      (aF, b, k0, k1 = 5)
    "bump": _with_extra(["A0", "A1", "b"], ["k0", "k1"],
                        lambda al, be, xe, q, chi, F, E, nu, *r: (al, be * (1 + nu * (xe[0] + xe[1] * chi) * F))),
    # bump + one spin term in P                                                       (aF, b, s1, k0 = 5)
    "bump_s1": _with_extra(["A0", "A1", "b", "s1"], ["k0"],
                           lambda al, be, xe, q, chi, F, E, nu, *r: (al, be * (1 + nu * xe[0] * F))),
    # spin in the MR amplitude coupling: a_F -> a_F + nu a_s chi1                      (aF, b, s1, a_s = 5)
    # round 3: the MR spin dependence from derived remnant functions instead of a chi1 polynomial
    "rem_dchi": _remform("D_chi", ["A0", "A1", "b"], 1),
    "rem_dchi2": _remform("D_chi", ["A0", "A1", "b"], 2),
    "rem_dbr": _remform("D_br", ["A0", "A1", "b"], 1),
    "rem_dbr2": _remform("D_br", ["A0", "A1", "b"], 2),
    "rem_dbr_s1": _remform("D_br", ["A0", "A1", "b", "s1"], 1),
    # wider MR copies, for variants that free inspiral coefficients
    "aF_bPP_S_P": _mask(["A0", "A1", "b", "s1", "s2", "P0", "P1"]),
    "aF_bPP_S_P0": _mask(["A0", "A1", "b", "s1", "s2", "P0"]),
    "alpha_spin": _with_extra(["A0", "A1", "b", "s1"], ["a_s"],
                              lambda al, be, xe, q, chi, F, E, nu, a_pp, a_f, *r: (a_pp * (1 + (a_f + nu * xe[0] * chi) * F), be)),
}


# ---------------------------------------------------------------- variants
VARIANTS_BASE = {"slim": dict(insp=None, mr="slim"), "slim__win-2.0": dict(insp=None, mr="slim", window=-2.0),
                 "insp8b_mr6": dict(insp=("pin0", ["b"]), mr="aF_bPP_S_P0"),
                 "insp8b__win-2.0": dict(insp=("pin0", ["b"]), mr="aF_bPP_S_P0", window=-2.0)}
# insp: None = stored T96 stage-1 theta_i (9c, no stage 1);  ("pin0", [names]) = those pinned to 0, rest refitted.
VARIANTS = {
    "slim": dict(insp=None, mr="slim"),
    "aF_P_S": dict(insp=None, mr="aF_P_S"),
    "aF_bPP_P": dict(insp=None, mr="aF_bPP_P"),
    "aF_bPP_P0_s1": dict(insp=None, mr="aF_bPP_P0_s1"),
    "bump": dict(insp=None, mr="bump"),
    "bump_s1": dict(insp=None, mr="bump_s1"),
    "alpha_spin": dict(insp=None, mr="alpha_spin"),
    "rem_dchi": dict(insp=None, mr="rem_dchi"),
    "rem_dchi2": dict(insp=None, mr="rem_dchi2"),
    "rem_dbr": dict(insp=None, mr="rem_dbr"),
    "rem_dbr2": dict(insp=None, mr="rem_dbr2"),
    "rem_dbr_s1": dict(insp=None, mr="rem_dbr_s1"),
    # round 4: combinations of the round-2/3 gains
    "slim__win-3.0": dict(insp=None, mr="slim", window=-3.0),
    "slim__win-1.5": dict(insp=None, mr="slim", window=-1.5),
    "slim__win-4.0": dict(insp=None, mr="slim", window=-4.0),
    "insp8__win-2.0": dict(insp=("pin0", ["s2"]), mr="aF_bPP_S_P0", window=-2.0),
    "insp8b__win-2.0": dict(insp=("pin0", ["b"]), mr="aF_bPP_S_P0", window=-2.0),
    "slim_win-2.0__joint": dict(insp=None, mr="slim", window=-2.0, **{"from": "slim__win-2.0"}, joint=True, budget_min=30),
    "insp8b__joint": dict(insp=("pin0", ["b"]), mr="aF_bPP_S_P0", **{"from": "insp8b_mr6"}, joint=True, budget_min=30),
    "insp8b_win-2.0__joint": dict(insp=("pin0", ["b"]), mr="aF_bPP_S_P0", window=-2.0, **{"from": "insp8b__win-2.0"},
                                  joint=True, budget_min=30),
    # round 5: half-split cross-validation of the top candidates (fit on one 48-point half of T96, score the other)
    **{f"{base}__half{h}": dict(VARIANTS_BASE[base], half=h,      # stage 1 always refit on the half (no leak)
                                insp=VARIANTS_BASE[base]["insp"] or ("pin0", [])) for base in ("slim", "slim__win-2.0", "insp8b_mr6", "insp8b__win-2.0")
       for h in (0, 1)},
    # trade inspiral coefficients for MR ones (stage 1 refit with the pinned ones at 0)
    "insp7_mr7": dict(insp=("pin0", ["c0", "c1"]), mr="aF_bPP_S_P"),
    "insp8_mr6": dict(insp=("pin0", ["s2"]), mr="aF_bPP_S_P0"),
    "insp8b_mr6": dict(insp=("pin0", ["b"]), mr="aF_bPP_S_P0"),
}


def ab(kind, ti, tm, q, chi, c, key):
    if kind == "split" and isinstance(tm, tuple) and tm[0] == "form":
        _, name, x = tm
        pre = "fit_" if key == "fit" else ""
        S = c[pre + "S"]
        a_pp, a_f, b_pp, b_e, F, E = _copy(ti, q, chi, c, pre)
        xc = SO.xnote(q, c, pre)
        ai = a_pp * (1 + a_f * F); bi = b_pp * (1 + b_e * E) * (1 + SO.so_coef(q, corrected=True) * chi * xc ** 1.5)
        am, bm = MR_FORMS[name][2](np.asarray(x, float), ti, q, chi, c, pre)
        return (1 - S) * ai + S * am, (1 - S) * bi + S * bm
    return SL.ab(kind, ti, tm, q, chi, c, key)


FF.ab = ab


def nm(pool, obj, x0, maxrep=5):
    x = np.array(x0, float); best = obj(x)
    for rep in range(maxrep):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 900 * len(x), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
        fr = obj(r.x, wide=True)
        done = fr > best * (1 - 1e-3)
        if fr < best:
            x, best = r.x, fr
        if done:
            break
    return x, best


def stage1(pool, ti0, pin):
    """Refit the inspiral copy with the named coefficients pinned to 0; MR copy = inspiral copy (no blend effect)."""
    free = [i for i, n in enumerate(NAMES) if n not in pin]
    base = np.array(ti0, float).copy(); base[[NAMES.index(n) for n in pin]] = 0.0
    def full(x):
        th = base.copy(); th[free] = x; return th
    def obj(x, wide=False):
        th = full(x); R = FF.run(pool, "split", th, th, wide=wide)
        return float(np.mean([R[k]["errs"][0] for k in FF.KEYS]))
    obj(base[free], wide=True)
    x, best = nm(pool, obj, base[free])
    return full(x), best


def stage2(pool, ti, mr):
    names, start, _ = MR_FORMS[mr]
    def obj(x, wide=False):
        R = FF.run(pool, "split", ti, ("form", mr, list(x)), wide=wide)
        return float(np.mean([R[k]["errs"][1] for k in FF.KEYS]))
    x0 = start(ti); obj(x0, wide=True)
    return nm(pool, obj, x0)


def joint(pool, ti, x, spec, budget_min):
    """Polish ALL the variant's coefficients together on mean full-window E ('full' alignment) -- a trade-off
    measurement on the split form, NOT the split protocol (whose fit is two-stage)."""
    pin = spec["insp"][1] if spec["insp"] else []
    free = [i for i, n in enumerate(NAMES) if n not in pin]; nf = len(free)
    def unpack(z):
        th = np.array(ti, float).copy(); th[free] = z[:nf]; return th, z[nf:]
    st = dict(best=np.inf, z=None, t=time.time())
    def obj(z, wide=False):
        if time.time() - st["t"] > budget_min * 60:
            raise TimeoutError
        th, xm = unpack(z)
        R = FF.run(pool, "split", th, ("form", spec["mr"], list(xm)), "full", "fit", wide)
        f = float(np.mean([R[k]["errs"][2] for k in FF.KEYS]))
        if f < st["best"]:
            st["best"], st["z"] = f, np.array(z, float)
        return f
    z0 = np.r_[np.asarray(ti)[free], x]; e0 = obj(z0, wide=True); conv = True
    try:
        nm(pool, obj, z0, maxrep=3)
    except TimeoutError:
        conv = False
    th, xm = unpack(st["z"])
    return th, xm, e0, st["best"], conv


def set_window(start):
    """Rebuild the blend S of every prepared case for a window starting at OT time T = start (end: ppBHPT peak)."""
    OT = M.OT; OT.WINDOW_START = float(start)
    for k, c in M.CASES.items():
        Sb, _ = OT.blend(c["t_bhpt"], c["h_bhpt"], *k)
        c["S"] = Sb; c["fit_S"] = np.interp(c["fit_t_bhpt"], c["t_bhpt"], Sb)


def stats(rows, key="full_E_full"):
    v = np.array([r[key] for r in rows]); return dict(median=float(np.median(v)), mean=float(v.mean()), max=float(v.max()))


def halves():
    """The same two 48/48 halves of T96 as spin_dep/overfit_checks_flux_orbit.py's split 0 (seed 1)."""
    p = np.random.default_rng(1).permutation(len(T96))
    return [T96[j] for j in p[:48]], [T96[j] for j in p[48:]]


def run_variant(pool, name):
    spec = VARIANTS[name]; t = time.time(); log = []
    train = T96 if "half" not in spec else halves()[spec["half"]]
    other = [] if "half" not in spec else halves()[1 - spec["half"]]
    FF.KEYS = train; FF.SEEDS.clear()
    if spec.get("from"):                                   # start from a finished variant, skip the two stages
        src = json.loads((OUT / f"{spec['from']}.json").read_text())
        ti = np.array(src["theta_insp"]); x = np.array(src["x_mr"])
        n_insp = 9 - (len(spec["insp"][1]) if spec["insp"] else 0)
    elif spec["insp"] is None:
        ti, n_insp = TI_T96.copy(), 9
    else:
        pin = spec["insp"][1]; ti, e1 = stage1(pool, TI_T96, pin); n_insp = 9 - len(pin)
        log.append(dict(stage="insp", mean_E_insp=e1, seconds=round(time.time() - t)))
        print(f"  [{name}] stage 1 ({n_insp}c): mean E_insp {e1:.4e}", flush=True)
    n_mr = len(MR_FORMS[spec["mr"]][0]); ncoef = n_insp + n_mr
    assert ncoef <= 14, (name, ncoef)
    if not spec.get("from"):
        FF.KEYS = train; FF.SEEDS.clear(); x, e2 = stage2(pool, ti, spec["mr"])
        log.append(dict(stage="mr", mean_E_MR=e2, seconds=round(time.time() - t)))
        print(f"  [{name}] stage 2 ({n_mr} MR coef): mean E_MR {e2:.4e}  ({time.time() - t:.0f}s)", flush=True)
    if spec.get("joint"):
        FF.KEYS = train; FF.SEEDS.clear(); ti, x, e0, e3, conv = joint(pool, ti, x, spec, spec.get("budget_min", 80))
        log.append(dict(stage="joint (all coefficients, mean full-window E, full alignment)", start=e0, end=e3,
                        converged=conv, seconds=round(time.time() - t)))
        print(f"  [{name}] joint polish: mean E_full {e0:.4e} -> {e3:.4e} (converged={conv}, {time.time() - t:.0f}s)", flush=True)
    tm = ("form", spec["mr"], list(map(float, x)))
    res = {}
    grids = [("T96", T96), ("H270", H.H270), ("lowq", LOWQ)] + ([("T96_other_half", other)] if other else [])
    for grid, keys in grids:
        FF.KEYS = keys; FF.SEEDS.clear(); res[grid] = FF.final(pool, "split", ti, tm)
    out = dict(variant=name, spec=spec, ncoef=ncoef, mr_names=MR_FORMS[spec["mr"]][0], x_mr=list(map(float, x)),
               theta_insp=list(map(float, ti)), names_insp=NAMES, log=log, seconds=round(time.time() - t),
               stats={g: dict(full=stats(r), insp_E_insp=stats(r, "insp_E_insp"), insp_E_MR=stats(r, "insp_E_MR"))
                      for g, r in res.items()},
               lowq_median={str(q): float(np.median([r["full_E_full"] for r in res["lowq"] if r["q"] == q])) for q in (2.0, 2.5)},
               gateP_failures=[(r["q"], r["chi1"]) for g in res for r in res[g] if r["beta_min"] <= 0 or r["alpha_min"] <= 0],
               cases=res)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1))
    s = out["stats"]
    print(f"RESULT {name:14s} {ncoef:2d}c | T96 {s['T96']['full']['median']:.3e}/{s['T96']['full']['mean']:.3e}/{s['T96']['full']['max']:.2e}"
          f" | H270 {s['H270']['full']['median']:.3e}/{s['H270']['full']['mean']:.3e}/{s['H270']['full']['max']:.2e}"
          f" | q2.5 {out['lowq_median']['2.5']:.2e} q2 {out['lowq_median']['2.0']:.2e} | H270 E_insp {s['H270']['insp_E_insp']['median']:.2e}"
          f" E_MR {s['H270']['insp_E_MR']['median']:.2e} | {out['seconds']}s", flush=True)
    return out


GATE_H270_MEDIAN = float(np.median([r["full_E_full"] for r in SLIM["H270"]]))
LOG = HERE / "EXPLORE_LOG.md"


def auto_round2(mode):
    """Pick the round-1 winner (lowest H270 mean full-window E, <= 14c, gate P clean) and queue its follow-ups."""
    R = {}
    for p in sorted(OUT.glob("*.json")):
        if "__" not in p.stem:
            d = json.loads(p.read_text()); R[p.stem] = d
    if "slim" not in R or abs(R["slim"]["stats"]["H270"]["full"]["median"] / GATE_H270_MEDIAN - 1) > 0.02:
        msg = f"AUTO round 2 ({mode}) ABORTED: gate variant 'slim' missing or off by > 2% from {GATE_H270_MEDIAN:.4e}"
        with LOG.open("a") as f:
            f.write("\n" + msg + "\n")
        sys.exit(msg)
    ok = {n: d for n, d in R.items() if d["ncoef"] <= 14 and not d["gateP_failures"]}
    w = min(ok, key=lambda n: ok[n]["stats"]["H270"]["full"]["mean"])
    spec = dict(VARIANTS[w])
    if mode == "windows":
        new = {f"{w}__win{s:+.1f}": dict(spec, window=s) for s in (-2.0, -0.5)}
    else:
        new = {f"{w}__joint": dict(spec, **{"from": w, "joint": True, "budget_min": 85})}
    VARIANTS.update(new)
    with LOG.open("a") as f:
        f.write(f"\nAUTO round 2 ({mode}, {time.strftime('%Y-%m-%d %H:%M')}): winner of {len(ok)} round-1 variants by H270 mean = "
                f"**{w}** (H270 {ok[w]['stats']['H270']['full']['median']:.3e} / {ok[w]['stats']['H270']['full']['mean']:.3e} / "
                f"{ok[w]['stats']['H270']['full']['max']:.2e}); running {list(new)}\n")
    return list(new)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--auto", choices=["windows", "joint"], default=None)
    a = ap.parse_args()
    names = list(a.variant) + (auto_round2(a.auto) if a.auto else [])
    for n in names:
        assert n in VARIANTS, n
    for k in T96 + H.H270 + LOWQ:
        M.CASES[k] = M.prepare(*k)
        rem(*k)                                            # before the fork: workers copy _REM
    for n in names:
        set_window(VARIANTS[n].get("window", -1.0))      # before the fork: workers copy M.CASES at fork time
        with get_context("fork").Pool(128) as pool:
            out = run_variant(pool, n)
        if a.auto:
            s = out["stats"]
            with LOG.open("a") as f:
                f.write(f"* {n} ({out['ncoef']}c): T96 {s['T96']['full']['median']:.3e} / {s['T96']['full']['mean']:.3e} / "
                        f"{s['T96']['full']['max']:.2e};  H270 {s['H270']['full']['median']:.3e} / {s['H270']['full']['mean']:.3e} / "
                        f"{s['H270']['full']['max']:.2e};  q2.5 {out['lowq_median']['2.5']:.2e}, q2 {out['lowq_median']['2.0']:.2e}\n")


if __name__ == "__main__":
    main()
