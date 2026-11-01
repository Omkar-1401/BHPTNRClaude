"""EXPLORATION pass 46 -- closed-form inputs in the pass-40 leader (user, 2026-10-21: closed forms wherever possible).

Replacements (exploration/closedform.py; constants fitted offline to theory / reference data, not to the calibration NR):
    Q   Kerr (2,2,0) QNM frequency and decay from a 6-term series in sqrt(1 - chi)            (was: qnm package)
    R   remnant M_f, chi_f from 16-term polynomial closed forms                                (was: NRSur3dq8Remnant)
    G   measured ppBHPT ringdown decay from an 8-constant closed form in (chi1, nu)            (was: per-waveform slope fit)
Variants on pass 40 (L4 coefficients + O2 R2), FROZEN unless marked:
    Cq   Q only        Cqr   Q + R        Call   Q + R + G        Call_r   Q + R + G, 18 coefficients refit on T96
Usage: python p46_closedform.py [budget].  Writes exploration/p46/p46.json.
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BUD = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
sys.argv = [sys.argv[0], "L4"]
import p40_r2closed as C                        # noqa: E402
import closedform as CF                         # noqa: E402
L = C.L; P6 = C.P6; A = C.A; PR = C.PR; D = L.D
S = L.G.M.Z.S; PC = D.PC
OUT = HERE / "p46"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p39" / "p39_L4.json").read_text()); TH = np.array(R["theta"])
REF = json.loads((HERE / "p40" / "p40.json").read_text())["O2n"]
_USE_G = {"v": False}


def get(k):
    """pass-39 get with the measured decay optionally replaced by its closed form."""
    c = D._get0(k)
    if "g_meas" not in c:
        if _USE_G["v"]:
            g = CF.gamma_meas(k[0], k[1])
        else:
            t, h = c["case"]["t_bhpt"], c["case"]["h_bhpt"]; s = t - c["tpk"]
            m = (s >= P6.S_RING) & (s <= D.SE); g = -float(np.polyfit(s[m], np.log(np.abs(h[m])), 1)[0])
        gpp = PC.w220_exact(k[1]) / (2 * PC.Q220_exact(k[1]))
        br, kap = c["anc"]; nu = k[0] / (1 + k[0]) ** 2
        c["g_meas"], c["g_pp"] = g, gpp
        c["anc"] = (br, kap + L.gfac(nu) * (g - gpp) / br)
    return c


def score(tag, th, out):
    pool = mp.get_context("fork").Pool(96)
    sc = PR.score(pool, th); pool.close()
    out[tag] = sc; t, h = sc["T96"], sc["H270"]; lq = [v for k, v in sc["lowq"].items() if k.startswith("2.5")]
    print(f"{tag:7s}: T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 median {h[1]:.4e} | q2.5 max {max(lq):.3e}"
          f"  (pass 40: {REF['T96'][0]:.4e} / {REF['T96'][1]:.4e} / {REF['T96'][2]:.4e})", flush=True)
    (OUT / "p46.json").write_text(json.dumps(out, indent=1))
    return sc


def main():
    P6.get = get; P6.ab = L.ab; C.use("O2n")
    out = {}
    CF.install(S, PC, qnm=True, remnant=False); score("Cq", TH, out)
    CF.install(S, PC, qnm=True, remnant=True); score("Cqr", TH, out)
    _USE_G["v"] = True; score("Call", TH, out)
    pool = mp.get_context("fork").Pool(96)
    th, ef = A.refit(pool, TH, list(R["free"]), BUD, "Call_r"); pool.close()
    out["Call_r_theta"] = list(map(float, th)); score("Call_r", th, out)


if __name__ == "__main__":
    main()
