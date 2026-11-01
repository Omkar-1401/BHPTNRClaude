"""One place to load a leader-candidate model into the exploration chain (for the generic plot / diagnostic scripts).

    load("p40")      pass 40: pass 39 L4 coefficients + closed-form R2 (O2)                         18c
    load("p40lean")  pass 40 lean: pass 42 K1 = pass 40 with the Ori–Thorne onset, T_start = -1     16c
    load("p49b12")   pass 49b lean-12: pass 40 lean with pb2, eb1, pb1, ea0 pruned                   12c
    load("p49b9")    pass 49c lean-9: lean-12 with pa0, r3, r1 pruned as well                          9c
Returns dict(P6, TH, R, label, out, tag, t100).  Call before any fork; it installs P6.get / P6.ab.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    argv0 = sys.argv[0]
    if name == "p40":
        sys.argv = [argv0, "L4"]
        import p40_r2closed as C
        P6 = C.P6; P6.get = C.L.get; P6.ab = C.L.ab; C.use("O2n")
        R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
        R = dict(R, T96_cases=json.loads((HERE / "p40" / "p40.json").read_text())["O2n"]["T96_cases"])
        info = dict(label="pass 40 (18 coef)", out=HERE / "plots_p40", tag="p40", t100="p40")
    elif name == "p40lean":
        sys.argv = [argv0, "K1"]
        import p42_otonset as K
        P6 = K.P6; P6.get = K.L.get; P6.ab = K.L.ab; K.C.use("O2n"); K.M.S1FUN = K.onset
        R = json.loads((HERE / "p42" / "p42_K1.json").read_text())
        info = dict(label="pass 40 lean (16 coef, OT onset)", out=HERE / "plots_p40lean", tag="p40lean", t100="k1")
    elif name in ("p48", "p48lean"):
        sys.argv = [argv0, "K1"]
        import p42_otonset as K                  # widened onset guard + OT onset hook (used only by p48lean)
        sys.argv = [argv0, "40", "v2"]
        import p47b_shapescf as B47              # pass-46 closed-form inputs + shape installer
        B = B47.B; P6 = B.P6
        B47.SH = json.loads((HERE / "shapes_closedform_v2.json").read_text())
        P6.get = B.get; P6.ab = B.L.ab; B.C.use("O2n")
        B.CF.install(B.S, B.PC, qnm=True, remnant=True); B._USE_G["v"] = True; B47.install_shapes(True, True)
        sc = json.loads((HERE / "p48" / "p48.json").read_text())
        if name == "p48":
            R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
            R = dict(R, T96_cases=sc["m48"]["T96_cases"])
            info = dict(label="pass 48 (18 coef, closed form)", out=HERE / "plots_p48", tag="p48", t100="m48")
        else:
            K.M.S1FUN = K.onset
            R = json.loads((HERE / "p42" / "p42_K1.json").read_text())
            R = dict(R, T96_cases=sc["lean48"]["T96_cases"])
            info = dict(label="pass 48 lean (16 coef, closed form, OT onset)", out=HERE / "plots_p48lean", tag="p48lean", t100="lean48")
    elif name in ("p49b12", "p49b9"):
        sys.argv = [argv0, "K1"]
        import p42_otonset as K                  # pass 40 lean chain (qnm, NRSur3dq8Remnant, splines) + OT onset
        P6 = K.P6; P6.get = K.L.get; P6.ab = K.L.ab; K.C.use("O2n"); K.M.S1FUN = K.onset
        n = int(name[4:])
        R = [s for s in json.loads((HERE / "p49b" / "p49.json").read_text())["steps"] if s["n"] == n][0]
        info = dict(label=f"pass {'49b' if n == 12 else '49c'} lean-{n} ({n} coef)", out=HERE / f"plots_{name}", tag=name,
                    t100=f"p49b_n{n}")
    else:
        raise ValueError(name)
    sys.argv = [argv0, name]
    info["out"].mkdir(exist_ok=True)
    return dict(P6=P6, TH=list(map(float, R["theta"])), R=R, **info)
