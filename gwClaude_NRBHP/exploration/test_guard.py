"""Guard for the SEALED test set (t100_set.py, 2026-10-13).  Importing this module wraps the two waveform-case builders
(fit_pn_anchored_spin.build_case, fit_fluxanchored_spin.build_case).  Every exploration fit, prune, validation grid or
diagnostic loads its cases through them, and any (q, chi1) within TOL of a sealed test point is refused.

Distance is measured in the unit box (q / 5, chi1 / 1.4).  TOL = 1e-3, i.e. |dq| ~ 0.005 and |dchi1| ~ 0.0014: it catches
a copied or re-rounded test point.  It does not catch an honest random draw that lands nearby (T96 / H270 sit >= 2.9e-3
away).  The error message never prints the test point.
Only t100_set.py may open the gate (ALLOW = True); it is used to build and score the test set itself.
Imported by p3_rd.py (and so every pass from 3 on) and by the early passes that build cases directly.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import fit_pn_anchored_spin as _S              # noqa: E402
import fit_fluxanchored_spin as _FS            # noqa: E402

TOL = 1e-3
POINTS = HERE / "t100" / "TEST_points.txt"
ALLOW = False
_P = None


class SealedTestPoint(RuntimeError):
    pass


def _points():
    global _P
    if _P is None:
        if not POINTS.exists():
            raise SealedTestPoint(f"{POINTS} is missing: the sealed-set guard cannot run (restore it, do not bypass)")
        _P = np.loadtxt(POINTS, comments="#", ndmin=2)
    return _P


def check(q, chi1):
    if ALLOW:
        return
    P = _points()
    d = np.hypot((P[:, 0] - float(q)) / 5.0, (P[:, 1] - float(chi1)) / 1.4).min()
    if d < TOL:
        raise SealedTestPoint(f"(q={float(q):.4f}, chi1={float(chi1):+.3f}) is within {TOL:g} (unit box) of a SEALED "
                              "test point; it may not be used for fitting, validation or selection")


def _wrap(mod):
    f = mod.build_case
    if getattr(f, "_sealed_guard", False):
        return

    def build_case(q, chi1, *a, **k):
        check(q, chi1)
        return f(q, chi1, *a, **k)
    build_case._sealed_guard = True
    mod.build_case = build_case


_wrap(_S)
_wrap(_FS)
