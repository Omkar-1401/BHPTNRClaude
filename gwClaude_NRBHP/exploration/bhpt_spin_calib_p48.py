"""Pass 48: the spinning ppBHPT (2,2) calibration map alpha(t), beta(t) with EVERY numerical input in closed form.

Stand-alone: numpy only (no qnm, no surfinBH, no spline tables).  = pass 40's 18 coefficients (pass 39 L4) with
    R2(x)        closed form O2 (pass 40)
    Kerr QNM     6-term series in sqrt(1 - chi) (pass 46)                    w, gamma within 9e-5 / 3e-4 of exact Kerr
    remnant      M_f, chi_f polynomial closed forms (pass 46)                within 5e-4 / 7e-4 of NRSur3dq8Remnant
    gamma_meas   8-constant closed form in (chi1, nu) (pass 46)              within 1.1% of the per-waveform slope
    shapes       phi_A, psi_A: tail + 4 sech bumps; phi_B, psi_B: tail + 6 sech^2 bumps fitted with their cumulative
                 integral (pass 47b; the beta shapes act on the integrated clock)
All closed-form constants were fitted OFFLINE to theory or reference data, never to the calibration NR.
Model equations: exploration/scaling_p40.md (unchanged except for the inputs above); this file: exploration/scaling_p48.md.
Inputs per case: q, chi1, the ppBHPT (2,2) waveform h on its clock t (m1), x_pp(t) and E(t) as in bhpt_spin_calib_p40.py.
Map: tau(t) = t0 + int_{-100}^t beta dt',  h_NR(tau(t)) = exp(i phi0) alpha(t) h_pp(t)  (t0, phi0 per-case alignment).
"""
from __future__ import annotations
import numpy as np

COEF = {"b": -0.023601062985639115, "b2": 0.21489598210843033, "P0": -4.200635268937013, "P1": 30.67659566400845, "s2": -18.472051931691894, "s3": 2.5528209096994967, "t1a": -102.5267796178166, "t1b": 79.12203396875586, "r1": 0.08169079240739548, "r3": -0.11160667650989235, "l0": -0.11459813280008839, "pa0": -0.09596927028980735, "pa2": -0.3677712409624297, "eb1": 0.09722212515619688, "pb0": 0.16771097569557578, "pb1": -0.27527723451556513, "pb2": 0.38567659442701024, "ea0": -0.06595793574625201}
NU5, NU_C, XCLIP, S_RING = 5.0 / 36.0, 0.04, 0.26, 15.0
R2_A, R2_B = 4.768, 21.68
QNM_W = [0.9918044418677526, -1.35690659709625, 1.3346138723679086, -0.8754764484513693, 0.33573278041178284, -0.05610005255804592]
QNM_G = [0.002781979933307037, 0.31998691348648595, -0.5121906019464096, 0.4425972079151339, -0.201742298941557, 0.03753727360563429]
REM_CHIF = [3.4036842113216665, -1.0091780066121987, -0.29763740715222786, -0.20976639715960252, -3.2953324411832092, 3.879431920307497, 1.5257745938838136, 1.8154458447655273, 3.1517020756446166, -22.223683428771487, -5.0069964188454605, -6.955436782503201, -1.9415161209335026, 62.427261768951205, 11.818634681063486, 11.658574674651755]     # c_ij, i (nu power) row-major over j (chi1 power), i, j = 0..3
REM_MF = [0.991711669750921, 1.3504842934998722, 1.971254259711671, -2.3992244153674056, 10.053478971021818, -23.850139394382172, -34.15642295932215, 48.753512728592476, -17.28829068883472, 169.7201574830331, 230.24725367529106, -264.8973499041564, 61.757782798973196, -347.0478195556696, -489.64177338145856, 440.7851415457486]
GMEAS = [-0.04275538559136163, -0.10589986940797971, 0.0827742503302796, 0.463087236310392, -0.3099255815693496, 0.0332619174352448, 0.5894340861388019, -0.06527583951800763]        # d0..d4, e0..e2
SHAPES = {"phiB": {"form": "sech2", "nb": 6, "params": [0.06044938782321446, 1173.77257339945, 83.1058773124736, 14.597813554695254, -12.629328656893616, 1.8346633996797415, -42.299830032179926, 56.99465135348513, -150.0055622015844, -16.88776369530587, 50.867739872048276, 149.0057597738564, -16.61698968341296, 50.56393652623447, -83.6112749642661, 14.572662314163765, 12.655104528463388, 23.56727216696091, 1189.1235531343452, 335.3800964710786], "clip": [-2000.0, 60.0]}, "phiA": {"form": "sech", "nb": 4, "params": [10.659098148219638, 268.66067589126624, -1.1313768433817535, 23.631613511617402, 23.710044507604533, -0.6739082822896687, 7.9808847592327625, 7.532690246763439, -0.4014377423775244, -53.42249982010452, 37.46662820713332, -13.404936557995066, 225.53625444021299, 256.4156037112303], "clip": [-2000.0, 8.0]}, "psiB": {"form": "sech2", "nb": 6, "params": [0.12310952971121905, 847.2374883906958, -1307.6050643676713, 1255.9624350289594, 508.821387902562, -0.23870246294297173, -82.81295445424138, 44.364646035461746, -1.2053108665267558, 2.3744174993348883, 25.03525807919985, 1.3898181406705894, -4.249227073570068, 14.819563052725497, 1.262732534719688, 51.91504306359985, 146.59345697767637, 406.4926999777428, 950.854713641159, 505.9513189432444], "clip": [-2000.0, 60.0]}, "psiA": {"form": "sech", "nb": 4, "params": [0.48097022762218095, 201.10482824514872, -2211.8924144863217, 0.06552893352630419, 12.593377360516813, -2.4118861431431933, -7.299403573597477, 34.212612087140556, 4424.775468662906, -0.30166696388196185, 12.59673830851738, -2212.1828641026837, -0.6667912028074487, 12.583163044041486], "clip": [-2000.0, 60.0]}}


def smooth5(u):
    u = np.clip(u, 0.0, 1.0); return u ** 3 * (10 - 15 * u + 6 * u * u)


def kappa_so(q):
    X1 = q / (1 + q); return 1.5 * (113 / 12 * X1 ** 1.2 - 19 / 6 * X1 ** 2 - 25 / 4 * X1)


def R2(x):
    x = np.clip(x, 0.005, 1.0 / 6.0); return 55.0 / 21.0 * x * (1 - 6 * x) * (1 + R2_A * x + R2_B * x * x)


def g_nu(nu):
    return 1.0 - np.exp(-(nu / NU_C) ** 2)


def w220(chi):
    s = np.sqrt(1 - chi); return sum(a * s ** k for k, a in enumerate(QNM_W))


def g220(chi):
    s = np.sqrt(1 - chi); return sum(a * s ** k for k, a in enumerate(QNM_G))


def isco_E(a):
    A = abs(a); z1 = 1 + (1 - A * A) ** (1 / 3) * ((1 + A) ** (1 / 3) + (1 - A) ** (1 / 3)); z2 = np.sqrt(3 * A * A + z1 * z1)
    r = 3 + z2 - np.sign(a) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2)); v = 1 / np.sqrt(r)
    return (1 - 2 * v ** 2 + a * v ** 3) / np.sqrt(1 - 3 * v ** 2 + 2 * a * v ** 3)


def _poly(c, nu, chi):
    return sum(c[4 * i + j] * nu ** i * chi ** j for i in range(4) for j in range(4))


def remnant(q, chi1):
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    return 1 - nu * (1 - isco_E(X1 ** 2 * chi1)) * _poly(REM_MF, nu, chi1), X1 ** 2 * chi1 + nu * _poly(REM_CHIF, nu, chi1)


def gamma_meas(q, chi1):
    nu = q / (1 + q) ** 2; d = GMEAS
    return g220(chi1) * (1 + d[0] + d[1] * chi1 + d[2] * chi1 ** 2 + d[3] * chi1 ** 3 + d[4] * chi1 ** 4
                         + nu * (d[5] + d[6] * chi1 + d[7] * chi1 ** 2))


def shape(name, s):
    d = SHAPES[name]; p = d["params"]; s = np.clip(np.asarray(s, float), *d["clip"])
    y = p[0] * np.where(s < 0, np.exp(np.minimum(s, 0) / abs(p[1])), 1.0)
    for k in range(d["nb"]):
        A, mu, w = p[2 + 3 * k: 5 + 3 * k]; z = np.clip((s - mu) / abs(w), -300, 300)
        y = y + A / (np.cosh(z) ** 2 if d["form"] == "sech2" else np.cosh(z))
    return y


def alpha_beta(q, chi1, t, h, x_pp, E, coef=None, tpk=None, t_full=None, h_full=None):
    """alpha(t), beta(t) on the grid t (m1); tpk from the full-resolution |h| peak.  (None, None) if the form is invalid."""
    c = COEF if coef is None else coef
    tf, hf = (t, h) if t_full is None else (t_full, h_full)
    tpk = float(tf[int(np.argmax(np.abs(hf)))]) if tpk is None else tpk
    nu, X1 = q / (1 + q) ** 2, q / (1 + q); w = nu / NU5; G = g_nu(nu); F = G / w
    mf, chif = remnant(q, chi1)
    br = w220(chi1) * mf / w220(chif); gK = g220(chi1); gNR = g220(chif) / mf
    kap = (gK + G * (gamma_meas(q, chi1) - gK)) / br - gNR
    s = np.asarray(t) - tpk
    xs = X1 ** -0.8 * np.minimum(x_pp, XCLIP)
    P = c["P0"] + c["P1"] * nu + F * (nu * c["s2"] * chi1 ** 2 + c["s3"] * chi1)
    bI = (X1 ** 1.2 * (1 + c["b"] * nu + c["b2"] * nu ** 2) + P * np.asarray(E)) * (1 + kappa_so(q) * chi1 * xs ** 1.5)
    if np.any(bI <= 0):
        return None, None
    laI = 1.2 * np.log(X1) + np.log1p(nu * R2(xs) / 2)
    s1 = c["t1a"] + c["t1b"] * chi1
    if not (-150 <= s1 <= 30 and s1 + 3 <= 0):
        return None, None
    W = smooth5((s - s1) / (0.0 - s1))
    lbM = np.log(br) + G * (c["r1"] * chi1 + c["r3"] * chi1 ** 2)
    laM = 1.2 * np.log(X1) + w * c["l0"] + kap * br * np.clip(s - S_RING, 0, None)
    lb = (1 - W) * np.log(bI) + W * lbM
    la = (1 - W) * laI + W * laM
    lb = lb + w * (c["pa0"] + c["pa2"] * chi1 ** 2) * shape("phiB", s) + G * c["eb1"] * chi1 * shape("psiB", s)
    la = la + G * (c["pb0"] + c["pb1"] * chi1 + c["pb2"] * chi1 ** 2) * shape("phiA", s) + w * c["ea0"] * shape("psiA", s)
    return np.exp(la), np.exp(lb)
