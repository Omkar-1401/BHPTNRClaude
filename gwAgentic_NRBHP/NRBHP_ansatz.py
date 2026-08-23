# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (UT Codex)
#     language: python
#     name: ut_codex
# ---

# %%
import gwsurrogate

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
# add the path to the script directory
sys.path.append("/home/omkarnm1401/UT_Austin/BHPTNRSurrogate/surrogates")
import BHPTNRSur1dq1e4 as bhptsur

# %%
tbhpt, hbhpt = bhptsur.generate_surrogate(q=5, calibrated=False)

# %%
nrsur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')


# %%
# convience wrapper for calling the NRHybSur3dq8 model
def generate_nrsur(q_input, f_low=None, t_start=None):
    chiA = [0, 0, 0.0]
    chiB = [0, 0, 0.0]
    dt = 0.1        # step size, Units of M
    if f_low==None:
        f_low=5e-3
    if t_start==None:
        t_start=-5000.1
    # dyn stands for dynamics and is always None for this model
    t, h, dyn = nrsur(q_input, chiA, chiB, dt=dt, f_low=f_low) 
    # we only take the last 5000M long waveform to calibrate
    indx = np.where(t>=t_start)
    t = t[indx]
    for mode in h.keys():
        h[mode] = h[mode][indx]
    return t,h


# %%
tnr, h_NRHybSur3dq8 = generate_nrsur(q_input=5)

# %%
q_input = 5
mass_norm = 1/(1+1/q_input)

# %%
h_NR_22 = h_NRHybSur3dq8[(2, 2)]

# %%
import gwtools


# %%
def mathcalE_error(h1, h2):
    """
    Compute time-domain waveform error.

    Calculates the normalized L2 error between two waveforms using the metric
    defined in Eq. 21 of https://arxiv.org/pdf/1701.00550.pdf

    Parameters:
    -----------
    h1 : array (complex)
        Reference waveform (used for normalization)
    h2 : array (complex)
        Comparison waveform (same length as h1)

    Returns:
    --------
    normed_errs : float
        Normalized error: (||h1||² + ||h2||² - 2*Re(<h1,h2>)) / (2*||h1||²)
        Value of 0 indicates perfect agreement, larger values indicate greater disagreement

    Notes:
    ------
    - Assumes h1 is the reference waveform and normalizes by its magnitude
    - Both waveforms should have the same time sampling
    - The metric is symmetric in the numerator but asymmetric due to normalization by ||h1||²
    """
    # Compute norms squared
    n1Sqr = np.sum(np.abs(h1)**2)
    n2Sqr = np.sum(np.abs(h2)**2)

    # Compute inner product
    dots = np.array([h1[i] * (h2[i].conjugate()) for i in range(len(h1))])
    sdot = np.real(np.sum(dots))

    # Normalize error by reference waveform magnitude
    normed_errs = ((n1Sqr + n2Sqr) - 2 * sdot) / (2 * n1Sqr)

    return normed_errs


# %%
mask = ((tbhpt  * mass_norm) > -5000)

# %%
plt.plot(tnr, np.real(h_NR_22), label = 'NR 22 mode')
plt.plot(tbhpt  * mass_norm, np.real(hbhpt[(2,2)])  * mass_norm, label = 'BHPT 22')

plt.xlim(-5000, 100)
plt.legend()

# %%
h_nr_33 = h_NRHybSur3dq8[(3, 3)]

plt.plot(tnr, np.real(h_nr_33), label = 'NR 33 mode')
plt.plot(tbhpt  * mass_norm, np.real(hbhpt[(3,3)])  * mass_norm, label = 'BHPT 33')

plt.xlim(-1000, 100)
plt.legend()

# %%
h_nr_44 = h_NRHybSur3dq8[(4, 4)]

plt.plot(tnr, np.real(h_nr_44), label = 'NR 44 mode')
plt.plot(tbhpt  * mass_norm, np.real(hbhpt[(4, 4)])  * mass_norm, label = 'BHPT 44')

plt.xlim(-1000, 100)
plt.legend()

# %%
h_nr_55 = h_NRHybSur3dq8[(5, 5)]

plt.plot(tnr, np.real(h_nr_55), label = 'NR 55 mode')
plt.plot(tbhpt  * mass_norm, np.real(hbhpt[(5, 5)])  * mass_norm, label = 'BHPT 55')

plt.xlim(-400, 100)
plt.legend()

# %%
h_bhpt_interp = gwtools.interpolate_h(tbhpt * mass_norm, hbhpt[(2,2)], tnr)
mathcalE_error(h_NR_22, h_bhpt_interp)


# %%
def alpha_ansatz(q, l):
    return 1 + A_alpha(l) / q + B_alpha(l) / q ** 2 + C_alpha(l) / q ** 3 + D_alpha(l) / q ** 4


# %%
def beta_ansatz(q):
    return 1 + A_beta / q + B_beta / q ** 2 + C_beta / q ** 3 + D_beta / q ** 4

# %% [markdown]
# plt.plot(tnr, np.real(h_NR_22), label = 'NR 22 mode')
# plt.plot(tbhpt  * (1 - 1.238/q_input + 1.596 / q_input ** 2 - 1.776 / q_input ** 3 + 1.0577 / q_input ** 4), np.real(hbhpt[(2,2)])  * (1 - 1.330/q_input + 2.72 / q_input ** 2 - 5.904 / q_input ** 3 + 5.548 / q_input ** 4), label = 'BHPT 22')
#
# plt.xlim(-5000, 100)
# plt.legend()
