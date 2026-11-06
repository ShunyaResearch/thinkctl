"""Frozen defaults used from E2 onwards (decided in E1).

FBCCA / TRCA filter bank: Chebyshev-I, order 4, 0.5 dB ripple, zero-phase (sosfiltfilt) applied to the
stare window only. Sub-band m covers [8m - 2, 88] Hz -> lower edges 6, 14, 22, 30, 38 Hz.
Weights w_m = m^-1.25 + 0.25 -> 1.25, 0.67, 0.50, 0.43, 0.38. References: 5 harmonics (sin+cos).
Mains notch: from E5 on, a 50 Hz notch (Q = 30, zero-phase on the epoch) is part of the default
pipeline for EVERY dataset (it used to be D4 only). scripts/notch_check.py measures what it changes.
"""
import numpy as np

LATENCY = 0.14  # s, visual latency skipped after stimulus onset
GAZE = 0.5  # s, gaze-shift time added to every selection for ITR
HI = 88.0
FB = dict(scheme="8m-2", n_bands=5, a=1.25, b=0.25)
N_HARM = 5
NOTCH_HZ = 50.0
NOTCH = {"d1": NOTCH_HZ, "d2": NOTCH_HZ, "d3": NOTCH_HZ, "d4": NOTCH_HZ}
NOTCH_BEFORE_E5 = {"d1": None, "d2": None, "d3": None, "d4": 50.0}  # what E1-E4 used


def subject_halves(subjects, seed=0):
    """The fixed tune/test split used in every experiment (identical to E1)."""
    perm = np.random.default_rng(seed).permutation(len(subjects))
    h0 = sorted(subjects[i] for i in perm[: len(perm) // 2])
    h1 = sorted(subjects[i] for i in perm[len(perm) // 2:])
    return h0, h1
