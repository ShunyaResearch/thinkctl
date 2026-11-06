"""Band-pass filters and filter banks (Chebyshev type I, zero-phase)."""
from functools import lru_cache

import numpy as np
from scipy.signal import cheb1ord, cheby1, sosfiltfilt


@lru_cache(maxsize=None)
def bandpass_sos(fs: float, lo: float, hi: float, order: int = 4, rp: float = 0.5, design: str = "fixed"):
    """Second-order-sections Chebyshev-I band-pass.

    design="fixed": given order (order 4 = what the earlier work used).
    design="chen":  order chosen by cheb1ord as in Chen et al. 2015 FBCCA code
                    (passband [lo, hi], stopband [lo-2, hi+10], 3 dB ripple, 40 dB stop).
    """
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    if design == "fixed":
        return cheby1(order, rp, [lo / nyq, hi / nyq], btype="band", output="sos")
    if design == "chen":
        ws = [max(lo - 2.0, 0.5) / nyq, min(hi + 10.0, 0.98 * nyq) / nyq]
        n, wn = cheb1ord([lo / nyq, hi / nyq], ws, 3, 40)
        return cheby1(n, rp, wn, btype="band", output="sos")
    raise ValueError(design)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, **kw) -> np.ndarray:
    """Zero-phase band-pass along the last axis (time)."""
    return sosfiltfilt(bandpass_sos(float(fs), float(lo), float(hi), **kw), x, axis=-1)


@lru_cache(maxsize=None)
def notch_ba(fs: float, f0: float, q: float = 30.0):
    from scipy.signal import iirnotch
    return iirnotch(f0, q, fs)


def notch(x: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase mains notch (50 Hz Europe/Asia, 60 Hz Americas) along the last axis."""
    from scipy.signal import filtfilt
    b, a = notch_ba(float(fs), float(f0), float(q))
    return filtfilt(b, a, x, axis=-1)


def filterbank(x: np.ndarray, fs: float, lows, hi: float, **kw) -> np.ndarray:
    """Apply one band-pass per lower edge. Returns array of shape (n_bands, *x.shape)."""
    return np.stack([bandpass(x, fs, lo, hi, **kw) for lo in lows])


def subband_lows(scheme: str, n_bands: int, f_min: float = 8.0) -> list:
    """Lower edges of filter-bank sub-bands (all bands share the same upper edge).

    '8m'     : 8, 16, 24, ...        (Chen et al. 2015 'M1', designed for 8-15.8 Hz stimuli)
    '6m'     : 6, 12, 18, ...
    '8m-2'   : 6, 14, 22, ...        (Nakanishi's TRCA code for the 12-target set)
    'harm'   : m*f_min - 2           (each band starts just below the m-th harmonic of the lowest target)
    """
    m = np.arange(1, n_bands + 1, dtype=float)
    if scheme == "8m":
        lows = 8 * m
    elif scheme == "6m":
        lows = 6 * m
    elif scheme == "8m-2":
        lows = 8 * m - 2
    elif scheme == "harm":
        lows = m * f_min - 2
    else:
        raise ValueError(scheme)
    return [float(v) for v in lows]
