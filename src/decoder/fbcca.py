"""Filter-bank CCA (Chen et al. 2015). No calibration needed.

score_k = sum_m w_m * rho_{m,k}^2,   w_m = m^(-a) + b
  rho_{m,k}: CCA correlation between sub-band m of the EEG and the references of target k.
"""
import numpy as np

from .cca import cca_corr, orth_basis, ref_signals
from .filters import filterbank, subband_lows


def fb_weights(n_bands: int, a: float = 1.25, b: float = 0.25) -> np.ndarray:
    m = np.arange(1, n_bands + 1, dtype=float)
    return m ** (-a) + b


def fbcca_rho(xfb: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """xfb: (B, T, c, n) filtered sub-bands; qy: (K, n, p). Returns rho (T, K, B)."""
    return np.stack([cca_corr(orth_basis(xb), qy) for xb in xfb], axis=-1)


def combine(rho: np.ndarray, w: np.ndarray) -> np.ndarray:
    """rho (..., B) -> weighted score (...)."""
    return (rho[..., : len(w)] ** 2) @ w


class FBCCA:
    """Calibration-free decoder. x: (T, channels, n) raw (unfiltered) window -> (T, K) scores."""

    def __init__(self, freqs, fs, n_bands=5, scheme="8m", hi=88.0, n_harm=5, a=1.25, b=0.25,
                 f_min=None, filt_kw=None, phases=None):
        self.freqs = np.asarray(freqs, float)
        self.fs = float(fs)
        self.lows = subband_lows(scheme, n_bands, f_min if f_min is not None else self.freqs.min())
        self.hi, self.n_harm = hi, n_harm
        self.w = fb_weights(n_bands, a, b)
        self.filt_kw = filt_kw or {}
        self._qy = {}

    def qy(self, n):
        if n not in self._qy:
            self._qy[n] = orth_basis(ref_signals(self.freqs, n, self.fs, self.n_harm))
        return self._qy[n]

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        single = x.ndim == 2
        if single:
            x = x[None]
        xfb = filterbank(x, self.fs, self.lows, self.hi, **self.filt_kw)
        s = combine(fbcca_rho(xfb, self.qy(x.shape[-1])), self.w)
        return s[0] if single else s
