"""Standard CCA for SSVEP (Lin et al. 2006), vectorised over trials and targets.

The largest canonical correlation between EEG X (channels x samples) and a reference Y
(sin/cos at the target frequency and its harmonics) equals the largest singular value of
Qx^T Qy, where Qx, Qy are orthonormal bases of the (centred) column spaces.
"""
import numpy as np


def ref_signals(freqs, n: int, fs: float, n_harm: int, phases=None) -> np.ndarray:
    """Sine/cosine references, shape (K targets, 2*n_harm, n samples)."""
    t = np.arange(n) / fs
    freqs = np.asarray(freqs, float)
    ph = np.zeros_like(freqs) if phases is None else np.asarray(phases, float)
    out = np.empty((len(freqs), 2 * n_harm, n))
    for h in range(1, n_harm + 1):
        arg = 2 * np.pi * h * freqs[:, None] * t[None, :] + h * ph[:, None]
        out[:, 2 * h - 2] = np.sin(arg)
        out[:, 2 * h - 1] = np.cos(arg)
    return out


def orth_basis(a: np.ndarray) -> np.ndarray:
    """a: (..., d, n) signals (rows=variables). Returns Q: (..., n, d), orthonormal basis of centred rows."""
    a = a - a.mean(-1, keepdims=True)
    q, _ = np.linalg.qr(np.swapaxes(a, -1, -2))
    return q


def cca_corr(qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """qx: (T, n, c), qy: (K, n, p) -> (T, K) max canonical correlation."""
    m = np.matmul(np.swapaxes(qx, -1, -2)[:, None], qy[None])  # (T, K, c, p)
    return np.linalg.svd(m, compute_uv=False)[..., 0]


def cca_scores(x: np.ndarray, freqs, fs: float, n_harm: int = 5, qy=None) -> np.ndarray:
    """x: (T, channels, n) already band-passed. Returns (T, K) canonical correlations."""
    if qy is None:
        qy = orth_basis(ref_signals(freqs, x.shape[-1], fs, n_harm))
    return cca_corr(orth_basis(x), qy)
