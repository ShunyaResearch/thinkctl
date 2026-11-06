"""Ensemble task-related component analysis with a filter bank (Nakanishi et al. 2018).

For each target k and sub-band m, TRCA finds a spatial filter w that makes the calibration
trials of that target as similar to each other as possible:
    maximise  w^T S w / w^T Q w,
    S = sum_{i != j} X_i X_j^T   (covariance BETWEEN different trials)
    Q = sum_i X_i X_i^T          (covariance WITHIN trials)
The "ensemble" filter W stacks the filters of all targets (channels x K).
Score of a test trial x for target k:
    r_{m,k} = corr( vec(W_m^T x_m), vec(W_m^T template_{m,k}) ),   template = mean of calibration trials
    score_k = sum_m a_m * sign(r_{m,k}) * r_{m,k}^2
"""
import numpy as np
from scipy.linalg import eigh

from .filters import filterbank, subband_lows
from .fbcca import fb_weights


def trca_filter(xk: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """xk: (n_trials, c, n) of one target. Returns the top TRCA spatial filter (c,)."""
    xk = xk - xk.mean(-1, keepdims=True)
    q = np.einsum("tcn,tdn->cd", xk, xk)
    s_sum = xk.sum(0)
    s = s_sum @ s_sum.T - q
    q = q + reg * np.trace(q) / q.shape[0] * np.eye(q.shape[0])
    _, vecs = eigh(s, q)
    return vecs[:, -1]


def corr_flat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between flattened a (T, ...) and b (K, ...) -> (T, K)."""
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return a @ b.T


class TRCA:
    """Calibrated decoder. Works on pre-filtered sub-band data (B, T, c, n) via fit_fb/scores_fb,
    or on raw windows via fit/scores (filters internally)."""

    def __init__(self, fs, n_bands=5, scheme="8m", hi=88.0, a=1.25, b=0.25, f_min=8.0,
                 ensemble=True, score="signed_sq", reg=1e-6, filt_kw=None):
        self.fs = float(fs)
        self.lows = subband_lows(scheme, n_bands, f_min)
        self.hi, self.ensemble, self.score, self.reg = hi, ensemble, score, reg
        self.w = fb_weights(n_bands, a, b)
        self.filt_kw = filt_kw or {}

    # --- pre-filtered interface (fast path for offline experiments) ---
    def fit_fb(self, xfb: np.ndarray, y: np.ndarray):
        """xfb: (B, T, c, n), y: (T,) target indices 0..K-1 (all K must be present)."""
        self.classes_ = np.unique(y)
        nb = xfb.shape[0]
        self.templates_ = np.stack([np.stack([xfb[m, y == k].mean(0) for k in self.classes_]) for m in range(nb)])
        self.filters_ = np.stack([
            np.stack([trca_filter(xfb[m, y == k], self.reg) for k in self.classes_], axis=1) for m in range(nb)
        ])  # (B, c, K)
        return self

    def rho_fb(self, xfb: np.ndarray) -> np.ndarray:
        """Correlations r (T, K, B)."""
        out = []
        for m in range(xfb.shape[0]):
            wm = self.filters_[m]  # (c, K)
            if self.ensemble:
                px = np.einsum("ck,tcn->tkn", wm, xfb[m])
                pt = np.einsum("ck,jcn->jkn", wm, self.templates_[m])
                out.append(corr_flat(px, pt))
            else:  # each target uses its own filter
                px = np.einsum("ck,tcn->tkn", wm, xfb[m])
                pt = np.einsum("ck,kcn->kn", wm, self.templates_[m])
                r = np.stack([corr_flat(px[:, k], pt[k:k + 1])[:, 0] for k in range(wm.shape[1])], 1)
                out.append(r)
        return np.stack(out, -1)

    def scores_fb(self, xfb: np.ndarray) -> np.ndarray:
        r = self.rho_fb(xfb)[..., : len(self.w)]
        if self.score == "signed_sq":
            return (np.sign(r) * r ** 2) @ self.w
        return r @ self.w  # plain weighted r (Nakanishi's MATLAB code)

    # --- raw-window interface (used by the streaming decoder) ---
    def fit(self, x: np.ndarray, y: np.ndarray):
        return self.fit_fb(filterbank(np.asarray(x, float), self.fs, self.lows, self.hi, **self.filt_kw), y)

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        single = x.ndim == 2
        if single:
            x = x[None]
        s = self.scores_fb(filterbank(x, self.fs, self.lows, self.hi, **self.filt_kw))
        return s[0] if single else s


def ensemble_scores(x, filters, templates, weights, signed_sq=True):
    """Ensemble-TRCA scores. x (B, T, c, n) time-centred sub-band test windows; filters (B, c, K);
    templates (B, K, c, n). Returns (T, K)."""
    out = 0
    for m in range(min(len(weights), x.shape[0])):
        px = np.einsum("ck,tcn->tkn", filters[m], x[m])
        pt = np.einsum("ck,jcn->jkn", filters[m], templates[m])
        r = corr_flat(px, pt)
        out = out + weights[m] * (np.sign(r) * r ** 2 if signed_sq else r)
    return out


class TRCAStats:
    """Precomputed per-block sufficient statistics for fast leave-block-out ensemble TRCA.

    For every block b, sub-band m and target k: the sum of (time-centred) trials and the sum of
    X X^T. Training on any set of blocks, and on any channel subset, is then just sums + indexing.
    """

    def __init__(self, xfb, y, block):
        self.x = xfb - xfb.mean(-1, keepdims=True)  # (B, T, C, n)
        self.y, self.block = np.asarray(y), np.asarray(block)
        self.classes = np.unique(self.y)
        self.blocks = np.unique(self.block)
        B, T, C, n = self.x.shape
        K, nb = len(self.classes), len(self.blocks)
        self.sums = np.zeros((nb, B, K, C, n))
        self.qs = np.zeros((nb, B, K, C, C))
        self.cnt = np.zeros((nb, K))
        for bi, b in enumerate(self.blocks):
            for ki, k in enumerate(self.classes):
                sel = (self.block == b) & (self.y == k)
                xs = self.x[:, sel]  # (B, t, C, n)
                self.sums[bi, :, ki] = xs.sum(1)
                self.qs[bi, :, ki] = np.einsum("btcn,btdn->bcd", xs, xs)
                self.cnt[bi, ki] = sel.sum()

    def fit(self, train_blocks, ch=None, reg=1e-6):
        """Returns filters (B, c, K) and templates (B, K, c, n) from the given blocks (indices into self.blocks)."""
        tb = np.asarray(train_blocks)
        ssum = self.sums[tb].sum(0)  # (B, K, C, n)
        q = self.qs[tb].sum(0)  # (B, K, C, C)
        cnt = self.cnt[tb].sum(0)  # (K,)
        if ch is not None:
            ch = np.asarray(ch)
            ssum = ssum[:, :, ch]
            q = q[:, :, ch][:, :, :, ch]
        s = np.einsum("bkcn,bkdn->bkcd", ssum, ssum) - q
        c = q.shape[-1]
        tr = np.trace(q, axis1=-2, axis2=-1)[..., None, None]
        q = q + reg * tr / c * np.eye(c)
        # generalized symmetric eigenproblem S w = l Q w via Cholesky whitening, batched
        L = np.linalg.cholesky(q)
        Li = np.linalg.inv(L)
        a = Li @ s @ np.swapaxes(Li, -1, -2)
        a = (a + np.swapaxes(a, -1, -2)) / 2
        _, v = np.linalg.eigh(a)
        w = np.swapaxes(Li, -1, -2) @ v[..., -1:]  # (B, K, c, 1)
        filters = np.swapaxes(w[..., 0], 1, 2)  # (B, c, K)
        templates = ssum / cnt[None, :, None, None]
        return filters, templates

    def scores(self, test_mask, filters, templates, weights, ch=None, signed_sq=True, cls=None):
        """cls: optional list of class indices -> ensemble uses only those targets' filters/templates."""
        x = self.x[:, test_mask]
        if ch is not None:
            x = x[:, :, np.asarray(ch)]
        if cls is not None:
            filters, templates = filters[:, :, cls], templates[:, cls]
        return ensemble_scores(x, filters, templates, weights, signed_sq)

    def lobo(self, weights, ch=None, n_train=None, rng=None, draws=1):
        """Leave-one-block-out scores (T, K). With n_train, train on n_train random other blocks
        (averaging over `draws` random choices is done by the caller)."""
        sc = np.full((len(self.y), len(self.classes)), np.nan)
        for bi, b in enumerate(self.blocks):
            others = [i for i in range(len(self.blocks)) if i != bi]
            tb = others if n_train is None else list(rng.choice(others, n_train, replace=False))
            f, t = self.fit(tb, ch)
            te = self.block == b
            sc[te] = self.scores(te, f, t, weights, ch)
        return sc
