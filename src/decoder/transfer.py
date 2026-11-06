"""Template-matching scorer that works with 0, 1 or more calibration trials (simple, linear).

For a test window X (per sub-band m) and a template T_k (user's own average, other subjects'
average, or a blend of both), in the spirit of tt-CCA / extended CCA:
  rho1 = CCA(X, sine refs of target k)                       (= FBCCA term, needs no data)
  rho2 = CCA(X, T_k)                                         (best match between X and the template)
  rho3 = corr(X^T w, T_k^T w), w = CCA spatial filter of X vs refs  (template seen through X's filter)
  score_k = sum_m a_m * [rho1^2 + rho2^2 + sign(rho3) rho3^2]
"""
import numpy as np

from .cca import orth_basis


def _canon(qx, qy):
    """Largest singular value and left singular vector of Qx^T Qy. qx (T,n,c), qy (K,n,p)."""
    m = np.matmul(np.swapaxes(qx, -1, -2)[:, None], qy[None])  # (T,K,c,p)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    return s[..., 0], u[..., 0]  # (T,K), (T,K,c)


def template_scores(xfb, templates, qy, weights):
    """xfb (B,T,c,n) test windows (filtered); templates (B,K,c,n); qy (K,n,p) orth sine refs."""
    total = 0.0
    for m in range(min(len(weights), xfb.shape[0])):
        x = xfb[m] - xfb[m].mean(-1, keepdims=True)
        q, r = np.linalg.qr(np.swapaxes(x, -1, -2))  # x^T = q r
        rho1, u = _canon(q, qy)
        tm = templates[m] - templates[m].mean(-1, keepdims=True)
        rho2, _ = _canon(q, orth_basis(tm))
        # spatial filter in channel space: x^T w = q u  ->  w = r^-1 u
        wv = np.linalg.solve(r[:, None], u[..., None])[..., 0]  # (T,K,c)
        px = np.einsum("tnc,tkc->tkn", q, u)
        pt = np.einsum("kcn,tkc->tkn", tm, wv)
        px = px - px.mean(-1, keepdims=True)
        pt = pt - pt.mean(-1, keepdims=True)
        rho3 = (px * pt).sum(-1) / (np.linalg.norm(px, axis=-1) * np.linalg.norm(pt, axis=-1) + 1e-12)
        total = total + weights[m] * (rho1 ** 2 + rho2 ** 2 + np.sign(rho3) * rho3 ** 2)
    return total
