"""Shared offline-evaluation helpers (windowing, filtering, block-wise cross-validation)."""
import numpy as np

from .filters import bandpass, filterbank
from .trca import TRCA


def windowed_fb(data, lows, hi, windows, latency, mode="window", filt_kw=None):
    """Sub-band data per window length. Returns {W: (B, T, c, n)}.

    mode="window": cut the analysis window first, then filter it (what a real-time system sees;
                   standard in the reference FBCCA/TRCA code).
    mode="epoch":  filter the whole epoch, then cut (what the earlier work did; the filter can
                   use data after the window end, which is slightly optimistic).
    """
    filt_kw = filt_kw or {}
    out = {}
    if mode == "epoch":
        full = filterbank(data.X, data.fs, lows, hi, **filt_kw)
        a = data.onset + int(round(latency * data.fs))
        for w in windows:
            out[w] = full[..., a:a + int(round(w * data.fs))]
    elif mode == "window":
        for w in windows:
            out[w] = filterbank(data.window(w, latency), data.fs, lows, hi, **filt_kw)
    else:
        raise ValueError(mode)
    return out


def windowed_bp(data, lo, hi, windows, latency, mode="window", filt_kw=None):
    """Single band-pass version of windowed_fb. Returns {W: (T, c, n)}."""
    r = windowed_fb(data, [lo], hi, windows, latency, mode, filt_kw)
    return {w: v[0] for w, v in r.items()}


def trca_lobo_scores(xfb, y, block, trca_kw=None, train_blocks=None):
    """Leave-one-block-out ensemble-TRCA. xfb (B, T, c, n). Returns scores (T, K).

    train_blocks: optional callable(test_block, all_blocks) -> list of blocks to train on
                  (default: all other blocks).
    """
    trca_kw = trca_kw or {}
    blocks = np.unique(block)
    k = len(np.unique(y))
    scores = np.full((len(y), k), np.nan)
    for b in blocks:
        te = block == b
        tb = [x for x in blocks if x != b] if train_blocks is None else train_blocks(b, blocks)
        tr = np.isin(block, tb)
        m = TRCA(1.0, **trca_kw)  # fs unused on the pre-filtered path
        m.w = m.w[: xfb.shape[0]]
        m.fit_fb(xfb[:, tr], y[tr])
        scores[te] = m.scores_fb(xfb[:, te])
    return scores


def load_prepped(dkey, root, subject, notch_hz="default", **load_kw):
    """Load a subject and apply the mains notch to the raw epochs.
    notch_hz: "default" -> config.NOTCH[dkey]; None -> no notch; a number -> that frequency."""
    from .config import NOTCH
    from .datasets import get_loader
    from .filters import notch
    d = get_loader(dkey)[0](root, subject, **load_kw)
    f0 = NOTCH.get(dkey) if notch_hz == "default" else notch_hz
    if f0:
        d.X = notch(d.X, d.fs, f0)
    return d


def default_fb(data, windows, latency=None):
    """Frozen default filter bank (config.FB) on each stare window. Returns {W: (B, T, c, n)}."""
    from .config import FB, HI, LATENCY
    from .filters import subband_lows
    lows = subband_lows(FB["scheme"], FB["n_bands"], float(data.freqs.min()))
    return windowed_fb(data, lows, HI, windows, LATENCY if latency is None else latency, "window")


def fb_w():
    from .config import FB
    from .fbcca import fb_weights
    return fb_weights(FB["n_bands"], FB["a"], FB["b"])


def summarize_by(df, keys, value="acc", n_targets_col=None, gaze=0.5):
    """Per group: mean, std, % subjects >= 90%, worst 3, mean ITR (if window & N known)."""
    import pandas as pd
    from .metrics import itr_bits_per_min, subject_summary
    rows = []
    for k, g in df.groupby(keys):
        s = subject_summary(g[value].values, g["subject"].values)
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r.update(mean=s["mean"], std=s["std"], pct_ge90=s["pct_ge90"], n=s["n"],
                 worst3="; ".join(f"S{int(w['subject'])}:{100 * w['value']:.0f}%" for w in s["worst"]))
        if "window" in g and (n_targets_col or "N") in g:
            n = int(g[n_targets_col or "N"].iloc[0])
            r["itr"] = float(np.mean(itr_bits_per_min(n, g[value].values, float(g["window"].iloc[0]) + gaze)))
        rows.append(r)
    return pd.DataFrame(rows)


def spread_subsets(freqs, n, draws=5, seed=0):
    """Evenly spread menus: sort targets by frequency, take every (K/n)-th starting at a random offset."""
    order = np.argsort(freqs)
    K = len(freqs)
    if n >= K:
        return [list(order)]
    rng = np.random.default_rng(seed + n)
    out, seen = [], set()
    for _ in range(200):
        off = rng.uniform(0, K / n)
        idx = tuple(sorted(int(order[int(np.floor(off + j * K / n)) % K]) for j in range(n)))
        if idx not in seen:
            seen.add(idx)
            out.append(list(idx))
        if len(out) == draws:
            break
    return out
