"""Per-trial decoder scores for 6-option menus at every window length, cached on disk.

Used by E5 (dry vs wet), E6 (idle detection), E7 (dynamic stopping) and the notch check, so the
expensive part (filtering + CCA / TRCA) runs once.

For each dataset "label" (d1, d2, d3, d4wet, d4dry) the cache holds
  scores  (methods, windows, menus, trials, 6) float32   decoder score of each trial for each menu option
  y, block, subject  (trials,)                           true target index (0..K-1), block, subject
  menus   (menus, 6)                                     target indices of each 6-option menu
Methods (all evaluated with no information from the test block or the test subject):
  fbcca : filter-bank CCA, no calibration.
  tmpl  : template scorer (src/transfer.py) with templates averaged over all OTHER subjects of the same
          dataset/electrode type (zero calibration for the new user).
  trca  : ensemble TRCA on the 6 menu targets, trained on the user's other blocks (leave-one-block-out).
"""
import hashlib
from pathlib import Path

import numpy as np

from .cca import cca_corr, orth_basis, ref_signals
from .config import FB, HI, LATENCY, N_HARM
from .filters import subband_lows
from .pipeline import fb_w, load_prepped, spread_subsets, windowed_fb
from .transfer import template_scores
from .trca import TRCAStats

METHODS = ("fbcca", "tmpl", "trca")
N_OPT = 6
LABELS = {"d1": ["d1"], "d2": ["d2"], "d3": ["d3"], "d4": ["d4wet", "d4dry"]}
MAX_W = {"d1": 3.0, "d2": 3.0, "d3": 2.0, "d4wet": 2.0, "d4dry": 2.0}  # limited by the shortest epochs


def window_grid(label, step=0.25, wmin=0.5):
    return [round(float(w), 2) for w in np.arange(wmin, MAX_W[label] + 1e-9, step)]


def dkey_of(label):
    return label[:2]


def _conditions(dkey, d):
    if dkey == "d4":
        return [("d4wet", d.select(d.cond == "wet")), ("d4dry", d.select(d.cond == "dry"))]
    return [(dkey, d)]


def _fb(d, windows):
    lows = subband_lows(FB["scheme"], FB["n_bands"], float(d.freqs.min()))
    return windowed_fb(d, lows, HI, windows, LATENCY, "window")


def _class_means(xfb, y, K):
    x = xfb - xfb.mean(-1, keepdims=True)
    return np.stack([x[:, y == k].mean(1) for k in range(K)], 1)  # (B, K, C, n)


def subject_means(dkey, root, s, windows, notch_hz="default"):
    """Per label and window: class-mean sub-band templates of one subject (B, K, C, n)."""
    d = load_prepped(dkey, root, s, notch_hz=notch_hz)
    out = {}
    for label, dc in _conditions(dkey, d):
        xw = _fb(dc, windows[label])
        out[label] = {w: _class_means(xw[w], dc.y, len(dc.freqs)) for w in windows[label]}
    return out


def subject_scores(dkey, root, s, windows, menus, totals=None, n_subj=None, methods=METHODS, notch_hz="default"):
    """Returns {label: dict(y, block, scores (M, J, n_menus, T, 6))} for one subject."""
    d = load_prepped(dkey, root, s, notch_hz=notch_hz)
    w = fb_w()
    out = {}
    for label, dc in _conditions(dkey, d):
        K = len(dc.freqs)
        xw = _fb(dc, windows[label])
        sc = np.zeros((len(methods), len(windows[label]), len(menus[label]), len(dc.y), N_OPT), np.float32)
        for j, W in enumerate(windows[label]):
            x = xw[W]
            qy = orth_basis(ref_signals(dc.freqs, x.shape[-1], dc.fs, N_HARM))
            for mi, m in enumerate(methods):
                if m == "fbcca":
                    full = (np.stack([cca_corr(orth_basis(xb), qy) for xb in x], -1) ** 2) @ w
                    for u, S in enumerate(menus[label]):
                        sc[mi, j, u] = full[:, S]
                elif m == "tmpl":
                    own = _class_means(x, dc.y, K)
                    tm = (totals[label][W] - own) / (n_subj - 1)  # leave-this-subject-out average
                    full = template_scores(x, tm, qy, w)
                    for u, S in enumerate(menus[label]):
                        sc[mi, j, u] = full[:, S]
                elif m == "trca":
                    st = TRCAStats(x, dc.y, dc.block)
                    for bi, b in enumerate(st.blocks):
                        f, t = st.fit([i for i in range(len(st.blocks)) if i != bi])
                        te = dc.block == b
                        for u, S in enumerate(menus[label]):
                            sc[mi, j, u, te] = st.scores(te, f, t, w, cls=list(S))
                else:
                    raise ValueError(m)
        out[label] = dict(y=dc.y.copy(), block=dc.block.copy(), scores=sc)
    return out


def _path(cache, label, notch_hz, windows, methods):
    tag = hashlib.md5(repr((list(windows), list(methods))).encode()).hexdigest()[:8]
    return Path(cache) / f"scores_{label}_notch{notch_hz or 'off'}_{tag}.npz"


def get_scores(dkey, root, cache, notch_hz="default", windows=None, methods=METHODS, n_jobs=None, log=print):
    """Load (or compute and save) the score cache for every label of a dataset. Returns {label: dict}."""
    from joblib import Parallel, delayed
    from . import N_JOBS
    from .config import NOTCH
    from .datasets import get_loader
    nz = NOTCH.get(dkey) if notch_hz == "default" else notch_hz
    labels = LABELS[dkey]
    windows = windows or {lab: window_grid(lab) for lab in labels}
    paths = {lab: _path(cache, lab, nz, windows[lab], methods) for lab in labels}
    if all(p.exists() for p in paths.values()):
        return {lab: dict(np.load(p, allow_pickle=False)) for lab, p in paths.items()}
    Path(cache).mkdir(parents=True, exist_ok=True)
    loader, subjects = get_loader(dkey)
    freqs = loader(root, subjects[0]).freqs
    menus = {lab: [list(S) for S in spread_subsets(freqs, N_OPT)] for lab in labels}
    totals = None
    if "tmpl" in methods:
        log(f"[{dkey}] pass 1: subject templates")
        totals = None
        for m in Parallel(n_jobs=n_jobs or N_JOBS, return_as="generator")(
                delayed(subject_means)(dkey, root, s, windows, nz) for s in subjects):
            if totals is None:
                totals = m
            else:
                for lab in labels:
                    for W in windows[lab]:
                        totals[lab][W] += m[lab][W]
    log(f"[{dkey}] pass 2: scores for {len(subjects)} subjects, windows {windows}")
    res = Parallel(n_jobs=n_jobs or N_JOBS)(
        delayed(subject_scores)(dkey, root, s, windows, menus, totals, len(subjects), methods, nz) for s in subjects)
    out = {}
    for lab in labels:
        r = [x[lab] for x in res]
        out[lab] = dict(
            scores=np.concatenate([x["scores"] for x in r], 3),
            y=np.concatenate([x["y"] for x in r]), block=np.concatenate([x["block"] for x in r]),
            subject=np.concatenate([np.full(len(x["y"]), s) for x, s in zip(r, subjects)]),
            menus=np.array(menus[lab]), windows=np.array(windows[lab]), methods=np.array(methods),
            freqs=np.asarray(freqs, float))
        np.savez_compressed(paths[lab], **out[lab])
    return out


def menu_view(c, method, W, menu):
    """Convenience: scores (T, 6) of one method / window / menu index, plus y."""
    mi = list(c["methods"]).index(method)
    j = int(np.argmin(np.abs(c["windows"] - W)))
    return c["scores"][mi, j, menu]


def per_subject_accuracy(c, label=None):
    """DataFrame (subject, method, window, acc): 6-option accuracy on target-present trials,
    averaged over the menus."""
    import pandas as pd
    rows = []
    subj = c["subject"]
    for mi, m in enumerate(c["methods"]):
        for j, W in enumerate(c["windows"]):
            hit = np.zeros(len(subj))
            cnt = np.zeros(len(subj))
            for u, S in enumerate(c["menus"]):
                pres = np.isin(c["y"], S)
                pred = S[c["scores"][mi, j, u].argmax(1)]
                hit += pres & (pred == c["y"])
                cnt += pres
            for s in np.unique(subj):
                k = subj == s
                rows.append(dict(label=label, subject=int(s), method=str(m), window=float(W),
                                 acc=float(hit[k].sum() / cnt[k].sum())))
    return pd.DataFrame(rows)
