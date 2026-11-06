"""Accuracy, ITR, subject summaries and ROC helpers."""
import numpy as np


def itr_bits_per_min(n: int, p, t: float):
    """Wolpaw ITR in bits/min.

    ITR = 60/T * [log2 N + P log2 P + (1-P) log2((1-P)/(N-1))]
      N: options, P: accuracy (0..1), T: seconds per selection (stare window + gaze shift).
    Accuracy at or below chance gives 0.
    """
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    if n < 2:
        return np.zeros_like(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = np.log2(n) + np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0.0)
        bits = bits + np.where(p < 1, (1 - p) * np.log2(np.where(p < 1, (1 - p) / (n - 1), 1)), 0.0)
    bits = np.where(p <= 1.0 / n, 0.0, bits)
    return bits * 60.0 / t


def subject_summary(values, subjects=None, worst=3, good=0.90) -> dict:
    """Mean, std across subjects, % of subjects >= `good` (accuracy 0..1), and the worst subjects."""
    v = np.asarray(values, float)
    subjects = list(range(1, len(v) + 1)) if subjects is None else list(subjects)
    order = np.argsort(v)[:worst]
    return {
        "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "pct_ge90": float(100 * np.mean(v >= good - 1e-9)), "n": int(len(v)),
        "worst": [{"subject": subjects[i], "value": float(v[i])} for i in order],
    }


def roc_curve(scores_pos, scores_neg):
    """True-pick rate vs false-pick rate as the acceptance threshold sweeps.

    scores_pos: confidence on target-present trials (accepted only if correct, so pass
                -inf for wrong picks to count them as never 'true').
    scores_neg: confidence on idle trials (any acceptance is a false pick).
    Returns thresholds, tpr, fpr (fpr ascending).
    """
    sp = np.asarray(scores_pos, float)
    sn = np.asarray(scores_neg, float)
    thr = np.unique(np.concatenate([sp[np.isfinite(sp)], sn[np.isfinite(sn)]]))[::-1]
    thr = np.concatenate([[np.inf], thr, [-np.inf]])
    tpr = np.array([(sp >= t).mean() for t in thr])
    fpr = np.array([(sn >= t).mean() for t in thr])
    return thr, tpr, fpr


def pooled_heldout(values, subjects, itr=None, good=0.90):
    """2-fold result: settings picked on half A are scored on half B and vice versa; the two held-out
    halves are pooled so every subject counts once. Returns mean, std, % >= good, worst 3 (+ mean ITR)."""
    s = subject_summary(values, subjects, good=good)
    out = dict(mean=s["mean"], std=s["std"], pct_ge90=s["pct_ge90"], n=s["n"],
               worst3="; ".join(f"S{int(w['subject'])}:{100 * w['value']:.0f}%" for w in s["worst"]))
    if itr is not None:
        out["itr"] = float(np.mean(itr))
    return out


CONF_MEASURES = ("max", "margin", "ratio")


def confidence(scores, axis=-1):
    """Confidence of the top pick from the menu scores s1 >= s2 >= ... >= s_min (along `axis`).
    Returns an array with a new last axis of size 3:
      max    = s1
      margin = s1 - s2
      ratio  = (s1 - s_min) / (s2 - s_min)   how many times further above the weakest option the winner is
                                             than the runner-up; works for any score scale or sign.
    Example: scores (0.50, 0.30, 0.25, 0.22, 0.21, 0.20) -> max 0.50, margin 0.20, ratio 0.30/0.10 = 3.0.
    """
    s = np.moveaxis(np.asarray(scores, float), axis, -1)
    o = np.sort(s, -1)
    s1, s2, lo = o[..., -1], o[..., -2], o[..., 0]
    return np.stack([s1, s1 - s2, (s1 - lo) / np.maximum(s2 - lo, 1e-12)], -1)
