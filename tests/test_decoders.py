"""Unit tests on synthetic signals. Run: .venv/bin/python -I -m pytest tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from decoder.cca import cca_scores  # noqa: E402
from decoder.fbcca import FBCCA  # noqa: E402
from decoder.filters import bandpass  # noqa: E402
from decoder.metrics import itr_bits_per_min  # noqa: E402
from decoder.trca import TRCA  # noqa: E402

FS = 250.0
FREQS = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 15.0])


def synth(f, n_trials=1, seconds=1.0, n_ch=8, snr_amp=1.0, seed=0, phase=0.0):
    """Sine at f (+ 2nd harmonic) mixed into channels with different gains, plus white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    s = np.sin(2 * np.pi * f * t + phase) + 0.4 * np.sin(2 * np.pi * 2 * f * t + 2 * phase)
    gains = np.linspace(0.3, 1.0, n_ch)[:, None]
    return np.stack([gains * s * snr_amp + rng.normal(0, 1.0, (n_ch, len(t))) for _ in range(n_trials)])


def test_cca_picks_10hz():
    x = bandpass(synth(10.0, n_trials=20, snr_amp=0.5), FS, 7, 90)
    pred = FREQS[cca_scores(x, FREQS, FS, n_harm=3).argmax(1)]
    assert (pred == 10.0).mean() >= 0.95


def test_fbcca_picks_10hz():
    dec = FBCCA(FREQS, FS, n_bands=3, scheme="8m", n_harm=3)
    pred = FREQS[dec.scores(synth(10.0, n_trials=20, snr_amp=0.5, seed=1)).argmax(1)]
    assert (pred == 10.0).mean() >= 0.95


def test_fbcca_single_window_shape():
    dec = FBCCA(FREQS, FS, n_bands=3)
    assert dec.scores(synth(12.0)[0]).shape == (len(FREQS),)


def test_trca_learns_templates():
    xs, ys = [], []
    for k, f in enumerate(FREQS):
        xs.append(synth(f, n_trials=6, snr_amp=0.3, seed=10 + k, phase=0.7 * k))
        ys += [k] * 6
    x, y = np.concatenate(xs), np.array(ys)
    dec = TRCA(FS, n_bands=3, f_min=8.0).fit(x, y)
    xt = np.concatenate([synth(f, n_trials=5, snr_amp=0.3, seed=100 + k, phase=0.7 * k) for k, f in enumerate(FREQS)])
    yt = np.repeat(np.arange(len(FREQS)), 5)
    assert (dec.scores(xt).argmax(1) == yt).mean() >= 0.9


def test_itr_values():
    # 4 options, perfect accuracy, 1 s per selection: log2(4)=2 bits -> 120 bits/min
    assert abs(itr_bits_per_min(4, 1.0, 1.0) - 120.0) < 1e-9
    # chance accuracy gives 0
    assert itr_bits_per_min(4, 0.25, 1.0) == 0.0
    # N=12, P=0.9, T=1.5: 3.585 + 0.9*log2(0.9) + 0.1*log2(0.1/11) = 2.769 bits -> 110.8 bits/min
    assert abs(itr_bits_per_min(12, 0.9, 1.5) - 110.8) < 0.2


def _trials(seed0=10, n=6, snr=0.3):
    xs, ys, bl = [], [], []
    for k, f in enumerate(FREQS):
        xs.append(synth(f, n_trials=n, snr_amp=snr, seed=seed0 + k, phase=0.7 * k))
        ys += [k] * n
        bl += list(range(n))
    return np.concatenate(xs), np.array(ys), np.array(bl)


def test_fast_trca_matches_reference():
    from decoder.filters import filterbank
    from decoder.pipeline import trca_lobo_scores
    from decoder.trca import TRCAStats
    from decoder.fbcca import fb_weights
    x, y, bl = _trials()
    xfb = filterbank(x, FS, [6, 14, 22], 88)
    ref = trca_lobo_scores(xfb, y, bl, {"n_bands": 3})
    fast = TRCAStats(xfb, y, bl).lobo(fb_weights(3))
    assert (ref.argmax(1) == fast.argmax(1)).mean() > 0.97
    assert (fast.argmax(1) == y).mean() >= 0.9


def test_template_scorer_with_one_trial():
    from decoder.cca import orth_basis, ref_signals
    from decoder.fbcca import fb_weights
    from decoder.filters import filterbank
    from decoder.transfer import template_scores
    x, y, bl = _trials(n=6, snr=0.4)  # template from ONE trial per target (block 0), test on 30 others
    xfb = filterbank(x, FS, [6, 14, 22], 88)
    tmpl = np.stack([np.stack([xfb[m, (y == k) & (bl == 0)].mean(0) for k in range(len(FREQS))]) for m in range(3)])
    te = bl > 0
    qy = orth_basis(ref_signals(FREQS, x.shape[-1], FS, 3))
    sc = template_scores(xfb[:, te], tmpl, qy, fb_weights(3))
    assert (sc.argmax(1) == y[te]).mean() >= 0.9


def test_spread_subsets_are_spread():
    from decoder.pipeline import spread_subsets
    f = np.round(np.arange(8, 16, 0.2), 1)
    for S in spread_subsets(f, 6):
        assert len(S) == 6 and np.min(np.diff(np.sort(f[S]))) >= 1.0
