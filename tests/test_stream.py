"""Stream-decoder tests on synthetic signals. Run: .venv/bin/python -I -m pytest tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from decoder.stream_decoder import (FakeBand, FBCCAScorer, RingBuffer, StreamDecoder, TemplateScorer,  # noqa: E402
                                flops_per_decision)

FS = 250.0
MENU = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])


def stream(segments, n_ch=6, seed=0, amp=0.8, mains=0.0):
    """segments: list of (seconds, freq or None). Sine (+2nd harmonic) per segment + white noise (+ 50 Hz)."""
    rng = np.random.default_rng(seed)
    parts, t0, onsets = [], 0.0, []
    gains = np.linspace(0.4, 1.0, n_ch)[:, None]
    for sec, f in segments:
        t = np.arange(int(sec * FS)) / FS
        s = np.zeros_like(t) if f is None else np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)
        x = gains * s * amp + rng.normal(0, 1, (n_ch, len(t))) + mains * np.sin(2 * np.pi * 50 * (t + t0))
        parts.append(x)
        onsets.append(t0)
        t0 += sec
    return np.concatenate(parts, 1), onsets


def replay(dec, x, markers=()):
    evs = []
    for chunk, _, mk in FakeBand(x, FS, 0.1, markers=markers):
        for t, _ in mk:
            dec.stimulus_start(t)
        evs += dec.push(chunk)
    return evs


def test_ring_buffer_wraparound():
    rb = RingBuffer(2, 10)
    data = np.arange(2 * 27, dtype=float).reshape(2, 27)
    for a in range(0, 27, 4):
        rb.append(data[:, a:a + 4])
    assert rb.n == 27
    np.testing.assert_array_equal(rb.get(17, 27), data[:, 17:27])
    np.testing.assert_array_equal(rb.get(20, 24), data[:, 20:24])
    try:
        rb.get(5, 10)
        assert False, "old samples must not be returned"
    except IndexError:
        pass


def test_sync_fixed_selects_10hz_despite_mains():
    x, on = stream([(1.0, None), (2.0, 10.0), (1.0, None)], mains=5.0)
    dec = StreamDecoder(FBCCAScorer(MENU, FS), FS, 6, "sync", window_s=1.0)
    evs = replay(dec, x, [(on[1], "go")])
    assert len(evs) == 1 and evs[0]["state"] == "SELECT" and MENU[evs[0]["option_id"]] == 10.0
    # window anchored at onset + latency, decided once the 0.1 s chunk containing its end arrived
    assert abs(evs[0]["timestamp"] - (on[1] + dec.latency_s + 1.0)) < 1 / FS
    assert 0 <= evs[0]["buffer_delay_s"] < 0.1 + 1e-9


def test_sync_dynamic_stops_early_on_clear_signal_and_idles_on_noise():
    x, on = stream([(0.5, None), (3.5, 11.0), (3.5, None)], amp=1.5)
    dec = StreamDecoder(FBCCAScorer(MENU, FS), FS, 6, "sync", dynamic=True, threshold=0.5, measure="margin")
    evs = replay(dec, x, [(on[1], "a"), (on[2], "b")])
    sel = [e for e in evs if e["state"] == "SELECT"]
    assert len(sel) == 1 and MENU[sel[0]["option_id"]] == 11.0 and sel[0]["window_s"] < 3.0
    # the noise-only attempt must end with a final IDLE at the longest window
    last = evs[-1]
    assert last["state"] == "IDLE" and last["final"] and abs(last["window_s"] - 3.0) < 1e-6


def test_async_picks_target_and_stays_quiet_on_noise():
    x, _ = stream([(6.0, None), (4.0, 12.0), (6.0, None)], amp=1.2, seed=3)
    dec = StreamDecoder(FBCCAScorer(MENU, FS), FS, 6, "async", window_s=1.0, threshold=0.5, measure="margin")
    evs = replay(dec, x)
    sel = [e for e in evs if e["state"] == "SELECT"]
    assert sel, "the 12 Hz segment must be selected"
    assert all(MENU[e["option_id"]] == 12.0 and 6.3 <= e["timestamp"] <= 10.9 for e in sel)
    gaps = np.diff([e["timestamp"] for e in sel])
    assert np.all(gaps >= 1.0 - 1e-9)  # refractory: a new pick needs a full fresh window


def test_template_scorer_requires_sync_and_float32_agrees():
    x, on = stream([(0.5, None), (2.0, 9.0), (1.0, None)], amp=1.0, seed=5)
    n = int(FS)
    t = np.arange(n) / FS
    tm = np.stack([np.sin(2 * np.pi * f * (t + 0.14)) for f in MENU])  # (K, n): idealised templates
    tm = np.broadcast_to(tm[None, :, None, :], (5, len(MENU), 6, n)).copy()
    try:
        StreamDecoder(TemplateScorer(MENU, FS, {1.0: tm}), FS, 6, "async")
        assert False, "phase-locked scorer must refuse async mode"
    except ValueError:
        pass
    picks = []
    for dt in [np.float64, np.float32]:
        dec = StreamDecoder(FBCCAScorer(MENU, FS, dtype=dt), FS, 6, "sync", window_s=1.0, dtype=dt)
        ev = replay(dec, x, [(on[1], "go")])[0]
        picks.append((ev["option_id"], ev["confidence"]))
    assert picks[0][0] == picks[1][0] and abs(picks[0][1] - picks[1][1]) < 1e-3


def test_flops_grow_with_window_and_channels():
    a = flops_per_decision("fbcca", 6, 250, 6)
    assert flops_per_decision("fbcca", 6, 500, 6) > 1.8 * a
    assert flops_per_decision("fbcca", 8, 250, 6) > a
    assert flops_per_decision("trca", 6, 250, 6) < flops_per_decision("tmpl", 6, 250, 6)
