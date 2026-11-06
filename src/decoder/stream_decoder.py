"""Real-time streaming SSVEP decoder: the interface the real EEG band and the UI will use.

    band  --(EEG chunks, e.g. 0.1 s)-->  StreamDecoder.push(chunk)  --> list of events
    UI    --(menu starts flickering)-->  StreamDecoder.stimulus_start(t)

Each evaluation emits one event (a dict):
    {"state": "SELECT", "option_id": i, "confidence": c, "timestamp": t, "window_s": w, ...}
    {"state": "IDLE",   "option_id": None, "confidence": c, "timestamp": t, "window_s": w, ...}
option_id indexes the menu given to the decoder; timestamp is stream time (s) at the window end.

Two timing modes:
  sync  (default): the UI calls stimulus_start(t0) when the menu starts flickering. Windows are anchored at
                   t0 + latency and grow: fixed mode evaluates once at window_s; dynamic mode evaluates every
                   step_s from min_window_s to max_window_s and selects as soon as confidence >= threshold.
                   If the threshold is never reached the attempt ends with a final IDLE event ("final": True).
                   Template methods (transfer templates, TRCA) need this mode: they are phase-locked.
  async:           no markers. A sliding window of window_s is evaluated every eval_every_s; after a SELECT the
                   buffer is cleared so the next decision needs a full fresh window. FBCCA only.

Signal path per window (same as the offline experiments): causal 50 Hz notch on the incoming stream ->
ring buffer -> window -> zero-phase filter bank on the window -> scorer -> confidence (src.metrics).
"""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import iirnotch, lfilter, lfilter_zi, sosfiltfilt

from .cca import orth_basis, ref_signals
from .config import FB, HI, LATENCY, N_HARM, NOTCH_HZ
from .fbcca import fb_weights
from .filters import bandpass_sos, subband_lows
from .metrics import CONF_MEASURES, confidence
from .transfer import template_scores
from .trca import ensemble_scores


# --------------------------------------------------------------------------- data source
class FakeBand:
    """Replays a continuous recording x (channels, samples) in chunks, optionally in real time.
    Yields (chunk (channels, m), t_end_seconds, markers_in_chunk) where markers are (t_seconds, label)."""

    def __init__(self, x, fs, chunk_s=0.1, realtime=False, markers=()):
        self.x, self.fs, self.realtime = np.asarray(x), float(fs), realtime
        self.m = int(round(chunk_s * fs))
        self.markers = sorted(markers)

    def __iter__(self):
        t_start = time.perf_counter()
        mi = 0
        for a in range(0, self.x.shape[1], self.m):
            b = min(a + self.m, self.x.shape[1])
            t_end = b / self.fs
            if self.realtime:
                lag = t_start + t_end - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            mk = []
            while mi < len(self.markers) and self.markers[mi][0] < t_end:
                mk.append(self.markers[mi])
                mi += 1
            yield self.x[:, a:b], t_end, mk


class RingBuffer:
    """Fixed-capacity multi-channel buffer. Keeps the absolute sample count for anchored windows."""

    def __init__(self, n_channels, capacity, dtype=np.float64):
        self.buf = np.zeros((n_channels, capacity), dtype)
        self.cap, self.n = capacity, 0  # n = total samples ever written

    def append(self, chunk):
        m = chunk.shape[1]
        if m >= self.cap:
            self.buf[:] = chunk[:, -self.cap:]
        else:
            i = self.n % self.cap
            j = min(m, self.cap - i)
            self.buf[:, i:i + j] = chunk[:, :j]
            self.buf[:, :m - j] = chunk[:, j:]
        self.n += m

    def get(self, start, stop):
        """Samples [start, stop) in absolute sample indices (must still be in the buffer)."""
        if stop > self.n or start < self.n - self.cap or start < 0:
            raise IndexError(f"samples {start}:{stop} not in buffer (have {max(0, self.n - self.cap)}:{self.n})")
        idx = np.arange(start, stop) % self.cap
        return self.buf[:, idx]


# --------------------------------------------------------------------------- scorers
class _FilterBank:
    def __init__(self, fs, f_min, dtype):
        self.lows = subband_lows(FB["scheme"], FB["n_bands"], f_min)
        self.sos = [bandpass_sos(float(fs), float(lo), float(HI)).astype(dtype) for lo in self.lows]
        self.w = fb_weights(FB["n_bands"], FB["a"], FB["b"]).astype(dtype)

    def __call__(self, x):  # (C, n) -> (B, 1, C, n)
        return np.stack([sosfiltfilt(s, x, axis=-1) for s in self.sos])[:, None]


class FBCCAScorer:
    """Calibration-free. scores(x (C, n)) -> (K,)."""
    phase_locked = False
    name = "fbcca"

    def __init__(self, freqs, fs, n_harm=N_HARM, dtype=np.float64):
        self.freqs, self.fs, self.n_harm, self.dtype = np.asarray(freqs, float), float(fs), n_harm, dtype
        self.fb = _FilterBank(fs, float(np.min(freqs)), dtype)
        self._qy = {}

    def qy(self, n):
        if n not in self._qy:
            self._qy[n] = orth_basis(ref_signals(self.freqs, n, self.fs, self.n_harm)).astype(self.dtype)
        return self._qy[n]

    def scores(self, x):
        xfb = self.fb(x.astype(self.dtype, copy=False))
        qy = self.qy(x.shape[-1])
        out = 0
        for m, xb in enumerate(xfb):
            qx = orth_basis(xb)  # (1, n, C)
            s = np.linalg.svd(np.swapaxes(qx, -1, -2)[:, None] @ qy[None], compute_uv=False)[0, :, 0]
            out = out + self.fb.w[m] * s ** 2
        return out


class TemplateScorer:
    """Transfer templates (or the user's own). templates: {window_s: (B, K, C, n)} for the menu targets."""
    phase_locked = True
    name = "tmpl"

    def __init__(self, freqs, fs, templates, n_harm=N_HARM, dtype=np.float64):
        self.base = FBCCAScorer(freqs, fs, n_harm, dtype)
        self.templates = {int(round(float(k) * fs)): v.astype(dtype) for k, v in templates.items()}

    def scores(self, x):
        xfb = self.base.fb(x.astype(self.base.dtype, copy=False))
        return template_scores(xfb, self.templates[x.shape[-1]], self.base.qy(x.shape[-1]), self.base.fb.w)[0]


class TRCAScorer:
    """Ensemble TRCA. models: {window_s: (filters (B, C, K), templates (B, K, C, n))} for the menu targets."""
    phase_locked = True
    name = "trca"

    def __init__(self, freqs, fs, models, dtype=np.float64):
        self.fs, self.dtype = float(fs), dtype
        self.fb = _FilterBank(fs, float(np.min(freqs)), dtype)
        self.models = {int(round(float(k) * fs)): (f.astype(dtype), t.astype(dtype)) for k, (f, t) in models.items()}

    def scores(self, x):
        xfb = self.fb(x.astype(self.dtype, copy=False))
        xfb = xfb - xfb.mean(-1, keepdims=True)
        f, t = self.models[x.shape[-1]]
        return ensemble_scores(xfb, f, t, self.fb.w)[0]


# --------------------------------------------------------------------------- decoder
@dataclass
class StreamDecoder:
    scorer: object
    fs: float
    n_channels: int
    mode: str = "sync"  # "sync" | "async"
    window_s: float = 1.0  # fixed window (sync without dynamic, or async)
    dynamic: bool = False  # sync only: grow the window and stop early
    min_window_s: float = 0.5
    max_window_s: float = 3.0
    step_s: float = 0.25
    eval_every_s: float = 0.1  # async only
    latency_s: float = LATENCY
    threshold: float = -np.inf  # accept a pick only if confidence >= threshold
    measure: str = "margin"  # max | margin | ratio (src.metrics.confidence)
    notch_hz: float = NOTCH_HZ  # None to disable
    dtype: type = np.float64
    timing: list = field(default_factory=list)  # (wall ms, cpu ms) per evaluation

    def __post_init__(self):
        if self.mode == "async" and getattr(self.scorer, "phase_locked", False):
            raise ValueError("phase-locked scorers (templates, TRCA) need mode='sync' with stimulus_start()")
        self.mi = CONF_MEASURES.index(self.measure)
        longest = max(self.window_s, self.max_window_s if self.dynamic else 0)
        self.ring = RingBuffer(self.n_channels, int(np.ceil((longest + self.latency_s + 1.0) * self.fs)), self.dtype)
        if self.notch_hz:
            b, a = iirnotch(self.notch_hz, 30.0, self.fs)
            self._ba = (b.astype(self.dtype), a.astype(self.dtype))
            self._zi = None
        self.reset()

    # ---- control
    def reset(self):
        """Forget the current attempt (call when the menu changes)."""
        self.anchor = None  # absolute sample index where the current window starts (sync)
        self.next_w = None  # next window length to evaluate (s), sync
        self.async_from = self.ring.n  # async: earliest sample a window may start at
        self.next_eval = None

    def stimulus_start(self, t):
        """UI: the menu started flickering at stream time t (s)."""
        self.reset()
        self.anchor = int(round((t + self.latency_s) * self.fs))
        self.next_w = self.min_window_s if self.dynamic else self.window_s

    # ---- data in
    def _notch(self, chunk):
        if not self.notch_hz:
            return chunk
        b, a = self._ba
        if self._zi is None:
            self._zi = (lfilter_zi(b, a).astype(self.dtype)[None, :] * chunk[:, :1])
        y, self._zi = lfilter(b, a, chunk, axis=-1, zi=self._zi)
        return y

    def push(self, chunk):
        """chunk: (channels, m) new samples. Returns the list of events produced (often empty)."""
        self.ring.append(self._notch(np.asarray(chunk, self.dtype)))
        return self._sync_events() if self.mode == "sync" else self._async_events()

    # ---- evaluation
    def _evaluate(self, x, t_end, final=False):
        w0, c0 = time.perf_counter(), time.process_time()
        sc = self.scorer.scores(x)
        conf = float(confidence(sc)[self.mi])
        k = int(np.argmax(sc))
        wall, cpu = 1e3 * (time.perf_counter() - w0), 1e3 * (time.process_time() - c0)
        self.timing.append((wall, cpu))
        ev = dict(confidence=conf, timestamp=t_end, window_s=x.shape[-1] / self.fs, compute_ms=wall,
                  best_option=k, buffer_delay_s=self.ring.n / self.fs - t_end)
        if conf >= self.threshold:
            ev.update(state="SELECT", option_id=k)
        else:
            ev.update(state="IDLE", option_id=None, final=final)
        return ev

    def _sync_events(self):
        out = []
        while self.anchor is not None and self.anchor + int(round(self.next_w * self.fs)) <= self.ring.n:
            a, n = self.anchor, int(round(self.next_w * self.fs))
            last = (not self.dynamic) or self.next_w >= self.max_window_s - 1e-9
            ev = self._evaluate(self.ring.get(a, a + n), (a + n) / self.fs, final=last)
            out.append(ev)
            if ev["state"] == "SELECT" or last:
                self.anchor = None  # attempt over; wait for the next stimulus_start
            else:
                self.next_w = round(self.next_w + self.step_s, 6)
        return out

    def _async_events(self):
        out = []
        n = int(round(self.window_s * self.fs))
        step = int(round(self.eval_every_s * self.fs))
        if self.next_eval is None:
            self.next_eval = self.async_from + n
        while self.next_eval <= self.ring.n:
            e = self.next_eval
            ev = self._evaluate(self.ring.get(e - n, e), e / self.fs)
            out.append(ev)
            if ev["state"] == "SELECT":
                self.next_eval = e + n  # refractory: the next window must be all new data
            else:
                self.next_eval = e + step
        return out

    def state_bytes(self):
        """Memory held by the decoder state (ring buffer + references + templates/models)."""
        tot = self.ring.buf.nbytes
        s = self.scorer
        base = getattr(s, "base", s)
        tot += sum(v.nbytes for v in getattr(base, "_qy", {}).values())
        tot += sum(v.nbytes for v in getattr(s, "templates", {}).values())
        tot += sum(f.nbytes + t.nbytes for f, t in getattr(s, "models", {}).values())
        return int(tot)


# --------------------------------------------------------------------------- cost model
def flops_per_decision(method, C, n, K, B=None, H=N_HARM, n_sos=4, pad=27):
    """Approximate floating-point operations for ONE evaluation of one window (n samples, C channels,
    K menu options, B sub-bands, H harmonics). Precomputable parts (references, templates) excluded.
      filter bank : B x C x 2 passes x (n + 2 pad) x n_sos sections x 9 flops (biquad: 5 mul + 4 add)
      QR of window: B x (2 n C^2)                          (Householder)
      CCA, per target and band: 2 n C (2H)  (Qx^T Qy)  +  4 C (2H)^2 (top singular value)
      templates   : + per target and band 2 n C^2 + 4 C^3 (CCA with template) + 4 n C (projections)
      TRCA        : per band 2 C K n (spatial filtering) + 3 K^2 n (correlations); no QR / CCA.
    """
    B = B or FB["n_bands"]
    p = 2 * H
    filt = B * C * 2 * (n + 2 * pad) * n_sos * 9
    if method == "trca":
        return int(filt + B * (2 * C * K * n + 3 * K * K * n))
    qr = B * 2 * n * C * C
    cca = B * K * (2 * n * C * p + 4 * C * p * p)
    extra = B * K * (2 * n * C * C + 4 * C ** 3 + 4 * n * C) if method == "tmpl" else 0
    return int(filt + qr + cca + extra)


def notch_flops_per_second(C, fs):
    return int(C * fs * 9)
