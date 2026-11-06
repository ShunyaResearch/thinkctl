"""ReplayBand: streams real D4 wet EEG of one held-out subject in 0.1 s chunks, in real time.

Interface shared with the future live band:
    band.start(); chunk, t_end = band.read()   # chunk (channels, samples) raw microvolts; t_end in band seconds
    band.fs, band.channels, band.wall_time(t)  # band clock -> time.perf_counter() clock; band.stop()
Replay-only hook (the scripted user's eyes):
    t_onset = band.look(freq_hz or None)
The next chunk starts a real trial of that frequency (None -> an 'idle' trial: a flicker that is NOT in the menu,
the offline stand-in for looking elsewhere). The trial starts 0.5 s before its stimulus onset; t_onset (band time) is
when the flicker must start. Between looks, idle trials are streamed back to back.
Trials are drawn at random with a fixed seed, without replacement per frequency (reshuffled when used up): no
cherry-picking. The subject's own data never enters the transfer templates (band/models.py).
"""
import queue, sys, threading, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))
from decoder.datasets import NAKANISHI_FREQS, load_wearable  # noqa: E402


class _Pool:
    def __init__(self, idx, rng):
        self.idx, self.rng, self.order = list(idx), rng, []

    def draw(self):
        if not self.order:
            self.order = list(self.rng.permutation(self.idx))
        return int(self.order.pop())


class ReplayBand:
    def __init__(self, data_root, subject, channels, menu_hz, seed=0, chunk_s=0.1, realtime=True):
        d = load_wearable(data_root, subject)
        d = d.select(d.cond == "wet").pick_channels(list(channels))
        self.X, self.y, self.fs, self.onset = d.X, d.y, float(d.fs), d.onset
        self.channels, self.subject = list(channels), subject
        self.m = int(round(chunk_s * self.fs))
        self.realtime = realtime
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.pools = {float(f): _Pool(np.flatnonzero(self.y == k), rng) for k, f in enumerate(NAKANISHI_FREQS)}
        self.active = [float(f) for f in menu_hz]  # frequencies currently on the menu (changes when the set swaps)
        self.idle_pool = _Pool(np.flatnonzero(~np.isin(NAKANISHI_FREQS[self.y], self.active)), rng)
        self.tape = np.zeros((len(channels), 0))
        self.n_out = 0  # samples emitted
        self.log = []  # (band time of stimulus onset, requested freq, trial index, true freq)
        self.q = queue.Queue(maxsize=50)
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.t0_wall = None

    # ---------------------------------------------------------------- tape
    def _append(self, trial):
        self.tape = np.concatenate([self.tape, self.X[trial]], 1)

    def set_active(self, menu_hz):
        """The menu's frequency set changed: 'looking elsewhere' must use frequencies outside the new set."""
        with self.lock:
            if [float(f) for f in menu_hz] != self.active:
                self.active = [float(f) for f in menu_hz]
                self.idle_pool = _Pool(np.flatnonzero(~np.isin(NAKANISHI_FREQS[self.y], self.active)), self.rng)

    def look(self, freq_hz):
        """Scripted user looks at freq_hz (or None = not at the menu) from the next chunk on."""
        with self.lock:
            pool = self.pools[float(freq_hz)] if freq_hz is not None else self.idle_pool
            k = pool.draw()
            self.tape = np.zeros((len(self.channels), 0))
            self._append(k)
            t_onset = (self.n_out + self.onset) / self.fs
            self.log.append((t_onset, freq_hz, k, float(NAKANISHI_FREQS[self.y[k]])))
            return t_onset

    def _next_chunk(self):
        with self.lock:
            while self.tape.shape[1] < self.m:
                self._append(self.idle_pool.draw())
            c, self.tape = self.tape[:, :self.m], self.tape[:, self.m:]
            self.n_out += self.m
            return c, self.n_out / self.fs

    # ---------------------------------------------------------------- streaming
    def _run(self):
        self.t0_wall = time.perf_counter()
        while not self._stop.is_set():
            c, t_end = self._next_chunk()
            if self.realtime:
                lag = self.t0_wall + t_end - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            self.q.put((c, t_end))

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        while self.t0_wall is None:
            time.sleep(0.001)

    def read(self, timeout=1.0):
        return self.q.get(timeout=timeout)

    def wall_time(self, t_band):
        return self.t0_wall + t_band

    def stop(self):
        self._stop.set()
