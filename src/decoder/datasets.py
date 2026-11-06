"""Dataset loaders -> one common format.

SSVEPData fields
  X        (trials, channels, samples) float64 epochs, including pre-stimulus data
  y        (trials,) target index into `freqs`
  fs       sampling rate (Hz)
  freqs    (K,) stimulus frequency per target index
  phases   (K,) stimulus phase (rad) per target index (0 if unknown)
  ch_names list of channel names (same order as X axis 1)
  subject  subject id (int)
  block    (trials,) block / run index, used for cross-validation (never split randomly)
  onset    sample index of stimulus onset in each epoch
  cond     (trials,) optional condition label (e.g. 'wet'/'dry'), else None
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio


@dataclass
class SSVEPData:
    X: np.ndarray
    y: np.ndarray
    fs: float
    freqs: np.ndarray
    phases: np.ndarray
    ch_names: list
    subject: int
    block: np.ndarray
    onset: int
    name: str = ""
    cond: np.ndarray = None
    info: dict = field(default_factory=dict)

    def pick_channels(self, names):
        idx = [self.ch_names.index(n) for n in names]
        return SSVEPData(self.X[:, idx], self.y, self.fs, self.freqs, self.phases, [self.ch_names[i] for i in idx],
                         self.subject, self.block, self.onset, self.name, self.cond, dict(self.info))

    def select(self, mask):
        mask = np.asarray(mask)
        return SSVEPData(self.X[mask], self.y[mask], self.fs, self.freqs, self.phases, list(self.ch_names),
                         self.subject, self.block[mask], self.onset, self.name,
                         None if self.cond is None else self.cond[mask], dict(self.info))

    def window(self, length_s: float, latency_s: float, start_s: float = 0.0):
        """Return X[..., onset + latency + start : + length] (unfiltered)."""
        a = self.onset + int(round((latency_s + start_s) * self.fs))
        return self.X[..., a:a + int(round(length_s * self.fs))]


# ---------------------------------------------------------------- D1 Nakanishi 2015
NAKANISHI_FREQS = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
# JFPM phases: f0=9.25 Hz, df=0.5 Hz, phi0=0, dphi=0.5*pi (paper). phase_k = (f_k - 9.25)/0.5 * 0.5*pi.
NAKANISHI_PHASES = np.mod((NAKANISHI_FREQS - 9.25) / 0.5 * 0.5 * np.pi, 2 * np.pi)
# Dataset docs (doc/chan_locs.pdf) only label electrode 7 as Oz; others are BioSemi-128 cap positions.
# The names below are the usual 10-20 equivalents of those positions (approximate).
NAKANISHI_CH = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]


def load_nakanishi(root, subject: int) -> SSVEPData:
    eeg = sio.loadmat(Path(root) / "nakanishi2015" / "data" / f"s{subject}.mat")["eeg"]  # (12, 8, 1114, 15)
    if eeg.shape != (12, 8, 1114, 15):
        raise ValueError(f"unexpected shape {eeg.shape}")
    k, c, n, b = eeg.shape
    x = eeg.transpose(3, 0, 1, 2).reshape(b * k, c, n).astype(np.float64)  # trial order: block-major
    y = np.tile(np.arange(k), b)
    block = np.repeat(np.arange(b), k)
    return SSVEPData(x, y, 256.0, NAKANISHI_FREQS, NAKANISHI_PHASES, list(NAKANISHI_CH), subject, block,
                     onset=38, name="nakanishi2015")


NAKANISHI_SUBJECTS = list(range(1, 11))


# ---------------------------------------------------------------- Tsinghua 64-channel montage (D2, D3)
# Channel order as listed in BETA's own files (suppl_info.chan); Benchmark uses the same lab montage
# (64-channels.loc). Verified empirically for D2 by SSVEP SNR per channel (see scripts/check_datasets.py).
TSINGHUA64 = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
              "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
              "M1", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "PZ",
              "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
OCCIPITAL9 = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]


def _tsinghua_jfpm(f0=8.0, df=0.2, dphi=0.5 * np.pi, n=40, per_row=8):
    """Benchmark target order: 8, 9, ..., 15, 8.2, 9.2, ..., 15.8 (5 rows of 8)."""
    freqs = np.array([f0 + c + r * df for r in range(n // per_row) for c in range(per_row)])
    phases = np.mod((np.round((freqs - f0) / df)) * dphi, 2 * np.pi)
    return freqs, phases


BENCH_FREQS, BENCH_PHASES = _tsinghua_jfpm()
BENCH_SUBJECTS = list(range(1, 35))  # S35 not in the NEMAR mirror


def load_benchmark(root, subject: int, channels=OCCIPITAL9) -> SSVEPData:
    """D2 Wang et al. 2016. File 'data': (64 ch, 1500 samples, 40 targets, 6 blocks), 250 Hz, onset at 0.5 s."""
    raw = sio.loadmat(Path(root) / "benchmark" / f"S{subject}.mat")["data"]
    if raw.shape != (64, 1500, 40, 6):
        raise ValueError(f"unexpected shape {raw.shape}")
    idx = [TSINGHUA64.index(c) for c in channels] if channels is not None else list(range(64))
    raw = raw[idx]
    c, n, k, b = raw.shape
    x = raw.transpose(3, 2, 0, 1).reshape(b * k, c, n).astype(np.float64)
    return SSVEPData(x, np.tile(np.arange(k), b), 250.0, BENCH_FREQS, BENCH_PHASES,
                     [TSINGHUA64[i] for i in idx], subject, np.repeat(np.arange(b), k), onset=125, name="benchmark")


# ---------------------------------------------------------------- D3 BETA (Liu et al. 2020)
BETA_SUBJECTS = list(range(1, 71))


def load_beta(root, subject: int, channels=OCCIPITAL9) -> SSVEPData:
    """D3 BETA. data.EEG: (64, 750|1000, 4 blocks, 40 targets), 250 Hz, onset at 0.5 s.
    S1-S15: 2 s stimulation; S16-S70: 3 s. Pre-filtered 3-100 Hz by the authors."""
    m = sio.loadmat(Path(root) / "beta" / f"S{subject}.mat")["data"][0, 0]
    eeg = m["EEG"]
    si = m["suppl_info"][0, 0]
    names = [str(row[3][0]).upper() for row in si["chan"]]
    freqs = np.asarray(si["freqs"], float).ravel()
    phases = np.asarray(si["phases"], float).ravel()
    fs = float(np.asarray(si["srate"]).ravel()[0])
    idx = [names.index(c) for c in channels] if channels is not None else list(range(len(names)))
    eeg = eeg[idx]
    c, n, b, k = eeg.shape
    x = eeg.transpose(2, 3, 0, 1).reshape(b * k, c, n).astype(np.float64)
    return SSVEPData(x, np.tile(np.arange(k), b), fs, freqs, phases, [names[i] for i in idx], subject,
                     np.repeat(np.arange(b), k), onset=int(round(0.5 * fs)), name="beta",
                     info={"stim_s": (n / fs) - 1.0})


# ---------------------------------------------------------------- D4 Wearable (Zhu et al. 2021)
WEAR_CH = ["POZ", "PO3", "PO4", "PO5", "PO6", "OZ", "O1", "O2"]  # order from the paper (Sensors 21:1256)
WEAR_SUBJECTS = list(range(1, 103))


def load_wearable(root, subject: int) -> SSVEPData:
    """D4. data: (8 ch, 710 samples, 2 electrode types, 10 blocks, 12 targets), 250 Hz,
    onset at 0.5 s, NO filtering applied by the authors. cond = 'wet'/'dry'; blocks 0-9 within each.

    Electrode-type order: the Readme only states it for Impedance.mat ("1: wet, 2: dry"), but there headband 1
    has a median impedance of ~146 kOhm and headband 2 ~15 kOhm, i.e. 1 is physically the DRY band. In the EEG
    array, index 0 likewise has ~8x more 50 Hz power, ~8x larger DC offsets and 2x the std of index 1
    (median over S001-S020). So index 0 = dry, index 1 = wet (see docs/decoder.md)."""
    d = sio.loadmat(Path(root) / "wearable" / f"S{subject:03d}.mat")["data"]
    if d.shape != (8, 710, 2, 10, 12):
        raise ValueError(f"unexpected shape {d.shape}")
    c, n, e, b, k = d.shape
    x = d.transpose(2, 3, 4, 0, 1).reshape(e * b * k, c, n).astype(np.float64)
    cond = np.repeat(np.array(["dry", "wet"]), b * k)  # see docstring
    return SSVEPData(x, np.tile(np.arange(k), e * b), 250.0, NAKANISHI_FREQS, NAKANISHI_PHASES, list(WEAR_CH),
                     subject, np.tile(np.repeat(np.arange(b), k), e), onset=125, name="wearable", cond=cond)


# ---------------------------------------------------------------- registry
def get_loader(key: str):
    """Returns (load_fn(root, subject) -> SSVEPData, subject list)."""
    return {
        "d1": (load_nakanishi, NAKANISHI_SUBJECTS),
        "d2": (load_benchmark, BENCH_SUBJECTS),
        "d3": (load_beta, BETA_SUBJECTS),
        "d4": (load_wearable, WEAR_SUBJECTS),
    }[key]
