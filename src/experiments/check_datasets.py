"""Sanity checks for every dataset, computed from the data itself:
 - shapes / sampling rate / onset as loaded
 - frequency labels: for each target, the spectrum peak (fundamental) must sit at the labelled frequency
 - channel order (D2): the channels with the strongest SSVEP must be occipital
 - mains (50/60 Hz) line-noise level
Usage: python -I src/experiments/check_datasets.py --data data/ --out results/
"""
import argparse, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from decoder import N_JOBS  # noqa: E402
from decoder.datasets import get_loader, load_benchmark  # noqa: E402
from decoder.cca import cca_scores  # noqa: E402
from decoder.filters import bandpass, notch  # noqa: E402


def spec(x, fs, nfft):
    x = x - x.mean(-1, keepdims=True)
    return np.abs(np.fft.rfft(x * np.hanning(x.shape[-1]), n=nfft, axis=-1)) ** 2, np.fft.rfftfreq(nfft, 1 / fs)


def snr_at(p, f, fr, bw=1.0):
    i = np.argmin(np.abs(f - fr))
    nb = (np.abs(f - fr) > 0.25) & (np.abs(f - fr) < bw)
    return p[..., i] / p[..., nb].mean(-1)


def freq_check(dkey, root, s):
    """Label check: CCA accuracy at the longest clean window using the labelled frequencies,
    vs. a control where every label is shifted to the neighbouring target (should collapse)."""
    d = get_loader(dkey)[0](root, s)
    if d.cond is not None:
        d = d.select(d.cond == "wet")
    stim = (d.X.shape[-1] - d.onset) / d.fs - 0.5  # leave a margin after offset
    W = min(4.0, np.floor(stim * 4) / 4)
    x = bandpass(d.window(W, 0.14), d.fs, 7.0, 45.0)
    if dkey == "d4":
        x = notch(x, d.fs, 50.0)
    sc = cca_scores(x, d.freqs, d.fs, n_harm=3)
    acc = float((sc.argmax(1) == d.y).mean())
    acc_shift = float((sc.argmax(1) == np.roll(np.arange(len(d.freqs)), 1)[d.y]).mean())
    pe, fe = spec(d.X, d.fs, int(d.fs * 4))
    pe = pe.mean((0, 1))
    line = {str(hz): float(snr_at(pe, fe, hz, bw=4.0)) for hz in (50, 60) if hz < d.fs / 2}
    return dict(dataset=dkey, subject=s, fs=d.fs, onset=d.onset, X_shape=list(d.X.shape), n_blocks=int(len(np.unique(d.block))),
                check_window_s=W, cca_acc_true_labels=acc, cca_acc_shifted_labels=acc_shift, line_noise_ratio=line,
                ch_names=d.ch_names)


def channel_check(root, s):
    d = load_benchmark(root, s, channels=None)
    w = d.window(2.0, 0.14)
    p, f = spec(w, d.fs, int(d.fs * 20))
    snr = np.stack([snr_at(p[d.y == k].mean(0), f, fr) for k, fr in enumerate(d.freqs)])  # (K, C)
    return snr.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s) for k, subs in [("d1", range(1, 11)), ("d2", [1, 2, 3, 4, 5]), ("d3", [1, 2, 20, 21, 50]),
                                    ("d4", [1, 2, 3, 50, 102])] for s in subs]
    res = Parallel(n_jobs=N_JOBS)(delayed(freq_check)(k, a.data, s) for k, s in jobs)
    for r in res:
        print(r["dataset"], r["subject"], r["X_shape"], "fs", r["fs"], "onset", r["onset"], "blocks", r["n_blocks"],
              "W", r["check_window_s"], "CCA acc true %.2f / shifted %.2f" % (r["cca_acc_true_labels"], r["cca_acc_shifted_labels"]), "line", {k: round(v, 1) for k, v in r["line_noise_ratio"].items()})
    snr = np.mean(Parallel(n_jobs=N_JOBS)(delayed(channel_check)(a.data, s) for s in range(1, 9)), 0)
    from decoder.datasets import TSINGHUA64
    top = [TSINGHUA64[i] for i in np.argsort(snr)[::-1][:12]]
    print("D2 channels with strongest SSVEP SNR (should be occipital):", top)
    json.dump({"per_subject": res, "d2_top_snr_channels": top,
               "d2_channel_snr": dict(zip(TSINGHUA64, map(float, snr)))}, open(out / "dataset_checks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
