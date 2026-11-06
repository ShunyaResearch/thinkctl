"""E5 side checks on D4 (Wearable).

1. Electrode order: which index of the electrode axis is wet and which is dry? Impedance.mat headband medians,
   and per-recording 50 Hz power, DC offset and std of the raw EEG (subjects 1-20).
2. Mains handling: 6-option accuracy (menu 0, subjects 1-40, 1 s and 2 s) of FBCCA, transfer templates
   (leave-subject-out within these 40) and TRCA (leave-one-block-out) for:
     none        : no 50 Hz removal
     iir         : default notch (iirnotch Q=30, filtfilt on the epoch)
     iir_longpad : same, epoch mean removed and padlen = epoch length (tests the edge-transient idea)
     regress     : least-squares removal of 50 Hz and 100 Hz sines over the epoch
     iir_hi45    : default notch, filter-bank upper edge 45 Hz instead of 88 Hz
     none_hi45   : no notch, upper edge 45 Hz
3. Inter-trial phase locking at Oz (band 6-88 Hz, 0.25 s segments from -0.25 s to 2.0 s; subjects 1-40).
Usage: python -I src/experiments/e5_d4_checks.py --data data/ --out results/
"""
import argparse, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.signal import filtfilt, sosfiltfilt, welch  # noqa: E402

from decoder import N_JOBS  # noqa: E402
from decoder.cca import cca_corr, orth_basis, ref_signals  # noqa: E402
from decoder.config import FB, LATENCY  # noqa: E402
from decoder.datasets import get_loader  # noqa: E402
from decoder.filters import bandpass_sos, notch, notch_ba, subband_lows  # noqa: E402
from decoder.pipeline import default_fb, fb_w, spread_subsets, windowed_fb  # noqa: E402
from decoder.transfer import template_scores  # noqa: E402
from decoder.trca import TRCAStats  # noqa: E402

W = [1.0, 2.0]
SUBJ = list(range(1, 41))
VARS = ["none", "iir", "iir_longpad", "regress", "iir_hi45", "none_hi45"]
SEG = np.arange(-0.25, 2.01, 0.25)


def regress(x, fs, freqs=(50.0, 100.0)):
    t = np.arange(x.shape[-1]) / fs
    a = np.concatenate([np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], 1) for f in freqs], 1)
    return x - x @ (a @ np.linalg.pinv(a)).T


def clean(x, fs, v):
    if v.startswith("none"):
        return x
    if v in ("iir", "iir_hi45"):
        return notch(x, fs)
    if v == "iir_longpad":
        b, a = notch_ba(fs, 50.0, 30.0)
        return filtfilt(b, a, x - x.mean(-1, keepdims=True), axis=-1, padtype="even", padlen=x.shape[-1] - 1)
    if v == "regress":
        return regress(x, fs)
    raise ValueError(v)


def prep(root, s):
    d = get_loader("d4")[0](root, s)
    out = {}
    for v in VARS:
        dv = d.select(np.ones(len(d.y), bool))
        dv.X = clean(d.X, d.fs, v)
        for cond in ["wet", "dry"]:
            dc = dv.select(dv.cond == cond)
            if v.endswith("hi45"):
                xw = windowed_fb(dc, subband_lows(FB["scheme"], FB["n_bands"]), 45.0, W, LATENCY, "window")
            else:
                xw = default_fb(dc, W)
            out[(v, cond)] = (xw, dc.y, dc.block)
    return out, d.freqs


def class_means(xw, y):
    return {w: np.stack([(xw[w] - xw[w].mean(-1, keepdims=True))[:, y == k].mean(1) for k in range(12)], 1) for w in W}


def totals_job(root, s):
    p, _ = prep(root, s)
    return {k: class_means(xw, y) for k, (xw, y, _) in p.items()}


def acc_job(root, s, tot, n):
    p, freqs = prep(root, s)
    menu = list(spread_subsets(freqs, 6)[0])
    m_arr = np.array(menu)
    rows = []
    for (v, cond), (xw, y, blk) in p.items():
        own = class_means(xw, y)
        pres = np.isin(y, menu)
        for w in W:
            x = xw[w]
            qy = orth_basis(ref_signals(freqs, x.shape[-1], 250.0, 5))
            fb = (np.stack([cca_corr(orth_basis(xb), qy) for xb in x], -1) ** 2) @ fb_w()
            tp = template_scores(x, (tot[(v, cond)][w] - own[w]) / (n - 1), qy, fb_w())
            st = TRCAStats(x, y, blk)
            tr = np.zeros((len(y), 6))
            for bi, b in enumerate(st.blocks):
                f, t = st.fit([i for i in range(len(st.blocks)) if i != bi])
                te = blk == b
                tr[te] = st.scores(te, f, t, fb_w(), cls=menu)
            for name, sc in [("fbcca", fb[:, menu]), ("tmpl", tp[:, menu]), ("trca", tr)]:
                rows.append(dict(subject=s, cond=cond, variant=v, method=name, window=w,
                                 acc=float(np.mean(m_arr[sc[pres].argmax(1)] == y[pres]))))
    return rows


def order_job(root, s):
    d = sio.loadmat(Path(root) / "wearable" / f"S{s:03d}.mat")["data"]  # (8, 710, 2, 10, 12)
    r = {}
    for e in (0, 1):
        x = d[:, :, e].reshape(8, 710, -1)
        f, p = welch(x - x.mean(1, keepdims=True), fs=250, nperseg=250, axis=1)
        r[e] = dict(abs_dc_uv=float(np.median(np.abs(x.mean(1)))), power_50hz=float(np.median(p[:, f == 50])),
                    std_uv=float(np.median(x.std(1))))
    return r


def plv_job(root, s):
    d = get_loader("d4")[0](root, s)
    d.X = notch(d.X, d.fs)
    sos = bandpass_sos(250.0, 6.0, 88.0)
    out = {}
    for cond in ["wet", "dry"]:
        dc = d.select(d.cond == cond)
        x = sosfiltfilt(sos, dc.X, axis=-1)[:, dc.ch_names.index("OZ")]
        plv = np.zeros((len(SEG), 12))
        for k, f in enumerate(dc.freqs):
            xs = x[dc.y == k]
            for i, a in enumerate(SEG):
                i0 = dc.onset + int(round(a * 250))
                t = (np.arange(i0, i0 + 62) - dc.onset) / 250
                z = (xs[:, i0:i0 + 62] * np.exp(-2j * np.pi * f * t)).mean(-1)
                plv[i, k] = np.abs((z / np.abs(z)).mean())
        out[cond] = plv.mean(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    a = ap.parse_args()
    out = Path(a.out)
    rep = {}
    # 1. electrode order
    imp = sio.loadmat(Path(a.data) / "wearable" / "Impedance.mat")["Impedance"]  # (8, 10, 2, 102)
    rep["impedance_median_kohm"] = {"headband 1 (Readme: wet)": float(np.median(imp[:, :, 0])),
                                    "headband 2 (Readme: dry)": float(np.median(imp[:, :, 1]))}
    od = Parallel(n_jobs=N_JOBS)(delayed(order_job)(a.data, s) for s in range(1, 21))
    rep["eeg_by_electrode_index"] = {f"index {e}": {k: float(np.median([r[e][k] for r in od])) for k in od[0][e]}
                                     for e in (0, 1)}
    # 2. mains handling
    tot = None
    for m in Parallel(n_jobs=N_JOBS, return_as="generator")(delayed(totals_job)(a.data, s) for s in SUBJ):
        if tot is None:
            tot = m
        else:
            for k in tot:
                for w in W:
                    tot[k][w] += m[k][w]
    df = pd.DataFrame(sum(Parallel(n_jobs=N_JOBS)(delayed(acc_job)(a.data, s, tot, len(SUBJ)) for s in SUBJ), []))
    df.to_csv(out / "e5_d4_mains_per_subject.csv", index=False)
    tab = df.pivot_table(index=["cond", "method", "window"], columns="variant", values="acc")[VARS] * 100
    tab.round(1).to_csv(out / "e5_d4_mains.csv")
    # 3. phase locking over time
    pl = Parallel(n_jobs=N_JOBS)(delayed(plv_job)(a.data, s) for s in SUBJ)
    rep["phase_locking_oz"] = {"segment_start_s": SEG.round(2).tolist(),
                               **{c: np.mean([r[c] for r in pl], 0).round(3).tolist() for c in ["wet", "dry"]}}
    json.dump(rep, open(out / "e5_d4_checks.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(tab.round(1).to_string())


if __name__ == "__main__":
    main()
