"""E5: dry vs wet (gel) electrodes on D4 (Zhu 2021 Wearable: 102 subjects, the same 8 sites recorded with
both electrode types, 10 blocks x 12 targets each). 6-option menus, windows 0.5-2.0 s.

Methods: FBCCA (no calibration), transfer templates (zero calibration: templates averaged over the
other 101 subjects with the SAME electrode type), ensemble TRCA (own data, leave-one-block-out).
Outputs: accuracy drop dry vs wet (paired per subject), and how much longer one must stare with dry
electrodes to match wet at 1 s and at 2 s. Part b: does re-tuning FBCCA (bands, harmonics) for dry
electrodes help? Chosen 2-fold (tune on half A -> test on B, swap, pool).
Usage: python -I src/experiments/e5_dry_vs_wet.py --data data/ --out results/ --figs figures/
"""
import argparse, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from decoder import N_JOBS  # noqa: E402
from decoder.cca import cca_corr, orth_basis, ref_signals  # noqa: E402
from decoder.config import FB, GAZE, HI, LATENCY, N_HARM, subject_halves  # noqa: E402
from decoder.datasets import get_loader  # noqa: E402
from decoder.fbcca import fb_weights  # noqa: E402
from decoder.filters import subband_lows  # noqa: E402
from decoder.metrics import itr_bits_per_min, pooled_heldout  # noqa: E402
from decoder.pipeline import load_prepped, spread_subsets, summarize_by, windowed_fb  # noqa: E402
from decoder.scorecache import get_scores, per_subject_accuracy  # noqa: E402

NAMES = {"fbcca": "FBCCA (no calibration)", "tmpl": "transfer templates (zero calibration)",
         "trca": "ensemble TRCA (own 9 blocks)"}
RETUNE_NB = [2, 3, 4, 5]
RETUNE_NH = [2, 3, 4, 5]
RETUNE_W = [1.0, 2.0]


def match_time(win, acc_dry, target):
    """Smallest window where the dry mean-accuracy curve reaches `target` (linear interpolation)."""
    win, acc_dry = np.asarray(win), np.asarray(acc_dry)
    if acc_dry[0] >= target:
        return float(win[0])
    for i in range(1, len(win)):
        if acc_dry[i] >= target:
            a, b = acc_dry[i - 1], acc_dry[i]
            return float(win[i - 1] + (target - a) / (b - a) * (win[i] - win[i - 1]))
    return None


def retune_subject(root, s, menus):
    """FBCCA accuracy (6 options) for every (bands, harmonics) setting, per electrode type and window."""
    d = load_prepped("d4", root, s)
    lows = subband_lows(FB["scheme"], max(RETUNE_NB), float(d.freqs.min()))
    rows = []
    for cond in ["wet", "dry"]:
        dc = d.select(d.cond == cond)
        xw = windowed_fb(dc, lows, HI, RETUNE_W, LATENCY, "window")
        for W in RETUNE_W:
            x = xw[W]
            for nh in RETUNE_NH:
                qy = orth_basis(ref_signals(dc.freqs, x.shape[-1], dc.fs, nh))
                rho = np.stack([cca_corr(orth_basis(xb), qy) for xb in x], -1)  # (T, K, B)
                for nb in RETUNE_NB:
                    sc = (rho[..., :nb] ** 2) @ fb_weights(nb, FB["a"], FB["b"])
                    hit = cnt = 0
                    for S in menus:
                        S = np.asarray(S)
                        pres = np.isin(dc.y, S)
                        hit += (S[sc[pres][:, S].argmax(1)] == dc.y[pres]).sum()
                        cnt += pres.sum()
                    rows.append(dict(subject=s, cond=cond, window=W, n_bands=nb, n_harm=nh, acc=hit / cnt))
    return rows


def retune(root, menus, subjects):
    df = pd.DataFrame(sum(Parallel(n_jobs=N_JOBS)(delayed(retune_subject)(root, s, menus) for s in subjects), []))
    halves = subject_halves(subjects)
    out = []
    for cond in ["wet", "dry"]:
        for W in RETUNE_W:
            g = df[(df.cond == cond) & (df.window == W)]
            held, picks = [], []
            for tune, test in [(halves[0], halves[1]), (halves[1], halves[0])]:
                best = g[g.subject.isin(tune)].groupby(["n_bands", "n_harm"]).acc.mean().idxmax()
                picks.append(f"{best[0]} bands, {best[1]} harmonics")
                held.append(g[g.subject.isin(test) & (g.n_bands == best[0]) & (g.n_harm == best[1])])
            h = pd.concat(held)
            dflt = g[(g.n_bands == FB["n_bands"]) & (g.n_harm == N_HARM)]
            r = dict(cond=cond, window=W, picks_per_fold=picks)
            r.update({f"retuned_{k}": v for k, v in pooled_heldout(h.acc.values, h.subject.values).items()})
            r.update({f"default_{k}": v for k, v in pooled_heldout(dflt.acc.values, dflt.subject.values).items()})
            out.append(r)
    return df, pd.DataFrame(out)


def plot(summ, figdir):
    from decoder.plotstyle import SERIES, setup
    import matplotlib.pyplot as plt
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, col, lab in [(axes[0], "mean", "mean accuracy (%)"), (axes[1], "pct_ge90", "subjects reaching 90 % (%)")]:
        for i, m in enumerate(NAMES):
            for cond, ls, mk in [("wet", "-", "o"), ("dry", "--", "s")]:
                g = summ[(summ.method == m) & (summ.cond == cond)].sort_values("window")
                v = g[col] * (100 if col == "mean" else 1)
                ax.plot(g.window, v, ls=ls, marker=mk, color=SERIES[i], lw=2, ms=6,
                        label=f"{NAMES[m]}, {cond}")
        ax.set_xlabel("stare window (s)")
        ax.set_ylabel(lab)
        ax.set_title(f"D4 Wearable, 6 options: {lab}")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(figdir / "e5_dry_vs_wet.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--figs", default="figures/")
    a = ap.parse_args()
    out, figdir = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    cache = get_scores("d4", a.data, out / "cache")
    df = pd.concat([per_subject_accuracy(cache[f"d4{c}"], c) for c in ["wet", "dry"]]).rename(columns={"label": "cond"})
    df["N"] = 6
    df.to_csv(out / "e5_per_subject.csv", index=False)
    summ = summarize_by(df, ["cond", "method", "window"])
    summ.to_csv(out / "e5_summary.csv", index=False)

    # paired drop wet - dry, per subject
    piv = df.pivot_table(index=["method", "window", "subject"], columns="cond", values="acc").reset_index()
    piv["drop"] = piv.wet - piv.dry
    drop = piv.groupby(["method", "window"]).agg(
        wet=("wet", "mean"), dry=("dry", "mean"), drop_mean=("drop", "mean"), drop_std=("drop", "std"),
        subjects_worse_with_dry=("drop", lambda v: float(100 * np.mean(v > 0)))).reset_index()
    drop.to_csv(out / "e5_drop.csv", index=False)

    match = {}
    for m in NAMES:
        wet = summ[(summ.method == m) & (summ.cond == "wet")].sort_values("window")
        dry = summ[(summ.method == m) & (summ.cond == "dry")].sort_values("window")
        for ref in [1.0, 2.0]:
            target = float(wet[wet.window == ref]["mean"].iloc[0])
            t = match_time(dry.window.values, dry["mean"].values, target)
            match[f"{m} wet@{ref:g}s"] = dict(
                wet_acc=target, dry_window_needed_s=t,
                extra_stare_s=None if t is None else t - ref,
                note=None if t is not None else "dry never reaches it within 2.0 s (epochs end at 2 s)")
    menus = cache["d4wet"]["menus"].tolist()
    subjects = get_loader("d4")[1]
    rt_all, rt = retune(a.data, menus, subjects)
    rt_all.to_csv(out / "e5_fbcca_retune_all.csv", index=False)
    rt.to_csv(out / "e5_fbcca_retune.csv", index=False)
    json.dump({"match_wet": match, "menus_hz": [[float(cache["d4wet"]["freqs"][i]) for i in S] for S in menus],
               "fbcca_retune": rt.to_dict("records")}, open(out / "e5_dry_vs_wet.json", "w"), indent=1, default=str)
    plot(summ, figdir)
    print(summ[summ.window.isin([1.0, 2.0])][["cond", "method", "window", "mean", "std", "pct_ge90", "itr",
                                               "worst3"]].round(3).to_string(index=False))
    print(drop[drop.window.isin([1.0, 2.0])].round(3).to_string(index=False))
    print(json.dumps(match, indent=1))
    print(rt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
