"""E2: how many channels? Exhaustive search over channel subsets (D1: 8 ch, D2: 9 occipital ch).

For each k, the best k-channel subset is picked on the tuning half of subjects and scored on the
other half (and vice versa), so every subject is scored with a subset chosen without them.
Usage: python -I src/experiments/e2_channels.py --data data/ --out results/ --figs figures/
"""
import argparse, itertools, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from decoder import N_JOBS  # noqa: E402
from decoder.cca import cca_corr, orth_basis, ref_signals  # noqa: E402
from decoder.config import N_HARM, subject_halves  # noqa: E402
from decoder.datasets import get_loader  # noqa: E402
from decoder.pipeline import default_fb, fb_w, load_prepped, summarize_by  # noqa: E402
from decoder.trca import TRCAStats  # noqa: E402

WINDOWS = [1.0, 2.0]


def run_subject(dkey, root, s):
    d = load_prepped(dkey, root, s)
    C = len(d.ch_names)
    w = fb_w()
    fb = default_fb(d, WINDOWS)
    rows = []
    for W in WINDOWS:
        x = fb[W]
        qy = orth_basis(ref_signals(d.freqs, x.shape[-1], d.fs, N_HARM))
        stats = TRCAStats(x, d.y, d.block)
        for k in range(1, C + 1):
            for sub in itertools.combinations(range(C), k):
                xs = x[:, :, list(sub)]
                rho = np.stack([cca_corr(orth_basis(xb), qy) for xb in xs], -1)
                a_fb = float(((rho ** 2) @ w).argmax(1).__eq__(d.y).mean())
                a_tr = float((stats.lobo(w, ch=list(sub)).argmax(1) == d.y).mean())
                key = "+".join(d.ch_names[i] for i in sub)
                rows.append(dict(subject=s, window=W, k=k, subset=key, method="fbcca", acc=a_fb))
                rows.append(dict(subject=s, window=W, k=k, subset=key, method="trca", acc=a_tr))
    return rows


def analyse(dkey, df, out):
    subjects = sorted(df.subject.unique())
    halves = subject_halves(subjects)
    n_targets = 12 if dkey == "d1" else 40
    held, picks = [], []
    for fold, (tune, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        m = df[df.subject.isin(tune)].groupby(["method", "window", "k", "subset"]).acc.mean().reset_index()
        best = m.loc[m.groupby(["method", "window", "k"]).acc.idxmax()]
        for _, b in best.iterrows():
            picks.append(dict(fold=fold, method=b.method, window=b.window, k=b.k, subset=b.subset, tune_acc=b.acc))
            h = df[(df.method == b.method) & (df.window == b.window) & (df.subset == b.subset) & df.subject.isin(test)]
            held.append(h.assign(fold=fold))
    held = pd.concat(held)
    held["N"] = n_targets
    summ = summarize_by(held, ["method", "window", "k"])
    picks = pd.DataFrame(picks)
    # subset picked on ALL subjects: used only to name the recommended channels
    m_all = df.groupby(["method", "window", "k", "subset"]).acc.mean().reset_index()
    best_all = m_all.loc[m_all.groupby(["method", "window", "k"]).acc.idxmax()].rename(columns={"acc": "acc_all_subjects_optimistic"})
    # mean accuracy of an average (random) subset of size k, for context
    avg_k = df.groupby(["method", "window", "k", "subject"]).acc.mean().reset_index().groupby(["method", "window", "k"]).acc.mean()
    summ["avg_random_subset"] = [avg_k.loc[(r.method, r.window, r.k)] for r in summ.itertuples()]
    held.to_csv(out / f"e2_heldout_{dkey}.csv", index=False)
    summ.to_csv(out / f"e2_summary_{dkey}.csv", index=False)
    picks.to_csv(out / f"e2_picks_{dkey}.csv", index=False)
    best_all.to_csv(out / f"e2_best_subsets_all_{dkey}.csv", index=False)
    # recommendation: smallest k within 2 and 3 points of the full set (held-out means)
    rec = {}
    kmax = int(summ.k.max())
    for (meth, W), g in summ.groupby(["method", "window"]):
        full = float(g[g.k == kmax]["mean"].iloc[0])
        r = {"full_set_acc": full}
        for tol in (0.02, 0.03):
            ok = g[g["mean"] >= full - tol].sort_values("k")
            r[f"smallest_k_within_{int(tol * 100)}pts"] = int(ok.k.iloc[0])
        rec[f"{meth}@{W}s"] = r
    json.dump({"recommendation": rec,
               "best_4_5_6": {f"{r.method}@{r.window}s k={r.k}": {"all_subjects": r.subset,
                              "fold0": picks[(picks.fold == 0) & (picks.method == r.method) & (picks.window == r.window) & (picks.k == r.k)].subset.iloc[0],
                              "fold1": picks[(picks.fold == 1) & (picks.method == r.method) & (picks.window == r.window) & (picks.k == r.k)].subset.iloc[0]}
                              for r in best_all[best_all.k.isin([1, 2, 3, 4, 5, 6])].itertuples()}},
              open(out / f"e2_channels_{dkey}.json", "w"), indent=1)
    return summ, rec


def plot(summs, figdir):
    from decoder.plotstyle import setup, line_with_label
    import matplotlib.pyplot as plt
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    labels = {("fbcca", 1.0): "FBCCA, 1 s (no calibration)", ("fbcca", 2.0): "FBCCA, 2 s (no calibration)",
              ("trca", 1.0): "TRCA, 1 s (calibrated)", ("trca", 2.0): "TRCA, 2 s (calibrated)"}
    for ax, (dkey, summ) in zip(axes, summs.items()):
        for i, (key, lab) in enumerate(labels.items()):
            g = summ[(summ.method == key[0]) & (summ.window == key[1])].sort_values("k")
            line_with_label(ax, list(g.k), list(100 * g["mean"]), lab, i, direct=False)
        ax.set_title({"d1": "D1 Nakanishi (12 targets)", "d2": "D2 Benchmark (40 targets)"}[dkey] + " - held-out subjects")
        ax.set_xlabel("number of channels (best subset)")
        ax.legend(loc="lower right", fontsize=8.5)
    axes[0].set_ylabel("accuracy (%)")
    fig.tight_layout()
    fig.savefig(figdir / "e2_accuracy_vs_channels.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--figs", default="figures/")
    a = ap.parse_args()
    out, figdir = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    summs = {}
    for dkey in ["d1", "d2"]:
        subjects = get_loader(dkey)[1]
        rows = sum(Parallel(n_jobs=N_JOBS)(delayed(run_subject)(dkey, a.data, s) for s in subjects), [])
        df = pd.DataFrame(rows)
        df.to_csv(out / f"e2_all_subsets_{dkey}.csv.gz", index=False)
        summ, rec = analyse(dkey, df, out)
        summs[dkey] = summ
        print(f"== {dkey}")
        print(summ[["method", "window", "k", "mean", "std", "pct_ge90", "avg_random_subset", "worst3"]].round(3).to_string(index=False))
        print(json.dumps(rec, indent=1))
    plot(summs, figdir)


if __name__ == "__main__":
    main()
