"""E3: options per screen (N) vs stare time (window), FBCCA (no calibration) and TRCA (calibrated).

Menus of N < K targets use evenly spread frequencies (5 random offsets, same for all subjects).
Operating point: (N, window) with mean accuracy >= 90% and best ITR, chosen on the tuning half
of subjects, reported on the other half (both directions, same split as E1).
Usage: python -I src/experiments/e3_options_vs_time.py --data data/ --out results/ --figs figures/
"""
import argparse, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from decoder import N_JOBS, SEED  # noqa: E402
from decoder.cca import cca_corr, orth_basis, ref_signals  # noqa: E402
from decoder.config import GAZE, N_HARM, subject_halves  # noqa: E402
from decoder.datasets import get_loader  # noqa: E402
from decoder.metrics import itr_bits_per_min  # noqa: E402
from decoder.pipeline import default_fb, fb_w, load_prepped, spread_subsets, summarize_by  # noqa: E402
from decoder.trca import TRCAStats  # noqa: E402

WINDOWS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
NS = {"d1": [2, 4, 6, 8, 12], "d2": [2, 4, 6, 8, 12, 16, 20, 40]}


def run_subject(dkey, root, s, menus):
    d = load_prepped(dkey, root, s)
    w = fb_w()
    fb = default_fb(d, WINDOWS)
    rows = []
    for W in WINDOWS:
        x = fb[W]
        qy = orth_basis(ref_signals(d.freqs, x.shape[-1], d.fs, N_HARM))
        fbs = (np.stack([cca_corr(orth_basis(xb), qy) for xb in x], -1) ** 2) @ w  # (T, K)
        st = TRCAStats(x, d.y, d.block)
        fits = {}
        for bi, b in enumerate(st.blocks):
            fits[b] = st.fit([i for i in range(len(st.blocks)) if i != bi])
        for n, subsets in menus.items():
            acc_f, acc_t = [], []
            for S in subsets:
                S = np.array(S)
                m = np.isin(d.y, S)
                acc_f.append((S[fbs[m][:, S].argmax(1)] == d.y[m]).mean())
                correct = []
                for b in st.blocks:
                    te = m & (d.block == b)
                    f, t = fits[b]
                    sc = st.scores(te, f, t, w, cls=S)
                    correct.append(S[sc.argmax(1)] == d.y[te])
                acc_t.append(np.concatenate(correct).mean())
            rows.append(dict(subject=s, window=W, N=n, method="fbcca", acc=float(np.mean(acc_f))))
            rows.append(dict(subject=s, window=W, N=n, method="trca", acc=float(np.mean(acc_t))))
    return rows


RULES = {
    "mean_ge90": "mean accuracy >= 90 % on the tuning half (as specified)",
    "most_ge90": "at least 90 % of tuning-half subjects reach >= 90 % accuracy (stricter)",
}


def operating_point(df, subjects, rule="mean_ge90"):
    """Pick (N, W) per method on the tuning half (best ITR among settings passing `rule`);
    report on the test half. Both directions."""
    halves = subject_halves(subjects)
    out = []
    for fold, (tune, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        for meth in ["fbcca", "trca"]:
            g = df[(df.method == meth) & df.subject.isin(tune)].assign(good=lambda x: x.acc >= 0.9 - 1e-9)
            agg = g.groupby(["N", "window"]).agg(acc=("acc", "mean"), itr=("itr", "mean"),
                                                 good=("good", "mean")).reset_index()
            ok = agg[agg.acc >= 0.90] if rule == "mean_ge90" else agg[agg.good >= 0.90 - 1e-9]
            if ok.empty:
                out.append(dict(fold=fold, method=meth, rule=rule, N=None, window=None))
                continue
            best = ok.loc[ok.itr.idxmax()]
            h = df[(df.method == meth) & (df.N == best.N) & (df.window == best.window) & df.subject.isin(test)]
            out.append(dict(fold=fold, method=meth, rule=rule, N=int(best.N), window=float(best.window),
                            tune_acc=float(best.acc), tune_itr=float(best.itr),
                            test_acc=float(h.acc.mean()), test_std=float(h.acc.std(ddof=1)),
                            test_pct_ge90=float(100 * (h.acc >= 0.9).mean()), test_itr=float(h.itr.mean()),
                            test_subjects=list(map(int, h.subject)), _h=h))
    return out


def pooled(op):
    """Average of the two folds (each subject scored once, by the setting picked on the other half)."""
    from decoder.metrics import pooled_heldout
    res = []
    for (rule, meth) in sorted({(o["rule"], o["method"]) for o in op}):
        fo = [o for o in op if o["rule"] == rule and o["method"] == meth]
        hs = [o["_h"] for o in fo if "_h" in o]
        r = dict(rule=rule, method=meth, picks=[None if o["N"] is None else f"N={o['N']}, {o['window']} s" for o in fo],
                 folds_with_a_pick=len(hs))
        if hs:
            import pandas as pd
            h = pd.concat(hs)
            r.update(pooled_heldout(h.acc.values, h.subject.values, h.itr.values))
        res.append(r)
    return res


def plot(dkey, summ, figdir):
    from decoder.plotstyle import setup
    import matplotlib.pyplot as plt
    setup()
    fig, axes = plt.subplots(2, 2, figsize=(11, 3.2 + 0.45 * len(NS[dkey]) * 2))
    for j, meth in enumerate(["fbcca", "trca"]):
        for i, (col, lab, fmt) in enumerate([("mean", "accuracy (%)", "{:.0f}"), ("itr", "ITR (bits/min)", "{:.0f}")]):
            ax = axes[i, j]
            g = summ[summ.method == meth].pivot(index="N", columns="window", values=col)
            v = g.values * (100 if col == "mean" else 1)
            im = ax.imshow(v, cmap="Blues", aspect="auto", origin="lower",
                           vmin=0, vmax=100 if col == "mean" else np.nanmax(summ.itr))
            ax.set_xticks(range(len(g.columns)), [f"{c:g}" for c in g.columns])
            ax.set_yticks(range(len(g.index)), [str(n) for n in g.index])
            ax.grid(False)
            for a in range(v.shape[0]):
                for b in range(v.shape[1]):
                    ax.text(b, a, fmt.format(v[a, b]), ha="center", va="center", fontsize=8,
                            color="white" if v[a, b] > 0.6 * im.get_clim()[1] else "#0b0b0b")
            ax.set_xlabel("stare window (s)")
            ax.set_ylabel("options per screen (N)")
            ax.set_title(f"{'FBCCA (no calibration)' if meth == 'fbcca' else 'TRCA (calibrated)'}: {lab}")
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle({"d1": "D1 Nakanishi", "d2": "D2 Benchmark"}[dkey] + " - mean over all subjects")
    fig.tight_layout()
    fig.savefig(figdir / f"e3_heatmaps_{dkey}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--figs", default="figures/")
    ap.add_argument("--datasets", default="d1,d2")
    ap.add_argument("--analyse-only", action="store_true", help="reuse results/e3_per_subject_*.csv")
    a = ap.parse_args()
    out, figdir = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    report = {}
    for dkey in a.datasets.split(","):
        loader, subjects = get_loader(dkey)
        freqs = loader(a.data, subjects[0]).freqs
        menus = {n: spread_subsets(freqs, n) for n in NS[dkey]}
        if a.analyse_only:
            df = pd.read_csv(out / f"e3_per_subject_{dkey}.csv")
        else:
            rows = sum(Parallel(n_jobs=N_JOBS)(delayed(run_subject)(dkey, a.data, s, menus) for s in subjects), [])
            df = pd.DataFrame(rows)
            df["itr"] = [float(itr_bits_per_min(int(r.N), r.acc, r.window + GAZE)) for r in df.itertuples()]
            df.to_csv(out / f"e3_per_subject_{dkey}.csv", index=False)
        summ = summarize_by(df, ["method", "N", "window"])
        summ.to_csv(out / f"e3_summary_{dkey}.csv", index=False)
        op = operating_point(df, subjects) + operating_point(df, subjects, "most_ge90")
        two = pooled(op)
        for o in op:
            o.pop("_h", None)
        report[dkey] = {"menus": {str(k): [[float(freqs[i]) for i in S] for S in v] for k, v in menus.items()},
                        "rules": RULES, "operating_point_2fold_average": two, "operating_point_per_fold": op}
        pd.DataFrame(two).to_csv(out / f"e3_operating_point_2fold_{dkey}.csv", index=False)
        plot(dkey, summ, figdir)
        print(f"== {dkey}")
        print(summ[["method", "N", "window", "mean", "std", "pct_ge90", "itr", "worst3"]].round(3).to_string(index=False))
        print(pd.DataFrame(two).round(3).to_string(index=False))
    json.dump(report, open(out / "e3_operating_points.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
