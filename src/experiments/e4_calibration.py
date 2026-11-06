"""E4: how much calibration? 6 options, 1 s window, on D2 (6 blocks) and D3 BETA (4 blocks).

Methods (all linear, CPU):
  fbcca        : no calibration.
  etrca        : ensemble TRCA trained on n blocks of the user's own data (needs n >= 2: with one
                 trial per target the between-trial covariance S is zero, so TRCA is undefined).
  tmpl_user    : template scorer (src/transfer.py) with the user's own templates, n >= 1.
  tmpl_transfer: same scorer, template = (n * user + lam * others) / (n + lam); n = 0 means pure
                 cross-subject templates (average of all OTHER subjects). lam chosen on the tuning half.
Test block is always held out; training blocks are drawn from the remaining ones.
Calibration time = n blocks x 6 targets x (1.0 s stare + 0.14 s latency + 0.5 s cue/gaze shift).
Usage: python -I src/experiments/e4_calibration.py --data data/ --out results/ --figs figures/
"""
import argparse, itertools, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from decoder import N_JOBS, SEED  # noqa: E402
from decoder.cca import cca_corr, orth_basis, ref_signals  # noqa: E402
from decoder.config import GAZE, LATENCY, N_HARM, subject_halves  # noqa: E402
from decoder.datasets import get_loader  # noqa: E402
from decoder.pipeline import default_fb, fb_w, load_prepped, spread_subsets, summarize_by  # noqa: E402
from decoder.transfer import template_scores  # noqa: E402
from decoder.trca import TRCAStats  # noqa: E402

W = 1.0
N_OPT = 6
LAMS = [0.5, 1.0, 2.0, 4.0, 8.0]
MAX_DRAWS = 5


def subject_templates(dkey, root, s):
    d = load_prepped(dkey, root, s)
    st = TRCAStats(default_fb(d, [W])[W], d.y, d.block)
    return st.sums.sum(0) / st.cnt.sum(0)[None, :, None, None]  # (B, K, C, n)


def run_subject(dkey, root, s, transfer_tmpl, menus):
    d = load_prepped(dkey, root, s)
    w = fb_w()
    x = default_fb(d, [W])[W]
    st = TRCAStats(x, d.y, d.block)
    xc = st.x
    qy = orth_basis(ref_signals(d.freqs, x.shape[-1], d.fs, N_HARM))
    fbs = (np.stack([cca_corr(orth_basis(xb), qy) for xb in x], -1) ** 2) @ w
    rng = np.random.default_rng(SEED + s)
    nb = len(st.blocks)
    menu_mask = [np.isin(d.y, S) for S in menus]
    acc = {}  # (method, n, lam) -> list of correct flags

    def add(key, scores_all, te, cls_scores=None):
        for S, mm in zip(menus, menu_mask):
            S = np.asarray(S)
            sel = mm[te]
            sc = (cls_scores(S)[sel] if cls_scores is not None else scores_all[sel][:, S])
            acc.setdefault(key, []).extend(list(S[sc.argmax(1)] == d.y[te][sel]))

    for bi, b in enumerate(st.blocks):
        te = d.block == b
        add(("fbcca", 0, 0.0), fbs[te], te)
        xte = xc[:, te]
        add(("tmpl_transfer", 0, 0.0), template_scores(xte, transfer_tmpl, qy, w), te)
        others = [i for i in range(nb) if i != bi]
        for n in range(1, nb):
            combos = list(itertools.combinations(others, n))
            if len(combos) > MAX_DRAWS:
                combos = [combos[i] for i in rng.choice(len(combos), MAX_DRAWS, replace=False)]
            for tb in combos:
                tb = list(tb)
                user = st.sums[tb].sum(0) / st.cnt[tb].sum(0)[None, :, None, None]
                add(("tmpl_user", n, 0.0), template_scores(xte, user, qy, w), te)
                for lam in LAMS:
                    mix = (n * user + lam * transfer_tmpl) / (n + lam)
                    add(("tmpl_transfer", n, lam), template_scores(xte, mix, qy, w), te)
                if n >= 2:
                    f, t = st.fit(tb)
                    add(("etrca", n, 0.0), None, te, cls_scores=lambda S: st.scores(te, f, t, w, cls=S))
    return [dict(subject=s, method=k[0], n_blocks=k[1], lam=k[2], acc=float(np.mean(v))) for k, v in acc.items()]


def pick_lambda(df, subjects):
    """Choose lam on the tuning half (mean over n >= 1), report the test half; both directions."""
    halves = subject_halves(subjects)
    keep, chosen = [], {}
    tt = df[(df.method == "tmpl_transfer") & (df.n_blocks >= 1)]
    for fold, (tune, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        lam = float(tt[tt.subject.isin(tune)].groupby("lam").acc.mean().idxmax())
        chosen[fold] = lam
        keep.append(tt[(tt.lam == lam) & tt.subject.isin(test)])
    rest = df[~((df.method == "tmpl_transfer") & (df.n_blocks >= 1))]
    return pd.concat([rest] + keep), chosen


def plot(results, figdir):
    from decoder.plotstyle import setup, line_with_label
    import matplotlib.pyplot as plt
    setup()
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    names = {"fbcca": "FBCCA (no calibration)", "etrca": "ensemble TRCA (own data)",
             "tmpl_user": "template, own data only", "tmpl_transfer": "template, other users + own data"}
    for ax, (dkey, summ) in zip(axes, results.items()):
        for i, (m, lab) in enumerate(names.items()):
            g = summ[summ.method == m].sort_values("n_blocks")
            if m == "fbcca":
                ax.axhline(100 * g["mean"].iloc[0], color="#2a78d6", lw=2, label=lab)
                continue
            line_with_label(ax, list(g.minutes), list(100 * g["mean"]), lab, i, direct=False)
        ax.set_title({"d2": "D2 Benchmark", "d3": "D3 BETA (outside the lab)"}[dkey] + ": 6 options, 1 s window")
        ax.set_xlabel("calibration time (minutes)")
    axes[0].set_ylabel("accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(figdir / "e4_accuracy_vs_calibration.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--figs", default="figures/")
    ap.add_argument("--datasets", default="d2,d3")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/e4_summary_*.csv")
    a = ap.parse_args()
    out, figdir = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    trial_s = W + LATENCY + GAZE
    if a.plot_only:
        return plot({d: pd.read_csv(out / f"e4_summary_{d}.csv") for d in a.datasets.split(",")}, figdir)
    results, report = {}, {}
    for dkey in a.datasets.split(","):
        loader, subjects = get_loader(dkey)
        freqs = loader(a.data, subjects[0]).freqs
        menus = spread_subsets(freqs, N_OPT)
        tmpls = Parallel(n_jobs=N_JOBS)(delayed(subject_templates)(dkey, a.data, s) for s in subjects)
        total = np.sum(tmpls, 0)
        rows = sum(Parallel(n_jobs=N_JOBS)(
            delayed(run_subject)(dkey, a.data, s, (total - tmpls[i]) / (len(subjects) - 1), menus)
            for i, s in enumerate(subjects)), [])
        del tmpls
        df = pd.DataFrame(rows)
        df.to_csv(out / f"e4_per_subject_all_lams_{dkey}.csv", index=False)
        df, lam = pick_lambda(df, subjects)
        df["minutes"] = df.n_blocks * N_OPT * trial_s / 60
        df["window"], df["N"] = W, N_OPT  # so the summary also gets ITR
        df.to_csv(out / f"e4_per_subject_{dkey}.csv", index=False)
        summ = summarize_by(df, ["method", "n_blocks", "minutes"])
        summ.to_csv(out / f"e4_summary_{dkey}.csv", index=False)
        # where does the curve flatten? first n where the next block adds < 1 point (mean)
        flat = {}
        for m, g in summ.groupby("method"):
            g = g.sort_values("n_blocks")
            gains = np.diff(g["mean"].values)
            idx = np.where(gains < 0.01)[0]
            flat[m] = None if len(g) < 2 else (int(g.n_blocks.iloc[idx[0]]) if len(idx) else "no flattening within available blocks")
        report[dkey] = {"lambda_chosen_per_fold": lam, "menus_hz": [[float(freqs[i]) for i in S] for S in menus],
                        "flattens_at_blocks": flat, "trial_seconds": trial_s}
        results[dkey] = summ
        print(f"== {dkey}  lam per fold {lam}")
        print(summ[["method", "n_blocks", "minutes", "mean", "std", "pct_ge90", "itr", "worst3"]].round(3).to_string(index=False))
        print("flattens at:", flat)
    json.dump(report, open(out / "e4_calibration.json", "w"), indent=1, default=str)
    plot(results, figdir)


if __name__ == "__main__":
    main()
