"""E1: reproduce the earlier CCA/FBCCA/TRCA table on D1, diagnose FBCCA, tune FBCCA (D1 then D2).

Usage: python -I src/experiments/e1_fbcca_fix.py --data data/ --out results/ [--part repro|tune|all] [--dataset d1|d2|both]
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from decoder import N_JOBS, SEED  # noqa: E402
from decoder.cca import cca_corr, orth_basis, ref_signals  # noqa: E402
from decoder.datasets import get_loader  # noqa: E402
from decoder.fbcca import combine, fb_weights, fbcca_rho  # noqa: E402
from decoder.filters import subband_lows  # noqa: E402
from decoder.metrics import itr_bits_per_min, subject_summary  # noqa: E402
from decoder.pipeline import trca_lobo_scores, windowed_bp, windowed_fb  # noqa: E402

GAZE = 0.5
HI = 88.0


# ------------------------------------------------------------------ Part A: reproduction
def repro_subject(root, s):
    data = get_loader("d1")[0](root, s)
    windows = [0.5, 1.0, 2.0, 3.0]
    lat = 0.135  # what the earlier work used (38 + round(0.135*256) = sample 73)
    rows = []
    for mode in ["epoch", "window"]:
        bp = windowed_bp(data, 7.0, HI, windows, lat, mode)
        fb = windowed_fb(data, subband_lows("8m", 5), HI, windows, lat, mode)
        w = fb_weights(5)
        for W in windows:
            qy = orth_basis(ref_signals(data.freqs, bp[W].shape[-1], data.fs, 5))
            acc_cca = (cca_corr(orth_basis(bp[W]), qy).argmax(1) == data.y).mean()
            rho = fbcca_rho(fb[W], qy)  # (T, K, B)
            acc_fb = (combine(rho, w).argmax(1) == data.y).mean()
            band_acc = [(rho[..., m].argmax(1) == data.y).mean() for m in range(5)]
            acc_tr = (trca_lobo_scores(fb[W], data.y, data.block).argmax(1) == data.y).mean()
            rows.append(dict(subject=s, mode=mode, window=W, cca=acc_cca, fbcca=acc_fb, trca=acc_tr,
                             **{f"band{m + 1}_only": a for m, a in enumerate(band_acc)}))
    return rows


def run_repro(root, out):
    rows = sum(Parallel(n_jobs=N_JOBS)(delayed(repro_subject)(root, s) for s in range(1, 11)), [])
    df = pd.DataFrame(rows)
    df.to_csv(out / "e1_repro_d1_per_subject.csv", index=False)
    summ = df.groupby(["mode", "window"]).mean(numeric_only=True).drop(columns="subject").round(4)
    summ.to_csv(out / "e1_repro_d1_summary.csv")
    print(summ.to_string())
    return df


# ------------------------------------------------------------------ Part C: tuning
SCHEMES = ["8m", "6m", "8m-2", "harm"]
DESIGNS = ["fixed", "chen"]
MAX_B = 7
HARMS = [3, 4, 5]
A_GRID = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0]
B_GRID = [0.0, 0.25, 0.5, 1.0]
TUNE_WINDOWS = [0.5, 0.75, 1.0, 1.5, 2.0]
LAT = 0.14


def rho_subject(dkey, root, s):
    """Cache: rho[(scheme, design, W, nh)] -> (T, K, MAX_B); cca[(W, nh)] -> (T, K); plus labels."""
    data = get_loader(dkey)[0](root, s)  # D2 loader returns the 9 occipital channels by default
    out = {"y": data.y, "rho": {}, "cca": {}}
    fmin = float(data.freqs.min())
    qys = {(W, h): orth_basis(ref_signals(data.freqs, int(round(W * data.fs)), data.fs, h))
           for W in TUNE_WINDOWS for h in HARMS}
    bp = windowed_bp(data, 7.0, HI, TUNE_WINDOWS, LAT, "window")
    for W in TUNE_WINDOWS:
        qx = orth_basis(bp[W])
        for h in HARMS:
            out["cca"][(W, h)] = cca_corr(qx, qys[(W, h)]).astype(np.float32)
    for sch, des in itertools.product(SCHEMES, DESIGNS):
        lows = subband_lows(sch, MAX_B, fmin)
        fb = windowed_fb(data, lows, HI, TUNE_WINDOWS, LAT, "window", {"design": des})
        for W in TUNE_WINDOWS:
            qxs = [orth_basis(xb) for xb in fb[W]]
            for h in HARMS:
                out["rho"][(sch, des, W, h)] = np.stack([cca_corr(q, qys[(W, h)]) for q in qxs], -1).astype(np.float32)
    return out


def evaluate_grid(caches):
    """Per-subject accuracy for every FBCCA config and CCA config. Returns DataFrame."""
    rows = []
    for si, c in enumerate(caches):
        y = c["y"]
        for (W, h), sc in c["cca"].items():
            rows.append(dict(si=si, method="cca", scheme="-", design="-", nb=0, nh=h, a=np.nan, b=np.nan,
                             window=W, acc=(sc.argmax(1) == y).mean()))
        for (sch, des, W, h), rho in c["rho"].items():
            r2 = rho.astype(np.float64) ** 2
            for nb in range(3, MAX_B + 1):
                for a, b in itertools.product(A_GRID, B_GRID):
                    if a == 0 and b > 0:  # equal weights; b is redundant
                        continue
                    w = fb_weights(nb, a, b)
                    acc = ((r2[..., :nb] @ w).argmax(1) == y).mean()
                    rows.append(dict(si=si, method="fbcca", scheme=sch, design=des, nb=nb, nh=h, a=a, b=b,
                                     window=W, acc=acc))
    return pd.DataFrame(rows)


CFG = ["method", "scheme", "design", "nb", "nh", "a", "b"]


def pick_best(df, subj_idx):
    """Best FBCCA and CCA config by mean accuracy over windows on the given subjects."""
    d = df[df.si.isin(subj_idx)].groupby(CFG, dropna=False).acc.mean().reset_index()
    return {m: d[d.method == m].sort_values("acc", ascending=False).iloc[0].to_dict() for m in ["fbcca", "cca"]}


def cfg_mask(df, cfg):
    m = np.ones(len(df), bool)
    for k in CFG:
        v = cfg[k]
        m &= (df[k].isna() if (isinstance(v, float) and np.isnan(v)) else (df[k] == v)).to_numpy()
    return m


def run_tune(dkey, root, out):
    subjects = get_loader(dkey)[1]
    caches = Parallel(n_jobs=N_JOBS)(delayed(rho_subject)(dkey, root, s) for s in subjects)
    df = evaluate_grid(caches)
    df["subject"] = [subjects[i] for i in df.si]
    df.to_csv(out / f"e1_grid_{dkey}.csv.gz", index=False)
    n_targets = caches[0]["cca"][(1.0, 5)].shape[1]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(subjects))
    halves = [np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])]
    default_fb = dict(method="fbcca", scheme="8m", design="fixed", nb=5, nh=5, a=1.25, b=0.25)
    default_cca = dict(method="cca", scheme="-", design="-", nb=0, nh=5, a=np.nan, b=np.nan)
    report = {"dataset": dkey, "n_subjects": len(subjects), "n_targets": n_targets,
              "halves": [[subjects[i] for i in h] for h in halves], "folds": []}
    held = []
    for fold, (tune_idx, test_idx) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        best = pick_best(df, tune_idx)
        fold_rep = {"tuned_on": [subjects[i] for i in tune_idx], "tested_on": [subjects[i] for i in test_idx],
                    "best_fbcca": best["fbcca"], "best_cca": best["cca"]}
        for name, cfg in [("fbcca_tuned", best["fbcca"]), ("cca_tuned", best["cca"]),
                          ("fbcca_default", default_fb), ("cca_default", default_cca)]:
            d = df[cfg_mask(df, cfg) & df.si.isin(test_idx)]
            for _, r in d.iterrows():
                held.append(dict(fold=fold, config=name, subject=r.subject, window=r.window, acc=r.acc))
        report["folds"].append(fold_rep)
    hd = pd.DataFrame(held)
    hd.to_csv(out / f"e1_heldout_{dkey}.csv", index=False)
    # summary: per config & window, mean/std across held-out subjects (each subject appears once)
    summary = {}
    for (cfgname, W), g in hd.groupby(["config", "window"]):
        s = subject_summary(g.acc.values, g.subject.values)
        s["itr_mean"] = float(np.mean(itr_bits_per_min(n_targets, g.acc.values, W + GAZE)))
        summary.setdefault(cfgname, {})[str(W)] = s
    report["heldout_summary"] = summary
    # Final recommended setting: tuned on ALL subjects (for use downstream); held-out numbers above are the honest ones
    report["best_all_subjects"] = pick_best(df, np.arange(len(subjects)))
    json.dump(report, open(out / f"e1_tuning_{dkey}.json", "w"), indent=1, default=float)
    piv = hd.groupby(["config", "window"]).acc.mean().unstack().round(4)
    print(f"\n== {dkey}: held-out accuracy (each subject scored with config tuned on the other half)")
    print(piv.to_string())
    for f in report["folds"]:
        print("fold best fbcca:", {k: f["best_fbcca"][k] for k in CFG[1:] + ["acc"]})
        print("fold best cca  :", {k: f["best_cca"][k] for k in ["nh", "acc"]})
    return report, df


# ------------------------------------------------------------------ Part D: one setting for both datasets + figures
def _cfg_key(df):
    return df[CFG[1:]].astype(str).agg("|".join, axis=1)


def run_cross(out, figdir):
    """Pick ONE FBCCA setting using only the tuning halves of D1 and D2, score it on the held-out halves."""
    rep = {}
    grids, tun = {}, {}
    for d in ["d1", "d2"]:
        g = pd.read_csv(out / f"e1_grid_{d}.csv.gz")
        g = g[g.method == "fbcca"].copy()
        g["key"] = _cfg_key(g)
        grids[d] = g
        tun[d] = json.load(open(out / f"e1_tuning_{d}.json"))
    keys = set(grids["d1"].key) & set(grids["d2"].key)
    # harm scheme depends on the dataset's lowest frequency; fine (it is defined relative to the stimulus set)
    rows = []
    for fold in [0, 1]:
        score = {}
        for d in ["d1", "d2"]:
            tune_subj = tun[d]["folds"][fold]["tuned_on"]
            m = grids[d][grids[d].subject.isin(tune_subj)].groupby("key").acc.mean()
            score[d] = m / m.max()  # relative to the best config on that dataset's tuning half
        joint = (score["d1"].loc[list(keys)] + score["d2"].loc[list(keys)]) / 2
        best_key = joint.idxmax()
        for d in ["d1", "d2"]:
            test_subj = tun[d]["folds"][fold]["tested_on"]
            h = grids[d][(grids[d].key == best_key) & grids[d].subject.isin(test_subj)]
            for _, r in h.iterrows():
                rows.append(dict(dataset=d, fold=fold, config="fbcca_joint", subject=r.subject, window=r.window, acc=r.acc))
        rep[f"fold{fold}_joint_best"] = dict(zip(CFG[1:], best_key.split("|")))
    joint_df = pd.DataFrame(rows)
    # cross-dataset transfer: best config from one dataset's tuning half applied to the other dataset
    for src, dst in [("d1", "d2"), ("d2", "d1")]:
        cfg = tun[src]["folds"][0]["best_fbcca"]
        key = "|".join(str(cfg[k]) for k in CFG[1:])
        h = grids[dst][grids[dst].key == key].groupby("window").acc.mean()
        rep[f"{src}_best_on_{dst}"] = {"config": {k: cfg[k] for k in CFG[1:]}, "acc_by_window": h.round(4).to_dict()}
    all_rows = [joint_df]
    for d in ["d1", "d2"]:
        hd = pd.read_csv(out / f"e1_heldout_{d}.csv")
        hd["dataset"] = d
        all_rows.append(hd)
    comb = pd.concat(all_rows)
    comb.to_csv(out / "e1_heldout_all.csv", index=False)
    tab = comb.groupby(["dataset", "config", "window"]).acc.agg(["mean", "std"]).round(4)
    tab.to_csv(out / "e1_summary_heldout.csv")
    rep["heldout_table"] = {f"{d}|{c}|{w}": v for (d, c, w), v in tab["mean"].items()}
    json.dump(rep, open(out / "e1_cross.json", "w"), indent=1, default=str)
    print(tab["mean"].unstack().to_string())
    print(json.dumps({k: v for k, v in rep.items() if k != "heldout_table"}, indent=1, default=str))
    make_figures(comb, out, figdir)


def make_figures(comb, out, figdir):
    from decoder.plotstyle import setup, line_with_label, SERIES, TEXT2
    import matplotlib.pyplot as plt
    setup()
    names = {"cca_default": "CCA (5 harmonics)", "fbcca_default": "FBCCA, old setting",
             "fbcca_tuned": "FBCCA, tuned per dataset", "fbcca_joint": "FBCCA, one setting for both"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, d, title in zip(axes, ["d1", "d2"], ["D1 Nakanishi (12 targets, 8 ch)", "D2 Benchmark (40 targets, 9 ch)"]):
        for i, c in enumerate(names):
            g = comb[(comb.dataset == d) & (comb.config == c)].groupby("window").acc
            m = g.mean()
            line_with_label(ax, list(m.index), list(m.values * 100), names[c], i, direct=False)
        ax.set_title(title + "  -  held-out subjects")
        ax.set_xlabel("stare window (s)")
        ax.set_xlim(0.4, 2.1)
        ax.legend(loc="lower right", fontsize=8.5)
    axes[0].set_ylabel("accuracy (%)")
    fig.tight_layout()
    fig.savefig(figdir / "e1_fbcca_vs_cca.png")
    plt.close(fig)
    # diagnostic: accuracy of each sub-band alone (D1 reproduction, epoch mode)
    rp = pd.read_csv(out / "e1_repro_d1_per_subject.csv")
    rp = rp[rp["mode"] == "window"].groupby("window").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    edges = ["8-88 Hz", "16-88 Hz", "24-88 Hz", "32-88 Hz", "40-88 Hz"]
    for i in range(5):
        line_with_label(ax, list(rp.index), list(rp[f"band{i + 1}_only"] * 100), f"sub-band {i + 1} ({edges[i]})", i, direct=False)
    ax.axhline(100 / 12, color=TEXT2, lw=1, ls="--")
    ax.annotate("chance (8.3%)", (2.6, 100 / 12), xytext=(0, 4), textcoords="offset points", fontsize=8, color=TEXT2)
    ax.set_xlabel("stare window (s)")
    ax.set_ylabel("accuracy using that sub-band alone (%)")
    ax.set_title("D1: sub-bands 4-5 carry no usable SSVEP")
    ax.set_xlim(0.4, 3.1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "e1_d1_subband_alone.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--part", default="all", choices=["repro", "tune", "cross", "all"])
    ap.add_argument("--figs", default="figures/")
    ap.add_argument("--dataset", default="both", choices=["d1", "d2", "both"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.part in ("repro", "all"):
        run_repro(a.data, out)
    if a.part in ("tune", "all"):
        for d in (["d1", "d2"] if a.dataset == "both" else [a.dataset]):
            run_tune(d, a.data, out)
    if a.part in ("cross", "all"):
        figdir = Path(a.figs)
        figdir.mkdir(parents=True, exist_ok=True)
        run_cross(out, figdir)


if __name__ == "__main__":
    main()
