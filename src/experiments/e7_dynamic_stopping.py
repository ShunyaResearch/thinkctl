"""E7: dynamic stopping. 6-option menus. Instead of a fixed window, grow the window in 0.25 s steps from
0.5 s (to 3.0 s on D1/D2; to 2.0 s on D3/D4, where the epochs end) and stop as soon as the confidence
(E6 measures: max, margin = s1 - s2, ratio = (s1 - s_min) / (s2 - s_min)) reaches a threshold; at the longest window the
decoder must decide anyway.

Selection (2-fold, pooled): on tuning half A choose (measure, threshold) for dynamic stopping and the
window for the fixed baseline; rule "acc90": best ITR among settings with mean accuracy >= 90 % on A
(if none reaches 90 %, the most accurate); rule "itr": best ITR with no accuracy constraint.
Score on half B; swap; pool. ITR uses T = mean decision time + 0.5 s gaze shift.

Part b, "dynamic + idle" (the rule the real system needs): no forced decision. The attempt selects at the first
window whose confidence >= threshold; if none does by the longest window, nothing is selected (IDLE).
Threshold chosen on half A as the lowest one giving <= 1 false activation per idle minute, using the E6 idle
trials (non-menu frequencies >= 0.5 Hz from the menu). Time per selection = mean attempt time / accept rate
+ 0.5 s (rejected attempts are repeated).
Usage: python -I src/experiments/e7_dynamic_stopping.py --data data/ --out results/ --figs figures/
"""
import argparse, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from decoder.config import GAZE, subject_halves  # noqa: E402
from decoder.metrics import confidence, itr_bits_per_min, pooled_heldout  # noqa: E402
from decoder.scorecache import LABELS, get_scores  # noqa: E402

METHODS = ["fbcca", "tmpl", "trca"]
MEASURES = ["max", "margin", "ratio"]
N_THR = 120
TITLES = {"d1": "D1 Nakanishi (gel, lab)", "d2": "D2 Benchmark (gel, lab)", "d3": "D3 BETA (gel, outside lab)",
          "d4wet": "D4 Wearable, wet", "d4dry": "D4 Wearable, dry"}
NAMES = {"fbcca": "FBCCA", "tmpl": "transfer templates", "trca": "TRCA (calibrated)"}


def present_trials(c, method):
    """Target-present trials pooled over menus: pred (J, n) correct flags, conf (3, J, n), subject (n,)."""
    mi = list(c["methods"]).index(method)
    corr, conf, subj = [], [], []
    for u, S in enumerate(c["menus"]):
        pres = np.isin(c["y"], S)
        sc = c["scores"][mi, :, u][:, pres].astype(float)  # (J, n, 6)
        conf.append(np.moveaxis(confidence(sc), -1, 0))  # (3, J, n)
        corr.append(S[sc.argmax(-1)] == c["y"][pres][None])
        subj.append(c["subject"][pres])
    return np.concatenate(corr, 1), np.concatenate(conf, 2), np.concatenate(subj)


def idle_trials(c, method, min_df=0.5):
    """Idle trials (E6 definition) pooled over menus: conf (3, J, n), subject (n,)."""
    mi = list(c["methods"]).index(method)
    freqs, y = c["freqs"], c["y"]
    conf, subj = [], []
    for u, S in enumerate(c["menus"]):
        pool = [k for k in range(len(freqs)) if k not in S and np.min(np.abs(freqs[k] - freqs[S])) >= min_df - 1e-9]
        idle = np.isin(y, pool)
        conf.append(np.moveaxis(confidence(c["scores"][mi, :, u][:, idle].astype(float)), -1, 0))
        subj.append(c["subject"][idle])
    return np.concatenate(conf, 2), np.concatenate(subj)


def first_cross(conf_m, windows, thr):
    """Index of the first window with conf >= thr, or -1; and the attempt duration."""
    hit = conf_m >= thr
    any_ = hit.any(0)
    j = np.where(any_, hit.argmax(0), -1)
    dur = np.where(any_, np.asarray(windows)[np.maximum(j, 0)], windows[-1])
    return j, dur


def idle_rule_eval(correct, conf_p, subj_p, conf_i, subj_i, windows, mi, thr):
    """Per-subject results of dynamic stopping with an idle threshold (no forced decision)."""
    jp, dp = first_cross(conf_p[mi], windows, thr)
    acc_ok = jp >= 0
    corr = np.where(acc_ok, correct[np.maximum(jp, 0), np.arange(len(jp))], False)
    ji, di = first_cross(conf_i[mi], windows, thr)
    rows = []
    for s in np.unique(subj_p):
        p, q = subj_p == s, subj_i == s
        a = float(acc_ok[p].mean())
        prec = float(corr[p][acc_ok[p]].mean()) if acc_ok[p].any() else 0.0
        t_att = float(dp[p].mean())
        t_sel = t_att / a + GAZE if a > 0 else np.inf
        rows.append(dict(subject=int(s), accept_rate=a, true_pick_rate=float(corr[p].mean()), precision=prec,
                         time_accepted=float(dp[p][acc_ok[p]].mean()) if acc_ok[p].any() else np.nan,
                         sec_per_selection=t_sel,
                         itr=float(itr_bits_per_min(6, prec, t_sel)) if np.isfinite(t_sel) else 0.0,
                         fa_per_min=float((ji[q] >= 0).sum() / (di[q].sum() / 60))))
    return pd.DataFrame(rows)


def idle_threshold(conf_i_m, windows, fa_per_min=1.0, n_grid=400):
    """Lowest threshold with <= fa_per_min false activations per idle minute (pooled)."""
    peak = conf_i_m.max(0)
    for thr in np.unique(np.quantile(peak, np.linspace(0.5, 1, n_grid))):
        j, d = first_cross(conf_i_m, windows, thr)
        if (j >= 0).sum() / (d.sum() / 60) <= fa_per_min:
            return float(thr)
    return float(np.nextafter(peak.max(), np.inf))


def per_subject(correct, times, subj):
    """correct, times (n,) -> per-subject acc, mean time, ITR (DataFrame)."""
    s_u, inv = np.unique(subj, return_inverse=True)
    cnt = np.bincount(inv)
    acc = np.bincount(inv, correct) / cnt
    tm = np.bincount(inv, times) / cnt
    itr = np.array([itr_bits_per_min(6, a, t + GAZE) for a, t in zip(acc, tm)], float)
    return pd.DataFrame(dict(subject=s_u, acc=acc, time=tm, itr=itr))


def dynamic(correct, conf_m, windows, thr):
    """Stop at the first window whose confidence >= thr (else the last)."""
    hit = conf_m >= thr
    j = np.where(hit.any(0), hit.argmax(0), len(windows) - 1)
    n = np.arange(correct.shape[1])
    return correct[j, n].astype(float), np.asarray(windows)[j]


def candidates(correct, conf, windows, subj, sel):
    """All candidate settings evaluated on the subjects in `sel` (bool mask over trials)."""
    out = []
    for j, W in enumerate(windows):
        ps = per_subject(correct[j, sel].astype(float), np.full(sel.sum(), W), subj[sel])
        out.append(dict(kind="fixed", measure=None, thr=None, window=W, acc=ps.acc.mean(), itr=ps.itr.mean(),
                        time=ps.time.mean()))
    for mi, m in enumerate(MEASURES):
        cm = conf[mi][:, sel]
        for thr in np.unique(np.quantile(cm, np.linspace(0, 1, N_THR))):
            a, t = dynamic(correct[:, sel], cm, windows, thr)
            ps = per_subject(a, t, subj[sel])
            out.append(dict(kind="dynamic", measure=m, thr=float(thr), window=None, acc=ps.acc.mean(),
                            itr=ps.itr.mean(), time=ps.time.mean()))
    return pd.DataFrame(out)


def pick(cands, rule):
    if rule == "acc90":
        ok = cands[cands.acc >= 0.90]
        return ok.loc[ok.itr.idxmax()] if len(ok) else cands.loc[cands.acc.idxmax()]
    return cands.loc[cands.itr.idxmax()]


def apply(setting, correct, conf, windows, subj, sel):
    if setting.kind == "fixed":
        j = list(windows).index(setting.window)
        return per_subject(correct[j, sel].astype(float), np.full(sel.sum(), setting.window), subj[sel])
    mi = MEASURES.index(setting.measure)
    a, t = dynamic(correct[:, sel], conf[mi][:, sel], windows, setting.thr)
    return per_subject(a, t, subj[sel])


def plot(curves, figdir):
    from decoder.plotstyle import SERIES, setup
    import matplotlib.pyplot as plt
    setup()
    labels = list(curves)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels) + 1.5 * (len(labels) == 1), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, labels):
        for i, m in enumerate(METHODS):
            fx, fy, dx, dy, meas = curves[lab][m]
            ax.plot(fx, 100 * fy, ls="--", marker="o", ms=5, color=SERIES[i], lw=1.6, label=f"{NAMES[m]}, fixed window")
            ax.plot(dx, 100 * dy, ls="-", color=SERIES[i], lw=2.2, label=f"{NAMES[m]}, dynamic stopping")
        ax.set_title(TITLES[lab], fontsize=10)
        ax.set_xlabel("mean decision time (s)")
        ax.set_ylim(30, 101)
    axes[0].set_ylabel("accuracy (%), 6 options")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("E7: accuracy vs decision time, fixed windows vs dynamic stopping (all subjects)", fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(figdir / "e7_dynamic_stopping.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--figs", default="figures/")
    ap.add_argument("--datasets", default="d1,d2,d3,d4")
    a = ap.parse_args()
    out, figdir = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    rows, per_subj, curves, idle_rows, per_subj_idle = [], [], {}, [], []
    for dkey in a.datasets.split(","):
        cache = get_scores(dkey, a.data, out / "cache")
        for lab in LABELS[dkey]:
            c = cache[lab]
            windows = [float(w) for w in c["windows"]]
            subjects = sorted(np.unique(c["subject"]).tolist())
            halves = subject_halves(subjects)
            curves[lab] = {}
            for m in METHODS:
                correct, conf, subj = present_trials(c, m)
                allc = candidates(correct, conf, windows, subj, np.ones(len(subj), bool))
                for rule in ["acc90", "itr"]:
                    held = {"fixed": [], "dynamic": []}
                    picks = {"fixed": [], "dynamic": []}
                    for tune, test in [(halves[0], halves[1]), (halves[1], halves[0])]:
                        cand = candidates(correct, conf, windows, subj, np.isin(subj, tune))
                        for kind in ["fixed", "dynamic"]:
                            s = pick(cand[cand.kind == kind], rule)
                            picks[kind].append(f"{s.window:g} s" if kind == "fixed" else f"{s.measure} >= {s.thr:.4g}")
                            held[kind].append(apply(s, correct, conf, windows, subj, np.isin(subj, test)))
                    for kind in ["fixed", "dynamic"]:
                        h = pd.concat(held[kind])
                        r = dict(label=lab, method=m, rule=rule, kind=kind, picks_per_fold=picks[kind],
                                 mean_time_s=float(h.time.mean()), time_std=float(h.time.std(ddof=1)))
                        r.update(pooled_heldout(h.acc.values, h.subject.values, h.itr.values))
                        rows.append(r)
                        per_subj.append(h.assign(label=lab, method=m, rule=rule, kind=kind))
                # curves for the figure: fixed windows, and the dynamic sweep with the most-picked measure
                dyn = allc[allc.kind == "dynamic"]
                best_meas = dyn.loc[dyn.itr.idxmax()].measure
                f = allc[allc.kind == "fixed"]
                d = dyn[dyn.measure == best_meas].sort_values("time")
                curves[lab][m] = (f.time.values, f.acc.values, d.time.values, d.acc.values, best_meas)
                # part b: dynamic + idle threshold
                conf_i, subj_i = idle_trials(c, m)
                held, picks = [], []
                for tune, test in [(halves[0], halves[1]), (halves[1], halves[0])]:
                    tp, ti = np.isin(subj, tune), np.isin(subj_i, tune)
                    best = None
                    for mi, meas in enumerate(MEASURES):
                        thr = idle_threshold(conf_i[mi][:, ti], windows)
                        r = idle_rule_eval(correct[:, tp], conf[:, :, tp], subj[tp], conf_i[:, :, ti], subj_i[ti],
                                           windows, mi, thr)
                        if best is None or r.itr.mean() > best[2]:
                            best = (mi, thr, r.itr.mean())
                    mi, thr, _ = best
                    picks.append(f"{MEASURES[mi]} >= {thr:.4g}")
                    sp, si = np.isin(subj, test), np.isin(subj_i, test)
                    held.append(idle_rule_eval(correct[:, sp], conf[:, :, sp], subj[sp], conf_i[:, :, si], subj_i[si],
                                               windows, mi, thr))
                h = pd.concat(held)
                r = dict(label=lab, method=m, picks_per_fold=picks, max_window_s=windows[-1])
                for col in ["true_pick_rate", "accept_rate", "precision", "time_accepted", "sec_per_selection",
                            "itr", "fa_per_min"]:
                    r[col] = float(h[col].replace(np.inf, np.nan).mean())
                r["sec_per_selection_median"] = float(h.sec_per_selection.median())
                r["pct_subjects_fa_le1"] = float(100 * np.mean(h.fa_per_min <= 1 + 1e-9))
                r["pct_subjects_true_pick_ge90"] = float(100 * np.mean(h.true_pick_rate >= 0.9 - 1e-9))
                r["worst3_true_pick"] = pooled_heldout(h.true_pick_rate.values, h.subject.values)["worst3"]
                idle_rows.append(r)
                per_subj_idle.append(h.assign(label=lab, method=m))
            print(f"{lab} done", flush=True)
    idle_df = pd.DataFrame(idle_rows)
    idle_df.to_csv(out / "e7_dynamic_idle.csv", index=False)
    pd.concat(per_subj_idle).to_csv(out / "e7_dynamic_idle_per_subject.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "e7_summary.csv", index=False)
    pd.concat(per_subj).to_csv(out / "e7_per_subject.csv", index=False)
    # dynamic minus fixed, per (label, method, rule)
    cmp_rows = []
    for (lab, m, rule), g in df.groupby(["label", "method", "rule"], sort=False):
        fx, dy = g[g.kind == "fixed"].iloc[0], g[g.kind == "dynamic"].iloc[0]
        cmp_rows.append(dict(label=lab, method=m, rule=rule, fixed_time=fx.mean_time_s, dyn_time=dy.mean_time_s,
                             fixed_acc=fx["mean"], dyn_acc=dy["mean"], fixed_itr=fx.itr, dyn_itr=dy.itr,
                             itr_gain_pct=100 * (dy.itr / fx.itr - 1) if fx.itr > 0 else None))
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(out / "e7_dynamic_vs_fixed.csv", index=False)
    json.dump(dict(rows=df.to_dict("records"), comparison=cmp_rows, dynamic_idle=idle_df.to_dict("records")),
              open(out / "e7_dynamic.json", "w"), indent=1, default=str)
    plot(curves, figdir)
    print(df[["label", "method", "rule", "kind", "picks_per_fold", "mean", "pct_ge90", "mean_time_s", "itr",
              "worst3"]].round(3).to_string(index=False))
    print(cmp_df.round(2).to_string(index=False))
    print(idle_df.drop(columns=["worst3_true_pick"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
