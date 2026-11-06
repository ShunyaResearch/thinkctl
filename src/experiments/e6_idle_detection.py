"""E6: "not looking" detection. 6-option menus at 1 s and 2 s windows, on D1, D2 (spec) and D3, D4 wet/dry
(the realistic numbers).

Simulation: a menu of 6 frequencies. Target-present trials = the user looks at one of the 6.
Idle trials = the user looks at a frequency NOT in the menu that is at least 0.5 Hz from every menu
frequency (a stand-in for looking elsewhere; pessimistic, because a real non-menu flicker still drives
an SSVEP). None of the datasets has rest segments as long as a 1 s window (pre-stimulus parts are
<= 0.5 s), so only the stand-in is used.

Decision rule: accept the top option only if confidence >= threshold. Confidence measures on the 6
menu scores s1 >= s2 >= ... >= s_min:  max = s1,  margin = s1 - s2,  ratio = (s1 - s_min) / (s2 - s_min).
False activations per minute (idle, one decision per window W): FA/min = false-pick rate x 60 / W.
Threshold: chosen on tuning half A as the lowest value giving <= 1 FA/min on A, applied to half B;
then swapped; the held-out halves are pooled (every subject scored once).
Cost: accepted picks are more accurate (precision), but some stares are rejected and must be repeated:
expected time per selection = W / accept rate + 0.5 s gaze shift.
Usage: python -I src/experiments/e6_idle_detection.py --data data/ --out results/ --figs figures/
"""
import argparse, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from decoder.config import GAZE, subject_halves  # noqa: E402
from decoder.metrics import confidence, itr_bits_per_min, pooled_heldout, roc_curve  # noqa: E402
from decoder.scorecache import LABELS, get_scores  # noqa: E402

WINDOWS = [1.0, 2.0]
METHODS = ["fbcca", "tmpl", "trca"]
MEASURES = ["max", "margin", "ratio"]
FA_PER_MIN = 1.0
IDLE_MIN_DF = 0.5  # Hz
TITLES = {"d1": "D1 Nakanishi (gel, lab)", "d2": "D2 Benchmark (gel, lab)", "d3": "D3 BETA (gel, outside lab)",
          "d4wet": "D4 Wearable, wet", "d4dry": "D4 Wearable, dry"}
NAMES = {"fbcca": "FBCCA", "tmpl": "transfer templates", "trca": "TRCA (calibrated)"}


def confidences(sc):
    """sc (T, 6) -> pred index (T,), conf (T, 3) for max / margin / ratio (src.metrics.confidence)."""
    return sc.argmax(1), confidence(sc)


def trials(c, method, W):
    """Pool target-present and idle trials over all menus. Returns a DataFrame."""
    mi = list(c["methods"]).index(method)
    j = int(np.argmin(np.abs(c["windows"] - W)))
    freqs, y = c["freqs"], c["y"]
    parts = []
    for u, S in enumerate(c["menus"]):
        fS = freqs[S]
        pool = [k for k in range(len(freqs)) if k not in S and np.min(np.abs(freqs[k] - fS)) >= IDLE_MIN_DF - 1e-9]
        pres, idle = np.isin(y, S), np.isin(y, pool)
        pred, conf = confidences(c["scores"][mi, j, u].astype(float))
        keep = pres | idle
        parts.append(pd.DataFrame(dict(subject=c["subject"][keep], present=pres[keep],
                                       correct=(S[pred] == y)[keep] & pres[keep],
                                       **{m: conf[keep, i] for i, m in enumerate(MEASURES)})))
    return pd.concat(parts, ignore_index=True)


def threshold_for(idle_conf, max_rate):
    """Lowest threshold whose false-pick rate on these idle trials is <= max_rate."""
    v = np.sort(np.asarray(idle_conf))[::-1]
    allowed = int(np.floor(max_rate * len(v) + 1e-9))
    return float(np.nextafter(v[allowed], np.inf)) if allowed < len(v) else -np.inf


def per_subject(t, meas, thr, W):
    rows = []
    for s, g in t.groupby("subject"):
        p, i = g[g.present], g[~g.present]
        acc_mask = p[meas].values >= thr
        fpr = float(np.mean(i[meas].values >= thr))
        acc_rate = float(acc_mask.mean())
        tpr = float(np.mean(acc_mask & p.correct.values))
        prec = float(np.mean(p.correct.values[acc_mask])) if acc_mask.any() else 0.0
        acc0 = float(p.correct.mean())
        t_sel = W / acc_rate + GAZE if acc_rate > 0 else np.inf
        rows.append(dict(subject=int(s), fa_per_min=fpr * 60 / W, true_pick_rate=tpr, accept_rate=acc_rate,
                         precision=prec, acc_no_threshold=acc0, sec_per_selection=t_sel,
                         itr_with_threshold=float(itr_bits_per_min(6, prec, t_sel)) if np.isfinite(t_sel) else 0.0,
                         itr_no_threshold=float(itr_bits_per_min(6, acc0, W + GAZE))))
    return pd.DataFrame(rows)


def evaluate(t, W, subjects, measure=None):
    """2-fold. measure=None -> also choose the measure on the tuning half."""
    halves = subject_halves(subjects)
    max_rate = FA_PER_MIN * W / 60
    held, info = [], []
    for tune, test in [(halves[0], halves[1]), (halves[1], halves[0])]:
        tt = t[t.subject.isin(tune)]
        cands = [measure] if measure else MEASURES
        best = None
        for m in cands:
            thr = threshold_for(tt[~tt.present][m], max_rate)
            tpr = float(np.mean((tt[tt.present][m] >= thr) & tt[tt.present].correct))
            if best is None or tpr > best[2]:
                best = (m, thr, tpr)
        m, thr, _ = best
        info.append(dict(measure=m, threshold=thr))
        held.append(per_subject(t[t.subject.isin(test)], m, thr, W))
    return pd.concat(held, ignore_index=True), info


def summarize(h):
    r = {}
    r["sec_per_selection_median"] = float(h.sec_per_selection.median())
    for col in ["true_pick_rate", "fa_per_min", "accept_rate", "precision", "acc_no_threshold",
                "sec_per_selection", "itr_with_threshold", "itr_no_threshold"]:
        v = h[col].replace(np.inf, np.nan)
        r[col] = float(v.mean())
        r[col + "_std"] = float(v.std(ddof=1))
    r["pct_subjects_fa_le1"] = float(100 * np.mean(h.fa_per_min <= FA_PER_MIN + 1e-9))
    r["pct_subjects_true_pick_ge90"] = float(100 * np.mean(h.true_pick_rate >= 0.9 - 1e-9))
    r["worst3_true_pick"] = pooled_heldout(h.true_pick_rate.values, h.subject.values)["worst3"]
    return r


def plot(curves, chosen, figdir):
    from decoder.plotstyle import SERIES, setup
    import matplotlib.pyplot as plt
    setup()
    labels = list(curves)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels) + 1.5 * (len(labels) == 1), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, labels):
        i = 0
        for m in ["fbcca", "tmpl"]:
            for W, ls in [(1.0, "-"), (2.0, "--")]:
                fpm, tpr = curves[lab][(m, W)]
                ok = fpm > 0
                ax.plot(fpm[ok], 100 * tpr[ok], ls=ls, color=SERIES[0 if m == "fbcca" else 1], lw=2,
                        label=f"{NAMES[m]}, {W:g} s")
                i += 1
        ax.axvline(FA_PER_MIN, color="#888888", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.05, 60)
        ax.set_title(TITLES[lab], fontsize=10)
        ax.set_xlabel("false activations per idle minute")
    axes[0].set_ylabel("true pick rate (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"E6 idle detection, 6 options, confidence = {chosen} (dotted line: 1 false activation / min)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(figdir / "e6_idle_roc.png")
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
    rows, per_subj, report, roc_store = [], [], {}, {}
    for dkey in a.datasets.split(","):
        cache = get_scores(dkey, a.data, out / "cache")
        for lab in LABELS[dkey]:
            c = cache[lab]
            subjects = sorted(np.unique(c["subject"]).tolist())
            for m in METHODS:
                for W in WINDOWS:
                    t = trials(c, m, W)
                    for meas in MEASURES + [None]:
                        h, info = evaluate(t, W, subjects, meas)
                        r = dict(label=lab, method=m, window=W, measure=meas or "auto",
                                 measure_per_fold=[x["measure"] for x in info],
                                 threshold_per_fold=[x["threshold"] for x in info], **summarize(h))
                        pos = np.where(t.present & t.correct, t[info[0]["measure"]], -np.inf)
                        _, tpr, fpr = roc_curve(pos[t.present], t[~t.present][info[0]["measure"]])
                        r["auc_pooled"] = float(np.trapezoid(tpr, fpr))
                        rows.append(r)
                        per_subj.append(h.assign(label=lab, method=m, window=W, measure=meas or "auto"))
                    for meas in MEASURES:
                        pos = np.where(t.present & t.correct, t[meas], -np.inf)
                        _, tpr, fpr = roc_curve(pos[t.present], t[~t.present][meas])
                        roc_store[(lab, m, W, meas)] = (fpr * 60 / W, tpr)
            print(f"{lab} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "e6_summary.csv", index=False)
    pd.concat(per_subj).to_csv(out / "e6_per_subject.csv", index=False)
    auto = df[df.measure == "auto"]
    chosen = pd.Series(sum(auto.measure_per_fold.tolist(), [])).value_counts().idxmax()
    plot({lab: {(m, W): roc_store[(lab, m, W, chosen)] for m in ["fbcca", "tmpl"] for W in WINDOWS}
          for lab in dict.fromkeys(df.label)}, chosen, figdir)
    # ROC points for the figure, as CSV
    pd.DataFrame([dict(label=k[0], method=k[1], window=k[2], measure=k[3], fa_per_min=f, true_pick_rate=t)
                  for k, (fa, tp) in roc_store.items() for f, t in zip(fa[::max(1, len(fa) // 300)],
                                                                       tp[::max(1, len(tp) // 300)])]
                 ).to_csv(out / "e6_roc_points.csv.gz", index=False)
    report = dict(fa_target_per_min=FA_PER_MIN, idle_min_distance_hz=IDLE_MIN_DF, measure_chosen_most=chosen,
                  rows=auto.to_dict("records"))
    json.dump(report, open(out / "e6_idle.json", "w"), indent=1, default=str)
    cols = ["label", "method", "window", "measure", "true_pick_rate", "fa_per_min", "pct_subjects_fa_le1",
            "precision", "acc_no_threshold", "sec_per_selection_median", "itr_with_threshold", "itr_no_threshold", "auc_pooled"]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
