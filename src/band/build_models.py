"""Decoder models for the demo menu, from the Wearable dataset (D4, wet electrodes). CPU only.

Menu: 6 D4 frequencies 9.25, 10.25, 11.25, 12.25, 13.25, 14.25 Hz (JFPM phases from the dataset).
Channels: the 6 design channels Oz, POz, O1, O2, PO5, PO6 (experiment E2). Default offline pipeline (50 Hz notch,
filter bank 6/14/22/30/38-88 Hz, 5 harmonics, latency 0.14 s), windows 0.5 to 2.0 s in 0.25 s steps.

Outputs (models/decoder/):
  d4wet_menu6.npz : template SUMS over all 102 subjects (per window, sub-band, target), the own class means of the
                    replay subjects (so their templates are leave-subject-out), thresholds per fold.
  d4wet_menu6.json: offline check of this menu (accuracy at 1 s / 2 s, dynamic + idle on held-out halves) and the
                    replay subjects (chosen at random with a fixed seed, not cherry-picked).
Run:  python -I src/band/build_models.py --set A   (and --set B)
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from decoder import N_JOBS  # noqa: E402
from decoder.cca import cca_corr, orth_basis, ref_signals  # noqa: E402
from decoder.config import FB, LATENCY, N_HARM, subject_halves  # noqa: E402
from decoder.datasets import NAKANISHI_FREQS, NAKANISHI_PHASES, WEAR_SUBJECTS  # noqa: E402
from decoder.metrics import confidence, pooled_heldout  # noqa: E402
from decoder.pipeline import default_fb, fb_w, load_prepped  # noqa: E402
from decoder.transfer import template_scores  # noqa: E402

SETS = {"A": [9.25, 10.25, 11.25, 12.25, 13.25, 14.25], "B": [9.75, 10.75, 11.75, 12.75, 13.75, 14.75]}
MENU_HZ = SETS["A"]  # set B is swapped in when a screen stays inconclusive (band/tune_swap.py)
MENU = [int(np.flatnonzero(NAKANISHI_FREQS == f)[0]) for f in MENU_HZ]


def menu_index(menu_hz):
    return [int(np.flatnonzero(NAKANISHI_FREQS == f)[0]) for f in menu_hz]


def model_name(set_id):
    return "d4wet_menu6" if set_id == "A" else f"d4wet_menu6_{set_id}"
CH6 = ["OZ", "POZ", "O1", "O2", "PO5", "PO6"]
WINDOWS = [round(0.5 + 0.25 * i, 2) for i in range(7)]  # 0.5 .. 2.0 s
N_REPLAY = 5
SEED = 0


def _e7():
    spec = importlib.util.spec_from_file_location("e7", ROOT / "src" / "experiments" / "e7_dynamic_stopping.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_wet(root, s):
    d = load_prepped("d4", root, s)
    return d.select(d.cond == "wet").pick_channels(CH6)


def means_job(root, s, menu=None):
    menu = menu or MENU
    d = load_wet(root, s)
    xw = default_fb(d, WINDOWS)
    out = {}
    for w in WINDOWS:
        x = xw[w] - xw[w].mean(-1, keepdims=True)
        out[w] = np.stack([x[:, d.y == k].mean(1) for k in menu], 1)  # (B, 6, C, n)
    return s, out


def scores_job(root, s, totals, own, n, menu=None):
    menu = menu or MENU
    d = load_wet(root, s)
    xw = default_fb(d, WINDOWS)
    w8 = fb_w()
    fs = d.fs
    tm_sc, fb_sc = [], []
    for w in WINDOWS:
        x = xw[w]
        qy = orth_basis(ref_signals(NAKANISHI_FREQS[menu], x.shape[-1], fs, N_HARM))
        tm = (totals[w] - own[w]) / (n - 1)
        tm_sc.append(template_scores(x, tm, qy, w8))
        fb_sc.append((np.stack([cca_corr(orth_basis(xb), qy) for xb in x], -1) ** 2) @ w8)
    return s, dict(y=d.y, block=d.block, tmpl=np.stack(tm_sc), fbcca=np.stack(fb_sc))  # (J, T, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "models" / "decoder"))
    ap.add_argument("--set", choices=list(SETS), default="A")
    a = ap.parse_args()
    global MENU_HZ, MENU
    MENU_HZ = SETS[a.set]
    MENU = menu_index(MENU_HZ)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subjects = list(WEAR_SUBJECTS)
    own = dict(Parallel(n_jobs=N_JOBS)(delayed(means_job)(a.data, s, MENU) for s in subjects))
    totals = {w: sum(own[s][w] for s in subjects) for w in WINDOWS}
    res = dict(Parallel(n_jobs=N_JOBS)(delayed(scores_job)(a.data, s, totals, own[s], len(subjects), MENU) for s in subjects))

    e7 = _e7()
    menu_arr = np.array(MENU)
    corr, conf, subj, conf_i, subj_i = {}, {}, [], [], []
    acc = {m: {w: [] for w in (1.0, 2.0)} for m in ("tmpl", "fbcca")}
    for m in ("tmpl", "fbcca"):
        cs, cf, ss, ci, si = [], [], [], [], []
        for s in subjects:
            r = res[s]
            pres, idle = np.isin(r["y"], MENU), ~np.isin(r["y"], MENU)
            sc = r[m]  # (J, T, 6)
            c = menu_arr[sc[:, pres].argmax(-1)] == r["y"][pres][None]
            cs.append(c)
            cf.append(np.moveaxis(confidence(sc[:, pres]), -1, 0))
            ss.append(np.full(pres.sum(), s))
            ci.append(np.moveaxis(confidence(sc[:, idle]), -1, 0))
            si.append(np.full(idle.sum(), s))
            for w in (1.0, 2.0):
                acc[m][w].append(float(c[WINDOWS.index(w)].mean()))
        corr[m], conf[m] = np.concatenate(cs, 1), np.concatenate(cf, 2)
        subj_m, conf_i_m, subj_i_m = np.concatenate(ss), np.concatenate(ci, 2), np.concatenate(si)
        if m == "tmpl":
            subj, conf_i, subj_i = subj_m, conf_i_m, subj_i_m

    # thresholds (transfer templates, margin), 2-fold: tuned on one half, applied to the other
    mi = e7.MEASURES.index("margin")
    halves = subject_halves(subjects)
    folds, held = [], []
    for k, (tune, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        ti, tp = np.isin(subj_i, tune), np.isin(subj, tune)
        thr = e7.idle_threshold(conf_i[mi][:, ti], WINDOWS, fa_per_min=1.0)
        thr_pause = e7.idle_threshold(conf_i[mi][:, ti], WINDOWS, fa_per_min=0.2)
        sp, si_ = np.isin(subj, test), np.isin(subj_i, test)
        h = e7.idle_rule_eval(corr["tmpl"][:, sp], conf["tmpl"][:, :, sp], subj[sp], conf_i[:, :, si_], subj_i[si_],
                              WINDOWS, mi, thr)
        held.append(h)
        folds.append(dict(fold=k, tune_subjects=tune, test_subjects=test, threshold=thr, threshold_pause=thr_pause))
    h = __import__("pandas").concat(held)

    rng = np.random.default_rng(SEED)
    replay = sorted(int(s) for s in rng.choice(subjects, N_REPLAY, replace=False))
    per = h.set_index("subject")
    summary = dict(
        menu_hz=MENU_HZ, menu_index=MENU, phases_rad=[float(NAKANISHI_PHASES[k]) for k in MENU], channels=CH6,
        windows=WINDOWS, latency=LATENCY, filter_bank=FB, n_harm=N_HARM, measure="margin",
        accuracy_fixed={m: {str(w): pooled_heldout(np.array(v), np.array(subjects)) for w, v in d.items()}
                        for m, d in acc.items()},
        dynamic_idle_heldout=dict(true_pick_rate=float(h.true_pick_rate.mean()), precision=float(h.precision.mean()),
                                  fa_per_min=float(h.fa_per_min.mean()),
                                  sec_per_selection_median=float(h.sec_per_selection.median()),
                                  itr=float(h.itr.mean())),
        folds=[{k: v for k, v in f.items() if k not in ("tune_subjects", "test_subjects")} for f in folds],
        replay_subjects={str(s): dict(
            fold_threshold=[f for f in folds if s in f["test_subjects"]][0]["threshold"],
            fold_threshold_pause=[f for f in folds if s in f["test_subjects"]][0]["threshold_pause"],
            offline_acc_tmpl_1s=acc["tmpl"][1.0][subjects.index(s)], offline_acc_tmpl_2s=acc["tmpl"][2.0][subjects.index(s)],
            offline_true_pick_rate=float(per.loc[s].true_pick_rate), offline_fa_per_min=float(per.loc[s].fa_per_min))
            for s in replay},
        replay_seed=SEED)
    np.savez_compressed(out / f"{model_name(a.set)}.npz",
                        **{f"total_{w}": totals[w] for w in WINDOWS},
                        **{f"own_{s}_{w}": own[s][w] for s in replay for w in WINDOWS},
                        n_subjects=len(subjects), windows=np.array(WINDOWS), menu_hz=np.array(MENU_HZ),
                        menu_index=np.array(MENU), phases=NAKANISHI_PHASES[MENU], channels=np.array(CH6),
                        replay_subjects=np.array(replay))
    summary["set"] = a.set
    json.dump(summary, open(out / f"{model_name(a.set)}.json", "w"), indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ("accuracy_fixed", "dynamic_idle_heldout", "folds", "replay_subjects")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
