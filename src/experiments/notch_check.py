"""Does the 50 Hz notch (now part of the default pipeline) change anything?
Same subjects, same 6-option menus, 1 s and 2 s windows, FBCCA / transfer templates / TRCA, with and without
the notch. Paired per-subject difference (notch on - notch off).
D2 and D3 were already notch-filtered by their authors at 50 Hz; D1 was recorded on 60 Hz mains; D4 is raw
(E1-E4 already used the notch on D4, so "off" is the new case there).
Usage: python -I src/experiments/notch_check.py --data data/ --out results/
"""
import argparse, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from decoder.scorecache import LABELS, get_scores, per_subject_accuracy  # noqa: E402

WINDOWS = [1.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--datasets", default="d1,d2,d3,d4")
    a = ap.parse_args()
    out = Path(a.out)
    rows = []
    for dkey in a.datasets.split(","):
        on = get_scores(dkey, a.data, out / "cache")  # default pipeline (notch on), full window grid
        off = get_scores(dkey, a.data, out / "cache", notch_hz=None,
                         windows={lab: WINDOWS for lab in LABELS[dkey]})
        for lab in LABELS[dkey]:
            x = per_subject_accuracy(on[lab], lab)
            x = x[x.window.isin(WINDOWS)]
            y = per_subject_accuracy(off[lab], lab)
            m = x.merge(y, on=["label", "subject", "method", "window"], suffixes=("_on", "_off"))
            rows.append(m)
        print(f"{dkey} done", flush=True)
    df = pd.concat(rows)
    df["diff"] = df.acc_on - df.acc_off
    df.to_csv(out / "notch_check_per_subject.csv", index=False)
    s = df.groupby(["label", "method", "window"]).agg(
        acc_off=("acc_off", "mean"), acc_on=("acc_on", "mean"), diff_mean=("diff", "mean"),
        diff_std=("diff", "std"), max_abs_diff=("diff", lambda v: float(np.max(np.abs(v)))),
        pct_ge90_off=("acc_off", lambda v: float(100 * np.mean(v >= 0.9 - 1e-9))),
        pct_ge90_on=("acc_on", lambda v: float(100 * np.mean(v >= 0.9 - 1e-9))),
        subjects_changed_gt2pts=("diff", lambda v: int(np.sum(np.abs(v) > 0.02)))).reset_index()
    s.to_csv(out / "notch_check.csv", index=False)
    json.dump(s.to_dict("records"), open(out / "notch_check.json", "w"), indent=1)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
