"""Pre-compute the per-trial score cache used by E5-E7 and the notch check (src/scorecache.py).
Each experiment script also builds whatever it needs on its own; this just does it up front.
Usage: python -I src/experiments/build_score_cache.py --data data/ --out results/ [--datasets d1,d2,d3,d4] [--no-notch]
"""
import argparse, os, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder.scorecache import get_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--datasets", default="d1,d2,d3,d4")
    ap.add_argument("--no-notch", action="store_true")
    a = ap.parse_args()
    for dkey in a.datasets.split(","):
        t0 = time.time()
        get_scores(dkey, a.data, Path(a.out) / "cache", notch_hz=None if a.no_notch else "default",
                   log=lambda m: print(m, flush=True))
        print(f"{dkey} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
