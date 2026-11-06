"""Swap the menu's frequency set when a screen stays inconclusive (offline, CPU, 2-fold), D4 wet, 102 subjects.

Set A = 9.25, 10.25, ..., 14.25 Hz (default); set B = 9.75, 10.75, ..., 14.75 Hz. Button b keeps its meaning; only
its frequency changes (A[b] <-> B[b]). Rule: after K stares in a row on the same screen without a pick, all buttons
switch to the other set (combine chain and counter reset). Each set has its own (T, T_comb) for the combine rule.
Simulation: present = the user keeps looking at button b for up to 8 stares (random trials of the button's current
frequency); idle = stares at a flicker that is not in the active set (complement set), swaps happen there too.
Time per stare = 0.5 s cue + window (2.0 s if no pick). FA per idle minute = false picks / idle window time.
Tuning per fold (joint): (T, T_comb) per set on half A (combine rule, <= 1 FA per idle minute); then for every K in
{2, 3, 4, never} both sets' thresholds are scaled by the smallest factor that keeps <= 1 FA per idle minute with
swapping; the best K on half A is scored on half B; swapped. Output: models/decoder/swap_tuning_joint.json (used by
band/brain.py).
Run: python -I src/band/tune_swap.py
"""
import json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from band.build_models import SETS, WINDOWS, means_job, menu_index, model_name, scores_job  # noqa: E402
from decoder import N_JOBS  # noqa: E402
from decoder.config import subject_halves  # noqa: E402
from decoder.datasets import WEAR_SUBJECTS  # noqa: E402

DATA = str(ROOT / "data")
W = np.array(WINDOWS)
CUE, MAX_STARES, N_IDLE = 0.5, 8, 60
KS = [2, 3, 4, None]


def scores(set_id):
    cache = ROOT / "models" / "decoder" / f"{model_name(set_id)}_scores.npz"
    if cache.exists():
        z = np.load(cache)
        return {int(s): z[f"S_{s}"] for s in z["subjects"]}, {int(s): z[f"y_{s}"] for s in z["subjects"]}
    menu, subjects = menu_index(SETS[set_id]), list(WEAR_SUBJECTS)
    own = dict(Parallel(n_jobs=N_JOBS)(delayed(means_job)(DATA, s, menu) for s in subjects))
    totals = {w: sum(own[s][w] for s in subjects) for w in WINDOWS}
    res = dict(Parallel(n_jobs=N_JOBS)(delayed(scores_job)(DATA, s, totals, own[s], len(subjects), menu) for s in subjects))
    S = {s: res[s]["tmpl"].astype(np.float32) for s in subjects}
    Y = {s: res[s]["y"] for s in subjects}
    np.savez_compressed(cache, subjects=np.array(subjects), **{f"S_{s}": S[s] for s in subjects},
                        **{f"y_{s}": Y[s] for s in subjects})
    return S, Y


def margin(S):
    p = np.sort(S, -1)
    return p[..., -1] - p[..., -2]


def stare(Sj, chain, T, Tc):
    """One stare: Sj (J, 6). Returns (pick or None, window time)."""
    m1 = margin(Sj)
    if chain:
        comb = (np.sum(chain, 0)[None] + Sj) / (len(chain) + 1)
        m2 = margin(comb)
    for j in range(len(W)):
        if m1[j] >= T:
            return int(Sj[j].argmax()), W[j]
        if chain and m2[j] >= Tc:
            return int(comb[j].argmax()), W[j]
    return None, W[-1]


class Draw:
    def __init__(self, y, rng):
        self.y, self.rng, self.left = y, rng, {}

    def trial(self, cls):
        if not self.left.get(cls):
            self.left[cls] = list(self.rng.permutation(np.flatnonzero(self.y == cls)))
        return int(self.left[cls].pop())


def simulate(s, S, Y, th, K, rng, sets=("A", "B")):
    """S[set][s] scores (J, trials, 6) vs that set's templates; Y[s] class labels (same trials for both sets)."""
    menus = {k: menu_index(SETS[k]) for k in sets}
    dr = Draw(Y[s], rng)
    out = dict(correct=0, wrong=0, none=0, n=0, time=0.0, swaps=0, fa=0, idle_s=0.0)
    for b in range(6):
        for rep in range(10):
            cur, chain, miss, t = sets[0], [], 0, 0.0
            res = "none"
            for _ in range(MAX_STARES):
                tr = dr.trial(menus[cur][b])
                pick, w = stare(S[cur][s][:, tr], chain, *th[cur])
                t += CUE + w
                if pick is not None:
                    res = "correct" if pick == b else "wrong"
                    break
                chain, miss = (chain + [S[cur][s][-1, tr]])[-2:], miss + 1
                if K and miss >= K and len(sets) > 1:
                    cur, chain, miss = sets[1] if cur == sets[0] else sets[0], [], 0
                    out["swaps"] += 1
            out[res] += 1
            out["n"] += 1
            out["time"] += t
    cur, chain, miss = sets[0], [], 0
    for _ in range(N_IDLE):
        other = sets[1] if (len(sets) > 1 and cur == sets[0]) else ("B" if cur == "A" else "A")
        cls = menu_index(SETS[other])[rng.integers(6)]  # a flicker that is not in the active set
        tr = dr.trial(cls)
        pick, w = stare(S[cur][s][:, tr], chain, *th[cur])
        out["idle_s"] += w
        if pick is not None:
            out["fa"] += 1
            chain, miss = [], 0
        else:
            chain, miss = (chain + [S[cur][s][-1, tr]])[-2:], miss + 1
            if K and miss >= K and len(sets) > 1:
                cur, chain, miss = sets[1] if cur == sets[0] else sets[0], [], 0
    return out


def pooled(rs):
    n = sum(r["n"] for r in rs)
    return dict(true_pick=sum(r["correct"] for r in rs) / n, wrong=sum(r["wrong"] for r in rs) / n,
                stuck=sum(r["none"] for r in rs) / n, sec_per_target=sum(r["time"] for r in rs) / n,
                fa_per_min=sum(r["fa"] for r in rs) / (sum(r["idle_s"] for r in rs) / 60),
                swaps_per_target=sum(r["swaps"] for r in rs) / n)


def run(subjects, S, Y, th, K, sets=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    return [simulate(s, S, Y, th, K, rng, sets) for s in subjects]


def tune_set(subjects, S, Y, set_id):
    """(T, T_comb) for one set alone (no swapping), max pick rate with <= 1 FA / idle min (combine rule)."""
    best = None
    for T in np.arange(0.80, 1.50, 0.04):
        for Tc in np.arange(0.30, 0.90, 0.02):
            p = pooled(run(subjects, S, Y, {set_id: (T, Tc)}, None, sets=(set_id,)))
            if p["fa_per_min"] <= 1.0 and (best is None or p["true_pick"] > best[1]["true_pick"]):
                best = ((float(T), float(Tc)), p)
    return best


def joint(S, Y, subjects, halves):
    """Fair test: for each K, scale both sets' thresholds by the smallest factor that keeps <= 1 FA / idle min on the
    tuning half (thresholds otherwise from tune_set), pick the best K (incl. never) on that half, score on the other."""
    out, held = [], []
    for f, (tune_s, test_s) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        base = {k: tune_set(tune_s, S, Y, k)[0] for k in SETS}
        cands = []
        for K in KS:
            for a in np.arange(1.0, 1.42, 0.02):
                th = {k: (v[0] * a, v[1] * a) for k, v in base.items()}
                p = pooled(run(tune_s, S, Y, th, K))
                if p["fa_per_min"] <= 1.0:
                    cands.append((p["true_pick"], K, float(a), th, p))
                    break
        best = max(cands, key=lambda c: c[0])
        r = run(test_s, S, Y, best[3], best[1], seed=1)
        held += list(zip(test_s, r))
        out.append(dict(fold=f, K=best[1], scale=best[2], tune=best[4], test=pooled(r),
                        candidates=[dict(K=c[1], scale=c[2], tune_true_pick=c[0], tune_fa=c[4]["fa_per_min"]) for c in cands],
                        thresholds={k: list(v) for k, v in best[3].items()}, test_subjects=test_s))
        print(f"joint fold {f}: " + "; ".join(f"K={c[1]} scale {c[2]:.2f} pick {c[0]:.3f} FA {c[4]['fa_per_min']:.2f}"
                                             for c in cands) + f" -> chosen K={best[1]}", flush=True)
    return out, pooled([r for _, r in held]), held


def main():
    S, Y = {}, None
    for k in SETS:
        S[k], Y = scores(k)
    subjects = sorted(Y)
    folds, held_p, _ = joint(S, Y, subjects, subject_halves(subjects))
    for f in folds:
        f.pop("test_subjects")
    json.dump(dict(folds=folds, heldout=held_p), open(ROOT / "models" / "decoder" / "swap_tuning_joint.json", "w"),
              indent=1, default=float)
    print(json.dumps(held_p, indent=1, default=float))


if __name__ == "__main__":
    main()
