"""E8: streaming decoder (src/stream_decoder.py) on a replay of D2 (Benchmark, 34 subjects, 250 Hz, 9 channels).

Replay ("fake band"): for every subject and block, 12 full 6 s epochs (raw, NOT notch-filtered: the decoder
applies its own causal 50 Hz notch) are concatenated in a seeded random order: the 6 targets of a 6-option
menu (menu 0 of the score cache) + 6 "idle" epochs whose target is NOT in the menu and >= 0.5 Hz from every
menu frequency (E6 definition). A marker at each stimulus onset plays the role of the UI's stimulus_start.
Data is pushed in 0.1 s chunks.

Models never see the test data: transfer templates = average of the OTHER 33 subjects; TRCA = the subject's
other 5 blocks; thresholds (dynamic + idle, async) = tuned on the other half of the subjects (2-fold).

Parts
  a  offline equivalence: sync, fixed 1 s / 2 s, no threshold, FBCCA / templates / TRCA -> % of picks equal to
     the offline score cache (which used a zero-phase notch) and accuracy.
  b  float32 FBCCA vs float64: % same pick, max confidence difference.
  c  sync dynamic + idle threshold (E7 part b rule, thresholds 2-fold) on the stream.
  d  async FBCCA (no markers, sliding window every 0.1 s, refractory after a pick), threshold 2-fold for
     <= 1 false activation per idle minute.
  e  cost: CPU ms per decision (single process), real-time replay latency (window end -> event), peak memory,
     FLOPs per decision and microcontroller feasibility.
Usage: python -I src/experiments/e8_streaming.py --data data/ --out results/ --figs figures/
"""
import argparse, importlib.util, json, os, resource, sys, time, tracemalloc
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from decoder import N_JOBS, SEED  # noqa: E402
from decoder.config import LATENCY, N_HARM, subject_halves  # noqa: E402
from decoder.datasets import get_loader  # noqa: E402
from decoder.metrics import CONF_MEASURES  # noqa: E402
from decoder.pipeline import default_fb, load_prepped  # noqa: E402
from decoder.scorecache import get_scores, subject_means  # noqa: E402
from decoder.stream_decoder import (FakeBand, FBCCAScorer, StreamDecoder, TemplateScorer, TRCAScorer,  # noqa: E402
                                flops_per_decision, notch_flops_per_second)
from decoder.trca import TRCAStats  # noqa: E402

DKEY = "d2"
DYN_W = [round(0.5 + 0.25 * i, 2) for i in range(11)]  # 0.5 .. 3.0 s
FIXED_W = [1.0, 2.0]
STIM_S = 5.0  # D2 stimulation length
N_IDLE = 6
MCU = {  # effective single-precision throughput (conservative: ~25-40 % of FPU peak) and on-chip RAM
    "ESP32-S3 (240 MHz, FPU)": dict(mflops=60.0, ram_kb=512),
    "nRF52840 (64 MHz Cortex-M4F)": dict(mflops=20.0, ram_kb=256),
}


def _e7():
    spec = importlib.util.spec_from_file_location("e7", Path(__file__).with_name("e7_dynamic_stopping.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- replay construction
def build_streams(d, menu, freqs, seed):
    """Per block: continuous raw stream (C, n), markers [(t, trial_index)], trial info list."""
    rng = np.random.default_rng(seed)
    fS = freqs[menu]
    pool = [k for k in range(len(freqs)) if k not in menu and np.min(np.abs(freqs[k] - fS)) >= 0.5 - 1e-9]
    out = {}
    n_ep = d.X.shape[-1]
    for b in np.unique(d.block):
        idx_p = [int(np.flatnonzero((d.block == b) & (d.y == k))[0]) for k in menu]
        idle_k = rng.choice(pool, N_IDLE, replace=False)
        idx_i = [int(np.flatnonzero((d.block == b) & (d.y == k))[0]) for k in idle_k]
        order = rng.permutation(idx_p + idx_i)
        x = np.concatenate([d.X[i] for i in order], -1)
        markers, info = [], []
        for e, i in enumerate(order):
            t0 = (e * n_ep + d.onset) / d.fs
            tgt = menu.index(int(d.y[i])) if int(d.y[i]) in menu else -1
            markers.append((t0, len(info)))
            info.append(dict(block=int(b), trial=int(i), y=int(d.y[i]), target=tgt, onset=t0))
        out[int(b)] = (x, markers, info)
    return out


def run_sync(dec, x, fs, markers, info, realtime=False):
    """Replay with markers. Returns one row per attempt (its final event) + all events."""
    attempts, cur = [], None
    t_wall0 = time.perf_counter()
    lat = []
    for chunk, t_end, mk in FakeBand(x, fs, 0.1, realtime, markers):
        for t, ti in mk:
            dec.stimulus_start(t)
            cur = ti
        for ev in dec.push(chunk):
            if realtime:
                lat.append(time.perf_counter() - t_wall0 - ev["timestamp"])
            if ev["state"] == "SELECT" or ev.get("final", True):
                it = info[cur]
                attempts.append(dict(**it, state=ev["state"], option=ev["option_id"], best=ev["best_option"],
                                     confidence=ev["confidence"], window_s=ev["window_s"]))
    return attempts, lat


def run_async_trace(dec, x, fs, markers, info):
    """threshold=+inf -> every evaluation is IDLE; keep (t_end, conf, best) for offline threshold sweeps."""
    tr = []
    for chunk, _, _ in FakeBand(x, fs, 0.1, False, markers):
        for ev in dec.push(chunk):
            tr.append((ev["timestamp"], ev["confidence"], ev["best_option"]))
    tr = np.array(tr)
    # label each window [t-W, t]: index of the epoch whose stimulation fully contains it, else -1 (transition)
    W = dec.window_s
    on = np.array([i["onset"] for i in info])
    lab = np.full(len(tr), -1)
    for e, o in enumerate(on):
        inside = (tr[:, 0] - W >= o + LATENCY - 1e-9) & (tr[:, 0] <= o + LATENCY + STIM_S + 1e-9)
        lab[inside] = e
    return tr, lab


# ----------------------------------------------------------------------------- per subject work
def tmpl_totals(root, subjects, windows):
    tot = None
    for m in Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(subject_means)(DKEY, root, s, {DKEY: windows}) for s in subjects):
        m = m[DKEY]
        if tot is None:
            tot = m
        else:
            for w in windows:
                tot[w] += m[w]
    return tot


def build_models(d_off, menu, totals, n_subj, windows):
    """Zero-calibration transfer templates {w: (B, 6, C, n)} and per-block TRCA models
    {block: {w: (filters, templates)}} trained on the other blocks. One window at a time to keep memory low."""
    tmpl, trca = {}, {int(b): {} for b in np.unique(d_off.block)}
    for w in windows:
        x = default_fb(d_off, [w])[w]
        xc = x - x.mean(-1, keepdims=True)
        own = np.stack([xc[:, d_off.y == k].mean(1) for k in range(len(d_off.freqs))], 1)
        tmpl[w] = ((totals[w] - own) / (n_subj - 1))[:, menu]
        del xc
        st = TRCAStats(x, d_off.y, d_off.block)
        del x
        for bi, b in enumerate(st.blocks):
            f, t = st.fit([i for i in range(len(st.blocks)) if i != bi])
            trca[int(b)][w] = (f[:, :, menu], t[:, menu])
        del st
    return tmpl, trca


def subject_job(root, s, menu, totals, n_subj, thr_dyn, seed):
    d_raw = get_loader(DKEY)[0](root, s)  # raw epochs: the stream decoder notches causally itself
    d_off = load_prepped(DKEY, root, s)  # offline pipeline (zero-phase notch) for training models
    freqs = d_raw.freqs
    fm = freqs[menu]
    C = d_raw.X.shape[1]
    fs = d_raw.fs
    allw = sorted(set(DYN_W) | set(FIXED_W))
    tmpl, trca_all = build_models(d_off, list(menu), totals, n_subj, allw)
    del d_off
    streams = build_streams(d_raw, list(menu), freqs, seed + s)
    rows_fixed, rows_dyn, traces = [], [], []
    for b, (x, markers, info) in streams.items():
        trca_models = trca_all[b]

        def scorer(method, dtype=np.float64):
            if method == "fbcca":
                return FBCCAScorer(fm, fs, N_HARM, dtype)
            if method == "tmpl":
                return TemplateScorer(fm, fs, tmpl, N_HARM, dtype)
            return TRCAScorer(fm, fs, trca_models, dtype)

        # a, b: fixed windows, no threshold
        for method in ["fbcca", "tmpl", "trca"]:
            for W in FIXED_W:
                for dt in ([np.float64, np.float32] if method == "fbcca" else [np.float64]):
                    dec = StreamDecoder(scorer(method, dt), fs, C, "sync", window_s=W, dtype=dt)
                    att, _ = run_sync(dec, x, fs, markers, info)
                    rows_fixed += [dict(subject=s, method=method, window=W, dtype=np.dtype(dt).name, **a) for a in att]
        # c: dynamic + idle threshold (tuned on the other half of the subjects)
        for method in ["fbcca", "tmpl", "trca"]:
            meas, thr = thr_dyn[method]
            dec = StreamDecoder(scorer(method), fs, C, "sync", dynamic=True, min_window_s=DYN_W[0],
                                max_window_s=DYN_W[-1], step_s=0.25, threshold=thr, measure=meas)
            att, _ = run_sync(dec, x, fs, markers, info)
            rows_dyn += [dict(subject=s, method=method, measure=meas, threshold=thr, **a) for a in att]
        # d: async FBCCA traces (thresholds applied afterwards)
        for W in FIXED_W:
            for mi, meas in enumerate(CONF_MEASURES):
                dec = StreamDecoder(scorer("fbcca"), fs, C, "async", window_s=W, threshold=np.inf, measure=meas)
                tr, lab = run_async_trace(dec, x, fs, markers, info)
                traces.append(dict(subject=s, block=b, window=W, measure=meas, t=tr[:, 0], conf=tr[:, 1],
                                   best=tr[:, 2].astype(int), epoch=lab,
                                   target=np.array([i["target"] for i in info])))
    return rows_fixed, rows_dyn, traces


# ----------------------------------------------------------------------------- async evaluation
def simulate_async(tr, thr, W, step=0.1):
    """Greedy picks with refractory W after each pick. Returns per-trace counts."""
    conf, best, ep, tgt = tr["conf"], tr["best"], tr["epoch"], tr["target"]
    n_ref = int(round(W / step))
    hits = np.flatnonzero(conf >= thr)
    picks, nxt = [], -1
    for i in hits:
        if i >= nxt:
            picks.append(i)
            nxt = i + n_ref
    picks = np.array(picks, int)
    idle_ev = (ep >= 0) & (tgt[np.maximum(ep, 0)] < 0)
    pres_ev = (ep >= 0) & (tgt[np.maximum(ep, 0)] >= 0)
    fa = int(np.sum(idle_ev[picks])) if len(picks) else 0
    pp = picks[pres_ev[picks]] if len(picks) else picks
    correct = int(np.sum(best[pp] == tgt[ep[pp]])) if len(pp) else 0
    first = []  # time from (onset + latency + W) to the first correct pick, per present epoch
    for e in np.flatnonzero(tgt >= 0):
        ok = pp[(ep[pp] == e) & (best[pp] == tgt[e])] if len(pp) else pp
        first.append(tr["t"][ok[0]] - tr["t"][np.flatnonzero(ep == e)[0]] if len(ok) else np.nan)
    return dict(fa=fa, idle_min=idle_ev.sum() * step / 60, correct=correct, wrong=len(pp) - correct,
                pres_min=pres_ev.sum() * step / 60, n_present=int(np.sum(tgt >= 0)),
                epochs_with_correct=int(np.sum(np.isfinite(first))), first_extra_s=first)


def async_threshold(trs, W, fa_per_min=1.0):
    allc = np.concatenate([t["conf"] for t in trs])
    for thr in np.unique(np.quantile(allc, np.linspace(0.5, 1, 200))):
        r = [simulate_async(t, thr, W) for t in trs]
        if sum(x["fa"] for x in r) / max(sum(x["idle_min"] for x in r), 1e-9) <= fa_per_min:
            return float(thr)
    return float(np.nextafter(allc.max(), np.inf))


def async_eval(traces, subjects):
    halves = subject_halves(subjects)
    rows = []
    for W in FIXED_W:
        best_rows = None
        for tune, test in [(halves[0], halves[1]), (halves[1], halves[0])]:
            best = None
            for meas in CONF_MEASURES:
                trs = [t for t in traces if t["window"] == W and t["measure"] == meas and t["subject"] in tune]
                thr = async_threshold(trs, W)
                r = [simulate_async(t, thr, W) for t in trs]
                score = sum(x["correct"] for x in r) / sum(x["pres_min"] for x in r)
                if best is None or score > best[2]:
                    best = (meas, thr, score)
            meas, thr, _ = best
            for s in test:
                r = [simulate_async(t, thr, W) for t in traces
                     if t["window"] == W and t["measure"] == meas and t["subject"] == s]
                fe = np.concatenate([x["first_extra_s"] for x in r])
                rows.append(dict(window=W, subject=s, measure=meas, threshold=thr,
                                 fa_per_min=sum(x["fa"] for x in r) / sum(x["idle_min"] for x in r),
                                 correct_per_min=sum(x["correct"] for x in r) / sum(x["pres_min"] for x in r),
                                 wrong_per_min=sum(x["wrong"] for x in r) / sum(x["pres_min"] for x in r),
                                 pct_targets_selected=100 * sum(x["epochs_with_correct"] for x in r)
                                 / sum(x["n_present"] for x in r),
                                 median_time_to_first_correct_s=float(np.nanmedian(fe)) + W + LATENCY
                                 if np.isfinite(fe).any() else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- cost
def cost_run(root, s, menu, totals, n_subj, realtime_s):
    """Single-process timing on one subject/block (no other work in this process)."""
    d_raw = get_loader(DKEY)[0](root, s)
    d_off = load_prepped(DKEY, root, s)
    fm, fs, C = d_raw.freqs[menu], d_raw.fs, d_raw.X.shape[1]
    allw = sorted(set(DYN_W) | set(FIXED_W))
    tmpl, trca_all = build_models(d_off, list(menu), totals, n_subj, allw)
    trca = trca_all[0]  # trained on blocks 1-5, tested on block 0
    x, markers, info = build_streams(d_raw.select(d_raw.block == 0), list(menu), d_raw.freqs, SEED)[0]
    mk = {"fbcca": lambda dt: FBCCAScorer(fm, fs, N_HARM, dt), "tmpl": lambda dt: TemplateScorer(fm, fs, tmpl, N_HARM, dt),
          "trca": lambda dt: TRCAScorer(fm, fs, trca, dt)}
    rows = []
    for method in mk:
        for dt in [np.float64, np.float32]:
            for W in FIXED_W:
                timing = []
                sc = mk[method](dt)
                for _ in range(3):  # 3 passes over the block (fresh stream each time): 36 decisions
                    dec = StreamDecoder(sc, fs, C, "sync", window_s=W, dtype=dt, timing=timing)
                    run_sync(dec, x, fs, markers, info)
                tm = np.array(timing[6:])  # drop warm-up
                tracemalloc.start()
                dec2 = StreamDecoder(mk[method](dt), fs, C, "sync", window_s=W, dtype=dt)
                run_sync(dec2, x, fs, markers, info)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append(dict(method=method, dtype=np.dtype(dt).name, window=W, n=len(tm),
                                 wall_ms_median=float(np.median(tm[:, 0])), wall_ms_p95=float(np.percentile(tm[:, 0], 95)),
                                 cpu_ms_median=float(np.median(tm[:, 1])), cpu_ms_p95=float(np.percentile(tm[:, 1], 95)),
                                 python_peak_kb=peak / 1024, decoder_state_kb=dec2.state_bytes() / 1024,
                                 flops=flops_per_decision(method, C, int(round(W * fs)), len(menu)),
                                 flops_6ch=flops_per_decision(method, 6, int(round(W * fs)), len(menu))))
    # real-time replay: latency from window end to emitted event (dynamic mode, worst case = most evaluations)
    lat_rows = []
    n_rt = int(realtime_s * fs)
    m_rt = [(t, i) for t, i in markers if t < realtime_s - 3.5]
    for method in ["fbcca", "trca"]:
        dec = StreamDecoder(mk[method](np.float64), fs, C, "sync", dynamic=True, max_window_s=DYN_W[-1],
                            threshold=np.inf)  # never stops early -> all 11 evaluations per attempt
        _, lat = run_sync(dec, x[:, :n_rt], fs, m_rt, info, realtime=True)
        lat = 1e3 * np.array(lat)
        lat_rows.append(dict(method=method, mode="sync dynamic, real time", n=len(lat),
                             latency_ms_median=float(np.median(lat)), latency_ms_p95=float(np.percentile(lat, 95)),
                             latency_ms_max=float(lat.max()), note="includes waiting for the 0.1 s chunk that "
                             "completes the window (0-100 ms) + compute"))
    return pd.DataFrame(rows), pd.DataFrame(lat_rows), C


def mcu_table(cost, C_design=6, K=6, fs=250.0):
    rows = []
    for method in ["fbcca", "tmpl", "trca"]:
        for W in FIXED_W:
            n = int(round(W * fs))
            fl = flops_per_decision(method, C_design, n, K)
            ring_kb = C_design * int((3.0 + LATENCY + 1.0) * fs) * 4 / 1024  # float32
            work_kb = 5 * C_design * n * 4 / 1024 + 2 * n * C_design * 4 / 1024  # sub-band copies + QR workspace
            model_kb = {"fbcca": K * n * 2 * N_HARM * 4 / 1024, "tmpl": 5 * K * C_design * n * 4 / 1024
                        + K * n * 2 * N_HARM * 4 / 1024, "trca": 5 * K * C_design * n * 4 / 1024 + 5 * C_design * K * 4 / 1024}[method]
            for chip, p in MCU.items():
                ms = fl / (p["mflops"] * 1e6) * 1e3
                ram = ring_kb + work_kb + model_kb
                rows.append(dict(method=method, window=W, channels=C_design, mflop_per_decision=fl / 1e6, chip=chip,
                                 est_ms_per_decision=ms, ram_kb_needed=ram, ram_kb_chip=p["ram_kb"],
                                 notch_mflops_per_s=notch_flops_per_second(C_design, fs) / 1e6,
                                 feasible=bool(ms < 250 and ram < 0.6 * p["ram_kb"])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figure
def plot(trace, thr, W, cost, figdir):
    from decoder.plotstyle import SERIES, setup
    import matplotlib.pyplot as plt
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    on = np.array(trace["onsets"])
    for e, o in enumerate(on):
        col = "#d9e7f5" if trace["target"][e] >= 0 else "#eeeeee"
        ax.axvspan(o, o + STIM_S, color=col, lw=0)
    ax.plot(trace["t"], trace["conf"], color="#555555", lw=1, label=f"confidence ({trace['measure']}, {W:g} s window)")
    ax.axhline(thr, color=SERIES[1], ls="--", lw=1.5, label="threshold (<= 1 false activation / idle min, 2-fold)")
    sty = {"correct": dict(marker="o", color=SERIES[0], mew=1, label="pick: correct"),
           "error": dict(marker="x", color="#c0392b", mew=2.5, label="pick: wrong option or during idle (counted)"),
           "transition": dict(marker="d", color="#999999", mew=1, label="pick: window straddles an onset (not scored)")}
    for kind, st in sty.items():
        idx = [i for i, k in trace["picks"] if k == kind]
        ax.plot(trace["t"][idx], trace["conf"][idx], ls="none", ms=8, **st)
    ax.set_xlabel("stream time (s);  blue = looking at a menu option, grey = looking at a non-menu flicker")
    ax.set_ylabel("confidence")
    ax.set_title("Async FBCCA on a 72 s replay (D2, one subject, one block)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    ax = axes[1]
    c = cost[(cost.dtype == "float64")]
    labels = [f"{m}\n{w:g} s" for m, w in zip(c.method, c.window)]
    ax.bar(range(len(c)), c.cpu_ms_median, color=SERIES[0], width=0.6)
    for i, v in enumerate(c.cpu_ms_median):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("CPU ms per decision (laptop, 1 core)")
    ax.set_title("Compute per decision (9 channels)")
    fig.tight_layout()
    fig.savefig(figdir / "e8_streaming.png")
    plt.close(fig)


def example_figure(root, s, menu, asy, cost, figdir):
    """Async FBCCA trace of subject s, block 0, 1 s window, with its 2-fold threshold; then plot."""
    r0 = asy[(asy.window == 1.0) & (asy.subject == s)].iloc[0]
    d_raw = get_loader(DKEY)[0](root, s)
    x, markers, info = build_streams(d_raw, list(menu), d_raw.freqs, SEED + s)[0]
    dec = StreamDecoder(FBCCAScorer(d_raw.freqs[menu], d_raw.fs, N_HARM), d_raw.fs, x.shape[0], "async",
                        window_s=1.0, threshold=np.inf, measure=r0.measure)
    tr, lab = run_async_trace(dec, x, d_raw.fs, markers, info)
    tgt = np.array([i["target"] for i in info])
    best = tr[:, 2].astype(int)
    picks, nxt = [], -1
    for i in np.flatnonzero(tr[:, 1] >= r0.threshold):
        if i >= nxt:
            e = lab[i]
            kind = "transition" if e < 0 else ("correct" if tgt[e] >= 0 and best[i] == tgt[e] else "error")
            picks.append((i, kind))
            nxt = i + 10
    trace = dict(t=tr[:, 0], conf=tr[:, 1], target=tgt, measure=r0.measure, onsets=[i["onset"] for i in info],
                 picks=picks)
    plot(trace, r0.threshold, 1.0, cost, figdir)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--figs", default="figures/")
    ap.add_argument("--subjects", default="all", help="'all' or comma list (quick test)")
    ap.add_argument("--realtime-s", type=float, default=40.0, help="seconds of real-time replay per method")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from saved results")
    a = ap.parse_args()
    out, figdir = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        c = get_scores(DKEY, a.data, out / "cache")[DKEY]
        subjects = get_loader(DKEY)[1] if a.subjects == "all" else [int(v) for v in a.subjects.split(",")]
        example_figure(a.data, subjects[0], [int(v) for v in c["menus"][0]],
                       pd.read_csv(out / "e8_async_per_subject.csv"), pd.read_csv(out / "e8_cost.csv"), figdir)
        return
    e7 = _e7()
    all_subjects = get_loader(DKEY)[1]
    subjects = all_subjects if a.subjects == "all" else [int(v) for v in a.subjects.split(",")]
    c = get_scores(DKEY, a.data, out / "cache")[DKEY]
    menu = [int(v) for v in c["menus"][0]]
    windows_c = [float(w) for w in c["windows"]]
    halves = subject_halves(all_subjects)

    # thresholds for part c: E7 part-b rule (dynamic + idle, <= 1 FA / idle min), tuned on the other half
    thr_dyn = {}  # (fold test half index) -> method -> (measure, thr)
    for hi_, (tune, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        thr_dyn[hi_] = {}
        for method in ["fbcca", "tmpl", "trca"]:
            correct, conf, subj = e7.present_trials(c, method)
            conf_i, subj_i = e7.idle_trials(c, method)
            tp, ti = np.isin(subj, tune), np.isin(subj_i, tune)
            best = None
            for mi, meas in enumerate(e7.MEASURES):
                thr = e7.idle_threshold(conf_i[mi][:, ti], windows_c)
                r = e7.idle_rule_eval(correct[:, tp], conf[:, :, tp], subj[tp], conf_i[:, :, ti], subj_i[ti],
                                      windows_c, mi, thr)
                if best is None or r.itr.mean() > best[2]:
                    best = (meas, thr, r.itr.mean())
            thr_dyn[hi_][method] = (best[0], best[1])
    print("dynamic+idle thresholds per fold:", thr_dyn, flush=True)

    t0 = time.time()
    totals = tmpl_totals(a.data, all_subjects, sorted(set(DYN_W) | set(FIXED_W)))
    print(f"template totals in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS)(
        delayed(subject_job)(a.data, s, menu, totals, len(all_subjects), thr_dyn[0 if s in halves[1] else 1], SEED)
        for s in subjects)
    print(f"replays in {time.time() - t0:.0f} s", flush=True)
    fixed = pd.DataFrame(sum([r[0] for r in res], []))
    dyn = pd.DataFrame(sum([r[1] for r in res], []))
    traces = sum([r[2] for r in res], [])
    fixed.to_csv(out / "e8_fixed_attempts.csv.gz", index=False)
    dyn.to_csv(out / "e8_dynamic_idle_attempts.csv", index=False)

    # a: offline equivalence
    mi_of = {m: list(c["methods"]).index(m) for m in ["fbcca", "tmpl", "trca"]}
    eq_rows = []
    pres = fixed[fixed.target >= 0].copy()
    for (method, W, dt), g in pres.groupby(["method", "window", "dtype"]):
        j = windows_c.index(W)
        off_pred = []
        for _, r in g.iterrows():
            k = np.flatnonzero((c["subject"] == r.subject) & (c["block"] == r.block) & (c["y"] == r.y))[0]
            off_pred.append(int(np.argmax(c["scores"][mi_of[method], j, 0, k])))
        off_pred = np.array(off_pred)
        acc_s = (g.best.values == g.target.values)
        eq_rows.append(dict(method=method, window=W, dtype=dt, n_trials=len(g),
                            same_pick_as_offline_pct=100 * float(np.mean(off_pred == g.best.values)),
                            stream_acc=float(acc_s.mean()), offline_acc=float(np.mean(off_pred == g.target.values))))
    eq = pd.DataFrame(eq_rows)
    # b: float32 vs float64 (FBCCA), same trials
    f64 = fixed[(fixed.method == "fbcca") & (fixed.dtype == "float64")].reset_index(drop=True)
    f32 = fixed[(fixed.method == "fbcca") & (fixed.dtype == "float32")].reset_index(drop=True)
    f3 = []
    for W in FIXED_W:
        p, q = f64[f64.window == W], f32[f32.window == W]
        f3.append(dict(window=W, n=len(p), same_pick_pct=100 * float(np.mean(p.best.values == q.best.values)),
                       max_abs_conf_diff=float(np.max(np.abs(p.confidence.values - q.confidence.values))),
                       acc64=float(np.mean(p.best.values[p.target.values >= 0] == p.target.values[p.target.values >= 0])),
                       acc32=float(np.mean(q.best.values[q.target.values >= 0] == q.target.values[q.target.values >= 0]))))
    f3 = pd.DataFrame(f3)
    # c: dynamic + idle, per subject then pooled
    dr = []
    for (method, s), g in dyn.groupby(["method", "subject"]):
        p, i = g[g.target >= 0], g[g.target < 0]
        acc_rate = float(np.mean(p.state == "SELECT"))
        ok = (p.state == "SELECT") & (p.option == p.target)
        t_att = float(p.window_s.mean())
        dr.append(dict(method=method, subject=s, accept_rate=acc_rate, true_pick_rate=float(ok.mean()),
                       precision=float(ok.sum() / max((p.state == "SELECT").sum(), 1)),
                       mean_attempt_s=t_att, sec_per_selection=t_att / acc_rate + 0.5 if acc_rate > 0 else np.inf,
                       fa_per_min=float((i.state == "SELECT").sum() / (i.window_s.sum() / 60))))
    dr = pd.DataFrame(dr)
    dsum = dr.groupby("method").agg(true_pick_rate=("true_pick_rate", "mean"), precision=("precision", "mean"),
                                    accept_rate=("accept_rate", "mean"), mean_attempt_s=("mean_attempt_s", "mean"),
                                    sec_per_selection_median=("sec_per_selection", "median"),
                                    fa_per_min=("fa_per_min", "mean"),
                                    pct_subjects_fa_le1=("fa_per_min", lambda v: 100 * float(np.mean(v <= 1 + 1e-9)))
                                    ).reset_index()
    # d: async
    asy = async_eval(traces, subjects) if set(halves[0]) & set(subjects) and set(halves[1]) & set(subjects) else pd.DataFrame()
    asum = asy.groupby(["window", "measure"]).agg(
        threshold=("threshold", "first"), fa_per_min=("fa_per_min", "mean"),
        pct_subjects_fa_le1=("fa_per_min", lambda v: 100 * float(np.mean(v <= 1 + 1e-9))),
        correct_per_min=("correct_per_min", "mean"), wrong_per_min=("wrong_per_min", "mean"),
        pct_targets_selected=("pct_targets_selected", "mean"),
        median_time_to_first_correct_s=("median_time_to_first_correct_s", "median")).reset_index() if len(asy) else asy

    # e: cost (single process, after the parallel part)
    cost, lat, C = cost_run(a.data, subjects[0], menu, totals, len(all_subjects), a.realtime_s)
    mcu = mcu_table(cost)
    for name, df in [("e8_offline_equivalence", eq), ("e8_float32", f3), ("e8_dynamic_idle_per_subject", dr),
                     ("e8_dynamic_idle", dsum), ("e8_async_per_subject", asy), ("e8_async", asum),
                     ("e8_cost", cost), ("e8_latency", lat), ("e8_mcu", mcu)]:
        df.to_csv(out / f"{name}.csv", index=False)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    json.dump(dict(dataset=DKEY, menu_hz=[float(c["freqs"][k]) for k in menu], channels=C, subjects=len(subjects),
                   dynamic_idle_thresholds=thr_dyn, offline_equivalence=eq.to_dict("records"),
                   float32=f3.to_dict("records"), dynamic_idle=dsum.to_dict("records"),
                   async_fbcca=asum.to_dict("records") if len(asum) else [], cost=cost.to_dict("records"),
                   latency=lat.to_dict("records"), mcu=mcu.to_dict("records"), process_max_rss_mb=ru),
              open(out / "e8_streaming.json", "w"), indent=1, default=str)

    if len(asy):
        example_figure(a.data, subjects[0], menu, asy, cost, figdir)

    pd.set_option("display.width", 200)
    for name, df in [("offline equivalence", eq), ("float32", f3), ("dynamic+idle", dsum), ("async", asum),
                     ("cost", cost), ("latency", lat), ("mcu", mcu)]:
        print(f"--- {name}\n{df.round(6 if name == 'float32' else 3).to_string(index=False)}")
    print(f"max RSS of main process {ru:.0f} MB")


if __name__ == "__main__":
    main()
