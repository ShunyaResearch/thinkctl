"""The 'brain' process: band -> stream decoder -> decisions. Runs on CPU in its own process so decoding never
disturbs the UI's frame timing.

Messages in (cmd_q):  ("attempt", attempt_id, gaze_hz_or_None, pause_mode)  |  ("quit",)
Messages out (ev_q):  ("ready", info) | ("onset", attempt_id, wall_time) | ("decision", attempt_id, event_dict)
Decoder: decoder.stream_decoder.StreamDecoder, sync mode, dynamic stopping 0.5-2.0 s in 0.25 s steps, transfer templates
(leave-subject-out), margin threshold from the 2-fold tuning half that does NOT contain this subject.
Pause mode: only option 'Resume' counts, with the stricter pause threshold (<= 0.2 false activations / min).
Combine attempts (tuned in band/tune_swap.py): an attempt that ends without a pick keeps its 2.0 s score vector; in the
next attempt on the same menu the brain also checks S_comb = mean(last <= 2 kept vectors, current scores) and picks
argmax(S_comb) if its margin >= T_comb. T and T_comb come from the 2-fold tuning half without this subject.
"""
import os, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))


def load_models(subject):
    """Leave-subject-out templates for both frequency sets + thresholds from the fold that did not tune on subject."""
    import json
    import numpy as np
    from decoder.config import subject_halves
    from decoder.datasets import WEAR_SUBJECTS
    swap = json.load(open(ROOT / "models/decoder/swap_tuning_joint.json"))["folds"]
    fold = 0 if subject in subject_halves(list(WEAR_SUBJECTS))[1] else 1  # fold k is tested on the other half
    sets = {}
    for set_id, name in (("A", "d4wet_menu6"), ("B", "d4wet_menu6_B")):
        z = np.load(ROOT / f"models/decoder/{name}.npz")
        info = json.load(open(ROOT / f"models/decoder/{name}.json"))
        n = int(z["n_subjects"])
        windows = [float(w) for w in z["windows"]]
        if f"own_{subject}_{windows[0]}" not in z:
            raise ValueError(f"subject {subject} is not a prepared replay subject: {list(z['replay_subjects'])}")
        T, Tc = swap[fold]["thresholds"][set_id]
        sets[set_id] = dict(templates={w: (z[f"total_{w}"] - z[f"own_{subject}_{w}"]) / (n - 1) for w in windows},
                            menu_hz=info["menu_hz"], phases=info["phases_rad"], channels=info["channels"],
                            T=T, T_comb=Tc, T_pause=info["replay_subjects"][str(subject)]["fold_threshold_pause"])
    return sets, dict(fold=fold, K=swap[fold]["K"])


class CapturingScorer:
    """Wraps a scorer and keeps the last score vector (the stream decoder only reports the confidence)."""

    def __init__(self, inner):
        self.inner, self.last = inner, None
        self.phase_locked = getattr(inner, "phase_locked", True)

    def scores(self, x):
        self.last = self.inner.scores(x)
        return self.last


def _margin(v):
    import numpy as np
    p = np.sort(v)
    return float(p[-1] - p[-2])


def brain_main(cmd_q, ev_q, cfg):
    import numpy as np
    from band.replay import ReplayBand
    from decoder.stream_decoder import StreamDecoder, TemplateScorer
    subject, resume_idx = cfg["subject"], cfg["resume_index"]
    sets, meta = load_models(subject)
    A = sets["A"]
    band = ReplayBand(cfg["data_root"], subject, A["channels"], A["menu_hz"], seed=cfg["seed"])
    scorers = {k: CapturingScorer(TemplateScorer(v["menu_hz"], band.fs, v["templates"])) for k, v in sets.items()}
    dec = StreamDecoder(scorers["A"], band.fs, len(A["channels"]), mode="sync", dynamic=True, min_window_s=0.5,
                        max_window_s=2.0, step_s=0.25, threshold=A["T"], measure="margin", notch_hz=50.0)
    band.start()
    ev_q.put(("ready", dict(subject=subject, fold=meta["fold"], K=meta["K"] if cfg.get("swap", True) else None,
                            sets={k: dict(menu_hz=v["menu_hz"], phases=v["phases"], T=v["T"], T_comb=v["T_comb"],
                                          T_pause=v["T_pause"]) for k, v in sets.items()},
                            threshold=A["T"], threshold_comb=A["T_comb"], threshold_pause=A["T_pause"],
                            channels=A["channels"], t0_wall=band.t0_wall)))
    attempt, pause, menu_key, chain, closed, cur = None, False, None, [], True, "A"
    use_combine = cfg.get("combine", True)
    while True:
        while not cmd_q.empty():
            cmd = cmd_q.get()
            if cmd[0] == "quit":
                band.stop()
                ev_q.put(("band_log", band.log, dec.timing))
                return
            if cmd[0] == "attempt":
                _, attempt, gaze, pause, key, set_id = cmd
                if key != menu_key or pause or set_id != cur:
                    chain = []  # evidence is only combined across attempts on the same menu and frequency set
                if set_id != cur:
                    cur = set_id
                    dec.scorer = scorers[cur]
                    band.set_active(sets[cur]["menu_hz"])
                menu_key, closed = key, False
                t_on = band.look(gaze)
                dec.threshold = sets[cur]["T_pause"] if pause else sets[cur]["T"]
                dec.stimulus_start(t_on)
                ev_q.put(("onset", attempt, band.wall_time(t_on)))
        try:
            chunk, t_end = band.read(timeout=0.5)
        except Exception:
            continue
        for ev in dec.push(chunk):
            if closed:
                continue  # this attempt already ended (combined pick)
            sc = scorers[cur].last
            ev["combined"], ev["set"] = 1, cur
            ev["scores"] = [float(v) for v in sc]  # per-button evidence for the UI
            if pause and ev["state"] == "SELECT" and ev["option_id"] != resume_idx:
                ev.update(state="IDLE", option_id=None, final=True, ignored="pause")  # decoder ended the attempt
            elif ev["state"] == "IDLE" and chain and use_combine and not pause:
                comb = (np.sum(chain, 0) + sc) / (len(chain) + 1)
                if _margin(comb) >= sets[cur]["T_comb"]:
                    ev.update(state="SELECT", option_id=int(np.argmax(comb)), confidence=_margin(comb),
                              combined=len(chain) + 1, final=True)
            if ev["state"] == "SELECT":
                chain, closed = [], True
            elif ev.get("final"):
                chain = (chain + [np.asarray(sc, float)])[-2:]
                closed = True
            ev["latency_ms"] = 1e3 * (time.perf_counter() - band.wall_time(ev["timestamp"]))
            ev_q.put(("decision", attempt, ev))
