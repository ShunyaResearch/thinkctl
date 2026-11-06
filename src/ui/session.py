"""Session controller: menus <-> flicker strip <-> brain process (band + decoder), plus the scripted user.

Attempt timeline (all buttons phase-locked to the attempt start, like the dataset's trials):
  cue 0.5 s (static; the user moves their eyes) -> flicker (decoder: dynamic stopping 0.5-2.0 s)
  -> SELECT: highlight the picked button 0.5 s, then the app executes the pick
  -> final IDLE (no confident pick by 2.0 s): next attempt.
Pause mode: only the 'Resume' button is shown and flickers; the decoder uses the stricter pause threshold.

ScriptedUser: knows the intended path (e.g. Email -> 1st e-mail -> Reply -> draft 2 -> Send -> Confirm). At each
attempt it looks at the button that moves it along the path from the CURRENT app state; if a wrong pick took the app
elsewhere it looks at Back (or Resume when paused), as a real user would. It may also look away (idle attempts,
e.g. while reading an e-mail): any pick then is a false activation, and the app executes it for real.
"""
import json, time

import numpy as np
from dataclasses import dataclass, field

CUE_S, HIGHLIGHT_S, MAX_ATTEMPT_S = 0.5, 0.5, 4.0


# --------------------------------------------------------------------------- scripted user
@dataclass
class Step:
    state: tuple  # app state (screen, context) in which this step is taken; "$name" = a bound value
    label: str  # button label to pick (exact), prefix if it ends with '*', or "#k" = the k-th option
    idle_before: int = 0  # attempts spent looking away first (reading)
    bind: str = ""  # remember the picked label under this name (e.g. which e-mail was the top one)


class ScriptedUser:
    def __init__(self, steps, pause_idle=0, pause_first=False):
        self.steps, self.idle_done, self.bound = steps, {}, {}
        self.pause_idle = pause_idle  # attempts spent looking away while paused (tests the strict pause threshold)
        self.pause_pending = pause_first  # test: pick Pause first, look away pause_idle attempts, then Resume

    def _resolve(self, x):
        if isinstance(x, str) and x.startswith("$"):
            return self.bound.get(x[1:], ("<unbound>", x))
        if isinstance(x, tuple):
            return tuple(self._resolve(v) for v in x)
        return x

    def target(self, state, menu_labels, paused):
        """Label to look at, or None (looking away)."""
        if self.pause_pending and not paused and "Pause" in menu_labels:
            return "Pause"
        if paused:
            self.pause_pending = False
            if self.pause_idle > 0:
                self.pause_idle -= 1
                return None
            return "Resume"
        for k in range(len(self.steps) - 1, -1, -1):  # deepest step whose state matches
            st = self.steps[k]
            if self._resolve(st.state) == state:
                if self.idle_done.get(k, 0) < st.idle_before:
                    self.idle_done[k] = self.idle_done.get(k, 0) + 1
                    return None
                if st.bind and st.bind in self.bound:  # look for the remembered item again, wherever it is now
                    lab = self.bound[st.bind]
                    return lab if lab in menu_labels else ("Back" if "Back" in menu_labels else None)
                lab = _match(st.label, menu_labels)
                if st.bind and lab and lab not in FIXED:
                    self.bound[st.bind] = lab
                if lab is None and "Back" in menu_labels:
                    return "Back"  # the wanted button is not offered: go back instead of staring at nothing
                return lab
        return "Back" if "Back" in menu_labels else None


def progress(user, state):
    """(step number, total, label) of the deepest scripted step matching the app state."""
    steps = user.steps
    for k in range(len(steps) - 1, -1, -1):
        if user._resolve(steps[k].state) == state:
            lab = steps[k].label.rstrip("*")
            if lab.startswith("#"):
                lab = user.bound.get(steps[k].bind) or f"option {int(lab[1:]) + 1}"
            return k + 1, len(steps), lab
    return None, len(steps), "recovering (Back)"


FIXED = {"Back", "Pause", "Resume", ""}


def _match(label, labels):
    if label.startswith("#"):  # k-th model-made option (never Back / Pause)
        k = int(label[1:])
        items = [l for l in labels if l not in FIXED]
        return items[k] if k < len(items) else ("Back" if "Back" in labels else None)
    if label.endswith("*"):
        for l in labels:
            if l.startswith(label[:-1]):
                return l
    return label if label in labels else None


# --------------------------------------------------------------------------- session
@dataclass
class AttemptLog:
    attempt: int
    screen: str
    gaze: str | None
    t_cue: float
    decision: str = ""  # SELECT label / IDLE
    outcome: str = ""  # correct / wrong / false_activation / miss / correct_idle
    window_s: float = 0.0
    confidence: float = 0.0
    latency_ms: float = 0.0
    t_decision: float = 0.0
    combined: int = 1  # number of attempts whose evidence produced this decision
    set_id: str = "A"  # frequency set on the buttons during this stare


@dataclass
class Session:
    app: object
    user: ScriptedUser
    strip: object
    cmd_q: object
    ev_q: object
    freqs: list
    resume_index: int
    max_seconds: float = 300.0
    attempts: list = field(default_factory=list)
    events: list = field(default_factory=list)
    timing: dict = field(default_factory=lambda: {"eeg_s": 0.0, "app_s": 0.0})
    stop_file: object = None

    def run(self):
        st = self.strip
        info = None
        while info is None:  # wait for the brain
            if not self.ev_q.empty():
                m = self.ev_q.get()
                info = m[1] if m[0] == "ready" else None
            st.draw([""] * 6, info=dict(state="STARTING…", lines=["loading recorded EEG + decoder"]))
        self.info = info
        self.set_id, streak, streak_key, self.swaps = "A", 0, None, []
        self.pick_counts = {}
        t_start = time.perf_counter()
        attempt, phase, t_phase, onset_wall = 0, "new", time.perf_counter(), None
        frame0, picked, conf, decision, gaze_label = None, None, None, None, None
        evidence, combined = None, 1
        app_t0 = None
        banner = None
        self.phase = "new"
        while True:
            now = time.perf_counter()
            self.phase = phase
            ev = st.events()
            if ev in ("stop", "quit") or (self.stop_file and self.stop_file.exists()):
                banner = "STOPPED (kill switch / Esc)"
                self.events.append(("stopped", now - t_start, ev or "STOP file"))
                break
            if self.app.done or now - t_start > self.max_seconds:
                break
            menu = self.app.menu()
            paused = self.app.paused
            labels = [o.label for o in menu.options][:6] + [""] * (6 - len(menu.options[:6]))
            if paused:
                labels = [""] * 6
                labels[self.resume_index] = "Resume"
            cur = info["sets"][self.set_id]
            thr = cur["T_pause"] if paused else cur["T"]
            # ---- drain decoder events
            while not self.ev_q.empty():
                m = self.ev_q.get()
                if m[0] == "onset" and m[1] == attempt:
                    onset_wall = m[2]
                elif m[0] == "decision" and m[1] == attempt and phase in ("cue", "flicker"):
                    e = m[2]
                    conf, combined = e["confidence"], e.get("combined", 1)
                    if e.get("scores"):
                        sc = np.asarray(e["scores"], float)
                        evidence = (sc - sc.min()) / (np.ptp(sc) + 1e-9)
                    if e["state"] == "SELECT" or e.get("final"):
                        decision = e
            # ---- state machine
            if phase == "app":
                if not self.app.busy:
                    self.timing["app_s"] += now - app_t0
                    phase = "new"
                st.draw(labels, info=self._info("WORKING…", (215, 165, 60), [self.app.status or "app is acting"],
                                                 now - t_start, attempt, banner=banner))
                continue
            if phase == "new":
                attempt += 1
                gaze_label = self.user.target(self.app.state(), labels, paused)
                gaze_k = labels.index(gaze_label) if gaze_label in labels else None
                key = (menu.screen, tuple(labels), str(self.app.state()))
                if key != streak_key:
                    streak, streak_key = 0, key  # inconclusive stares are counted per screen
                self.cmd_q.put(("attempt", attempt, cur["menu_hz"][gaze_k] if gaze_k is not None else None, paused,
                                key, self.set_id))
                self.attempts.append(AttemptLog(attempt, "pause" if paused else menu.screen, gaze_label, now - t_start,
                                                set_id=self.set_id))
                phase, t_phase, onset_wall, frame0, decision, conf, picked = "cue", now, None, None, None, None, None
                evidence, combined = None, 1
            a = self.attempts[-1]
            if phase == "cue":
                if onset_wall is not None and now >= onset_wall - 0.004:
                    phase, frame0 = "flicker", len(st.flips)
            if phase == "flicker" and decision is not None:
                self.timing["eeg_s"] += now - t_phase
                if decision["state"] == "SELECT":
                    k = decision["option_id"]
                    picked = k
                    a.decision = labels[k] if labels[k] else f"hidden#{k}"
                    a.outcome = ("false_activation" if a.gaze is None else
                                 "correct" if labels[k] == a.gaze else "wrong")
                else:
                    a.decision = "IDLE"
                    a.outcome = "correct_idle" if a.gaze is None else "miss"
                a.window_s, a.confidence = decision["window_s"], decision["confidence"]
                a.combined = decision.get("combined", 1)
                a.latency_ms, a.t_decision = decision.get("latency_ms", 0.0), now - t_start
                phase, t_phase = "result", now
            if phase == "flicker" and now - t_phase > MAX_ATTEMPT_S:  # safety net: no decoder answer
                a.decision, a.outcome, phase, t_phase = "TIMEOUT", "miss", "result", now
            # ---- draw
            gz = labels.index(a.gaze) if a.gaze in labels else None
            common = dict(conf=conf, thr=thr, combined=combined, evidence=evidence, gaze=gz)
            if phase == "cue":
                st.draw(labels, info=self._info("GET READY", (120, 220, 255), [self._where(menu, paused)],
                                                now - t_start, attempt, banner=banner, **common))
            elif phase == "flicker":
                active = [self.resume_index] if paused else None
                st.draw(labels, flicker_i=len(st.flips) - frame0, active=active,
                        info=self._info("LISTENING", (120, 220, 255), [self._where(menu, paused)], now - t_start,
                                        attempt, banner=banner, **common))
            elif phase == "result":
                if picked is not None:
                    st.draw(labels, highlight=picked,
                            info=self._info(f"PICKED: {labels[picked] or '?'}", (255, 210, 0),
                                            [self._where(menu, paused)], now - t_start, attempt, banner=banner,
                                            **{**common, "conf": a.confidence, "combined": a.combined}))
                else:
                    st.draw(labels, info=self._info("NO PICK (not confident)", (215, 165, 60),
                                                    [self._where(menu, paused)], now - t_start, attempt,
                                                    banner=banner, **common))
                if now - t_phase >= HIGHLIGHT_S:
                    if picked is not None and labels[picked]:
                        opt = ([o for o in menu.options if o.label == labels[picked]] or [None])[0]
                        if paused:
                            opt = "resume"
                        app_t0 = now
                        key2 = (menu.screen, labels[picked])
                        self.pick_counts[key2] = self.pick_counts.get(key2, 0) + 1
                        if self.pick_counts[key2] > 4:  # the same pick on the same screen again and again: stop
                            banner = f"LOOP GUARD: '{labels[picked]}' on {menu.screen} picked 5 times"
                            self.events.append(("loop_guard", now - t_start, key2))
                            break
                        self.app.pick(opt)
                        phase = "app"
                    else:
                        phase = "new"
                        # swap the frequency set after K inconclusive stares in a row on this screen (not in pause)
                        streak += 1
                        K = info.get("K")
                        if K and not paused and streak >= K:
                            self.set_id = "B" if self.set_id == "A" else "A"
                            nxt = info["sets"][self.set_id]
                            st.set_freqs(nxt["menu_hz"], nxt["phases"])
                            self.swaps.append((round(now - t_start, 1), menu.screen, self.set_id))
                            streak = 0
                    if picked is not None:
                        streak = 0
        st.draw([""] * 6, info=dict(state="STOPPED" if banner else "FINISHED", banner=banner,
                                    lines=[f"task {'done' if self.app.done else 'not done'}",
                                           f"{len(self.attempts)} stares, {time.perf_counter() - t_start:.0f} s"]))
        self.cmd_q.put(("quit",))
        log = None
        t_wait = time.perf_counter()
        while time.perf_counter() - t_wait < 3:
            if not self.ev_q.empty():
                m = self.ev_q.get()
                if m[0] == "band_log":
                    log = m
                    break
            time.sleep(0.01)
        self.band_log = log[1] if log else []
        self.decoder_timing = log[2] if log else []
        self.seconds = time.perf_counter() - t_start
        return self

    def _where(self, menu, paused):
        return f"screen: {'PAUSED' if paused else menu.screen}"

    def _info(self, state, color, lines, elapsed, attempt, **kw):
        k, n, lab = progress(self.user, self.app.state()) if self.user else (None, 0, "")
        head = [f"REPLAY · subject {self.info['subject']} · {getattr(self.app, 'display_name', 'app')}"]
        step = [f"task step {k}/{n}: {lab}" if k else f"task: {lab}"]
        fs = self.info["sets"][self.set_id]["menu_hz"]
        sw = f" · freq set {self.set_id} ({fs[0]:g}-{fs[-1]:g} Hz)"
        return dict(state=state, state_color=color, lines=head + lines + step +
                    [f"{elapsed:5.1f} s · stare {attempt}{sw}"], **kw)

    def summary(self):
        from collections import Counter
        c = Counter(a.outcome for a in self.attempts)
        sel = [a for a in self.attempts if a.outcome in ("correct", "wrong", "false_activation")]
        idle_min = sum((a.window_s or 2.0) for a in self.attempts if a.gaze is None) / 60
        return dict(seconds=round(self.seconds, 1), attempts=len(self.attempts), outcomes=dict(c),
                    correct_picks=c["correct"], wrong_picks=c["wrong"], false_activations=c["false_activation"],
                    misses=c["miss"], fa_per_idle_min=(c["false_activation"] / idle_min) if idle_min else None,
                    mean_decision_window_s=sum(a.window_s for a in sel) / len(sel) if sel else None,
                    decision_latency_ms_median=sorted(a.latency_ms for a in sel)[len(sel) // 2] if sel else None,
                    eeg_s=round(self.timing["eeg_s"], 1), app_s=round(self.timing["app_s"], 1),
                    swaps=getattr(self, "swaps", []), K=self.info.get("K"),
                    task_done=self.app.done, task_success=getattr(self.app, "success", None),
                    events=self.events)
