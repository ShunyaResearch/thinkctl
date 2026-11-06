"""Run one end-to-end session: replay band (held-out D4 subject) -> decoder -> flicker strip -> app.

  python -I src/demo.py --app fake --subject 28 --seed 1          # menus only, no model
  python -I src/demo.py --app email --subject 28 --seed 1 --dry-run    # real Gmail, actions only drawn
  python -I src/demo.py --app email --subject 28 --seed 1 --record     # real Gmail, real clicks (+ screen video)
Run inside `nix develop path:.` (xdotool / maim). Logs: logs/runs/<time>_<app>_S<subject>_seed<seed>/
Kill switch: Ctrl+Alt+K (scripts/killswitch.sh running) or Esc on the strip.
"""
import argparse, csv, json, multiprocessing as mp, os, sys, time
from dataclasses import asdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # nothing in this process uses the GPU (the model server is separate)
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))

from ui.session import Session, ScriptedUser, Step  # noqa: E402

RESUME_INDEX = 2  # the Resume button sits in the middle of the strip (11.25 Hz)


def fake_script(app, variant="email"):
    inbox = app.inbox_labels()
    from app.fake import DRAFTS
    from agent.options import short
    e, d = app.target
    return [Step(("home", None), "Email"), Step(("inbox", None), inbox[e]),
            Step(("email", e), "Reply", idle_before=2), Step(("drafts", e), short(DRAFTS[d]), idle_before=1),
            Step(("typed", (e, d)), "Send"), Step(("confirm", (e, d)), "Confirm send")]


def email_script():
    """Email -> the top e-mail (remembered) -> Reply -> draft 2 -> Send -> Confirm; reading pauses included."""
    return [Step(("home", None), "Email"), Step(("inbox", None), "#0", bind="email"),
            Step(("email", "$email"), "Reply", idle_before=2), Step(("drafts", "$email"), "#1", idle_before=1),
            Step(("typed", ("$email", 1)), "Send"), Step(("confirm", ("$email", 1)), "Confirm send")]


class Monitor:
    """Peak VRAM of the model server and peak RAM (RSS) of the demo processes + model server, sampled every 1 s."""

    def __init__(self, pids):
        import threading
        self.pids, self.peak_vram_mib, self.peak_rss_mb, self._stop = pids, 0, {}, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import subprocess
        while not self._stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    pid, mem = [x.strip() for x in line.split(",")]
                    self.peak_vram_mib = max(self.peak_vram_mib, int(mem))
            except Exception:
                pass
            for name, pid in self.pids.items():
                try:
                    kb = int([l for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS")][0].split()[1])
                    self.peak_rss_mb[name] = max(self.peak_rss_mb.get(name, 0), kb // 1024)
                except Exception:
                    pass
            time.sleep(1.0)

    def stop(self):
        self._stop = True
        return dict(peak_vram_mib=self.peak_vram_mib, peak_rss_mb=self.peak_rss_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", choices=["fake", "email"], default="fake")
    ap.add_argument("--dry-run", action="store_true", help="email app: plan and draw every action, click nothing")
    ap.add_argument("--record", action="store_true", help="record the screen (ffmpeg x11grab) to logs/videos/")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--max-seconds", type=float, default=240)
    ap.add_argument("--script", choices=["email", "pause"], default="email")
    ap.add_argument("--kill-after", type=float, default=None, help="test: press Ctrl+Alt+K after N seconds")
    ap.add_argument("--no-combine", action="store_true", help="single-attempt rule only (for comparison)")
    ap.add_argument("--no-swap", action="store_true", help="never swap the frequency set")
    a = ap.parse_args()
    from band.brain import brain_main
    from ui.strip import Strip, frame_report

    info = json.load(open(ROOT / "models/decoder/d4wet_menu6.json"))
    ctx = mp.get_context("spawn")
    cmd_q, ev_q = ctx.Queue(), ctx.Queue()
    brain = ctx.Process(target=brain_main, args=(cmd_q, ev_q, dict(subject=a.subject, seed=a.seed, data_root=a.data,
                                                                  resume_index=RESUME_INDEX, combine=not a.no_combine,
                                                                  swap=not a.no_swap)), daemon=True)
    brain.start()
    strip = Strip(info["menu_hz"], info["phases_rad"])
    if not strip.warning():
        cmd_q.put(("quit",))
        strip.close()
        return
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}_{a.app}{'-dry' if a.dry_run else ''}_S{a.subject}_seed{a.seed}"
    if a.app == "fake":
        from app.fake import FakeEmailApp
        app = FakeEmailApp()
        user = ScriptedUser(fake_script(app, a.script), pause_idle=3 if a.script == "pause" else 0,
                            pause_first=a.script == "pause")
    else:
        import urllib.request
        from agent.executor import Executor
        from agent.holo import Holo
        from app.email_app import RealEmailApp
        from settings import load_config
        cfg = load_config()
        cfg["panel"] = {"edge": "bottom", "size_px": strip.H}  # the model never sees the flicker strip
        try:
            urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=3)
        except Exception:
            print("model server not running: start scripts/run_holo_server.sh first")
            cmd_q.put(("quit",))
            strip.close()
            return
        # setup (not an agent action): bring the Firefox window to the front, then keep the strip on top of it.
        # wmctrl only lists real top-level windows (xdotool also finds hidden helper windows named "Firefox").
        import subprocess
        wl = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True).stdout.splitlines()
        ff = [l.split()[0] for l in wl if "Mozilla Firefox" in l]
        if ff:
            subprocess.run(["wmctrl", "-i", "-a", ff[0]])
            time.sleep(1.0)
            strip.keep_on_top()
        else:
            print("WARNING: no Firefox window found")
        app = RealEmailApp(Holo(cfg["holo_server"]), Executor(dry_run=a.dry_run, show_overlay=True), cfg)
        user = ScriptedUser(email_script())
    video = None
    if a.record:
        import shutil, subprocess
        (ROOT / "logs" / "videos").mkdir(parents=True, exist_ok=True)
        video = subprocess.Popen(["/usr/bin/ffmpeg", "-loglevel", "error",  # system ffmpeg: has x11grab "-f", "x11grab", "-framerate", "15",
                                  "-video_size", "2560x1600", "-i", os.environ.get("DISPLAY", ":1"), "-c:v", "libx264",
                                  "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
                                  str(ROOT / "logs" / "videos" / f"{run_name}.mp4")], stdin=subprocess.PIPE)
    srv = [l.split(",")[0].strip() for l in os.popen("nvidia-smi --query-compute-apps=pid,process_name "
                                                       "--format=csv,noheader").read().splitlines() if "llama" in l]
    mon = Monitor(dict(demo=os.getpid(), brain=brain.pid, **({"llama_server": srv[0]} if srv else {})))
    sess = Session(app, user, strip, cmd_q, ev_q, info["menu_hz"], RESUME_INDEX, max_seconds=a.max_seconds,
                   stop_file=ROOT / "STOP")
    if a.kill_after:  # test: press the global kill-switch hotkey while the strip is flickering
        import subprocess, threading, shutil

        def press():
            time.sleep(a.kill_after)
            while getattr(sess, "phase", "") != "flicker":
                time.sleep(0.01)
            time.sleep(0.3)
            (ROOT / "logs" / "kill_pressed_at").write_text(repr(time.perf_counter()))
            subprocess.run([shutil.which("xdotool"), "key", "--clearmodifiers", "ctrl+alt+k"])
        threading.Thread(target=press, daemon=True).start()
    sess.run()
    resources = mon.stop()
    if video:
        video.communicate(b"q", timeout=30)
    out = ROOT / "logs" / "runs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    summ = sess.summary()
    try:
        summ["frames"] = {k: v for k, v in frame_report(strip.flips, strip.flicker_log, info["menu_hz"],
                                                         info["phases_rad"], strip.R).items() if k != "buttons"}
    except Exception as e:  # e.g. no flicker happened
        summ["frames"] = str(e)
    dt = [t[1] for t in sess.decoder_timing]
    summ["decoder_cpu_ms_median"] = sorted(dt)[len(dt) // 2] if dt else None
    summ.update(subject=a.subject, seed=a.seed, app=a.app, script=a.script, threshold=sess.info["threshold"],
                threshold_pause=sess.info["threshold_pause"], threshold_comb=sess.info["threshold_comb"],
                thresholds_by_set=sess.info["sets"], fold=sess.info["fold"],
                combine=not a.no_combine, app_history=app.history, resources=resources,
                video=f"logs/videos/{run_name}.mp4" if video else None)
    if a.app == "email":
        bound = user.bound.get("email")
        summ.update(dry_run=a.dry_run, goals=app.goals, app_timing={k: round(v, 1) for k, v in app.timing.items()},
                    sent=app.sent, target_email=bound, final_status=app.status,
                    task_correct=bool(app.sent and app.sent["email"] == bound and app.sent["draft"] == 1),
                    time_split_s=dict(eeg_picks=summ["eeg_s"], model_menus=round(app.timing["menu_s"], 1),
                                      agent_actions=round(app.timing["agent_s"], 1)))
    if a.kill_after and (ROOT / "logs" / "kill_pressed_at").exists():
        t_key = float((ROOT / "logs" / "kill_pressed_at").read_text())
        last_flicker_flip = strip.flips[strip.flicker_log[-1][0]] if strip.flicker_log else None
        summ["kill_switch"] = dict(stop_file_created=(ROOT / "STOP").exists(),
                                   frames_after_key=sum(t > t_key for t in strip.flips),
                                   last_flicker_frame_after_key_ms=None if last_flicker_flip is None else
                                   round(1e3 * (last_flicker_flip - t_key), 1),
                                   stopped_screen_after_key_ms=round(1e3 * (strip.flips[-1] - t_key), 1))
    json.dump(summ, open(out / "summary.json", "w"), indent=1, default=str)
    with open(out / "attempts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(sess.attempts[0]).keys()))
        w.writeheader()
        w.writerows(asdict(x) for x in sess.attempts)
    json.dump(sess.band_log, open(out / "band_log.json", "w"), default=float)
    strip.close()
    brain.join(timeout=3)
    print(json.dumps({k: v for k, v in summ.items() if k != "app_history"}, indent=1, default=str))
    print("logs:", out)


if __name__ == "__main__":
    main()
