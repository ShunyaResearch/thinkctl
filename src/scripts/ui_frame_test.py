"""Frame-timing test: all 6 buttons flicker for 60 s; every frame time is logged.
Report: dropped frames (target < 0.5 %), measured refresh rate, achieved frequency per button.
Run inside `nix develop path:.`:  python -I src/scripts/ui_frame_test.py [--seconds 60]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

from ui.strip import Strip, frame_report  # noqa: E402

STOP = ROOT / "STOP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    info = json.load(open(ROOT / "models/decoder/d4wet_menu6.json"))
    strip = Strip(info["menu_hz"], info["phases_rad"])
    if not strip.warning():
        strip.close()
        print("warning not acknowledged: stopped")
        return
    labels = ["Email", "YouTube", "Discord", "Steam", "More", "Pause"]
    t_end = time.perf_counter() + a.seconds
    i, stopped = 0, None
    while time.perf_counter() < t_end:
        ev = strip.events()
        if ev in ("stop", "quit") or (i % 15 == 0 and STOP.exists()):
            stopped = ev or "STOP file"
            break
        strip.draw(labels, flicker_i=i, info=dict(state="FRAME TEST", lines=[f"{time.perf_counter() - t_end + a.seconds:5.1f} s"]))
        i += 1
    strip.draw(labels, info=dict(state="STOPPED" if stopped else "DONE"))
    rep = frame_report(strip.flips, strip.flicker_log, info["menu_hz"], info["phases_rad"], strip.R)
    rep["stopped_early"] = stopped
    out = ROOT / "logs" / "ckpt2"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frame_flips.npy", np.array(strip.flips))
    json.dump(rep, open(out / "frame_report.json", "w"), indent=1)
    strip.close()
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
