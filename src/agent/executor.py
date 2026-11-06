"""Executes actions with xdotool (X11), or only shows them (dry-run). Kill switch checked before every input.

Kill switch: the file STOP in the project folder (created by the hotkey, see scripts/killswitch.sh, or by hand).
While it exists, no input is sent and every goal ends with 'fail: stopped'. Delete the file to re-arm.
"""
import shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
STOP = ROOT / "STOP"
XKEYS = {"enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape", "tab": "Tab",
         "backspace": "BackSpace", "space": "space", "up": "Up", "down": "Down", "left": "Left", "right": "Right",
         "pageup": "Prior", "pagedown": "Next", "home": "Home", "end": "End", "ctrl": "ctrl", "alt": "alt",
         "shift": "shift"}


class Stopped(RuntimeError):
    pass


def stopped():
    return STOP.exists()


def check_stop():
    if stopped():
        raise Stopped("kill switch active (STOP file present)")


class Executor:
    def __init__(self, dry_run=True, show_overlay=True):
        self.dry_run, self.show_overlay = dry_run, show_overlay
        self.xdotool = shutil.which("xdotool")
        if not dry_run and self.xdotool is None:
            raise RuntimeError("xdotool not found: run inside `nix develop path:.`")

    def _x(self, *args):
        check_stop()
        subprocess.run([self.xdotool, *map(str, args)], check=True)

    def overlay(self, x, y, label, seconds=1.5):
        """Dry-run: draw the planned action on the real screen (a red ring + label, click-through, auto-closes)."""
        if self.show_overlay:
            subprocess.Popen([sys.executable, "-I", str(ROOT / "src" / "agent" / "overlay.py"), str(x), str(y), label,
                              str(seconds)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self, action, sx=None, sy=None, payload_text=None):
        """action: agent.actions.Action; (sx, sy): screen pixels for point actions."""
        check_stop()
        n = action.name
        if self.dry_run:
            if action.xy is not None:
                self.overlay(sx, sy, f"DRY RUN: {n}")
            return f"dry-run: {n} at ({sx},{sy})" if action.xy else f"dry-run: {action.short()}"
        if n == "click":
            self._x("mousemove", sx, sy)
            self._x("click", 1)
        elif n == "double_click":
            self._x("mousemove", sx, sy)
            self._x("click", "--repeat", 2, "--delay", 80, 1)
        elif n == "scroll":
            self._x("mousemove", sx, sy)
            for _ in range(action.args["amount"]):
                self._x("click", 4 if action.args["direction"] == "up" else 5)
        elif n == "key":
            keys = "+".join(XKEYS.get(k, k) for k in action.args["combo"].split("+"))
            self._x("key", "--clearmodifiers", keys)
        elif n == "type_text":
            if not payload_text:
                raise ValueError("type_text without payload")
            for i in range(0, len(payload_text), 20):  # small chunks, so the kill switch acts within ~0.5 s
                self._x("type", "--delay", 15, "--", payload_text[i:i + 20])
        else:
            raise ValueError(n)
        time.sleep(0.4)  # let the UI react before the next screenshot
        return f"executed {action.short()}"
