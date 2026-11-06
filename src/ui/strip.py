"""Flicker strip: an always-on-top window along the bottom screen edge with 6 flickering buttons + a status panel.

Sampled sinusoidal stimulation (works for any frequency on a 60 Hz or 144 Hz monitor):
    L(i) = 1/2 * (1 + sin(2*pi*f*i/R + phi))
    L = brightness 0..1, i = frames since the flicker started, f = button frequency (Hz), R = refresh rate (Hz),
    phi = the button's phase (D4 dataset: 0 or pi). Example: f = 10, R = 60, i = 3, phi = 0 -> sin(pi) = 0 -> L = 0.5.
Soft + small: only a 110 px square per button flickers, and its grey level is
    B = 0.3 + 0.4 * L   (30 % .. 70 % instead of black .. white; e.g. L = 0.5 -> B = 0.5, mid grey).
When not flickering the square rests at B = 0.5 (the same mean brightness, so start / stop does not flash).
Frames are paced by vsync (SDL 'SCALED' mode); every flip time is logged for the frame-timing report.
Stop flicker instantly: Esc (window focused) or the global kill switch Ctrl+Alt+K (STOP file).
"""
import os, subprocess, shutil, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
ACK = ROOT / "logs" / ".photosensitivity_ack"
TITLE = "thinkctl-flicker"

WARNING = ["PHOTOSENSITIVITY WARNING",
           "This program shows flickering lights at 9 to 14 Hz. Flicker in this range can trigger seizures in people",
           "with photosensitive epilepsy. Do not continue if you or anyone watching may be affected.",
           "Stop the flicker at any time: Esc (this window) or Ctrl+Alt+K (anywhere).",
           "Press ENTER to continue, or Esc to quit."]


def luminance(freqs, phases, i, R):
    """L(i) = 1/2 (1 + sin(2 pi f i / R + phi)) for every button."""
    return 0.5 * (1 + np.sin(2 * np.pi * np.asarray(freqs, float) * i / R + np.asarray(phases, float)))


class Strip:
    def __init__(self, freqs, phases, screen_w=2560, screen_h=1600, height=260, refresh=60.0, status_w=600,
                 square=100, depth=(0.3, 0.7)):
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"0,{screen_h - height}"
        import pygame
        self.pg = pygame
        pygame.init()
        self.W, self.H, self.R = screen_w, height, float(refresh)
        self.freqs, self.phases = np.asarray(freqs, float), np.asarray(phases, float)
        self.headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"  # tests: invisible window, 60 fps by timer
        if self.headless:
            self.scr = pygame.display.set_mode((self.W, self.H))
            self._clock = pygame.time.Clock()
        else:
            self.scr = pygame.display.set_mode((self.W, self.H), pygame.NOFRAME | pygame.SCALED, vsync=1)
        pygame.display.set_caption(TITLE)
        F = lambda size, bold=False: pygame.font.SysFont("dejavusans", size, bold=bold)  # noqa: E731
        self.f_big, self.font, self.small, self.tiny = F(34, True), F(23, True), F(20), F(16, True)
        n, gap = len(freqs), 14
        bw = (self.W - status_w - gap * (n + 1)) // n
        self.rects = [pygame.Rect(status_w + gap + k * (bw + gap), 10, bw, self.H - 20) for k in range(n)]
        self.squares = []
        for r in self.rects:
            sq = pygame.Rect(0, 0, square, square)
            sq.midtop = (r.centerx, r.y + 16)
            self.squares.append(sq)
        self.label_box = [pygame.Rect(r.x + 10, r.y + 16 + square + 8, r.w - 20, 60) for r in self.rects]
        self.ev_bars = [pygame.Rect(r.x + 18, r.bottom - 22, r.w - 36, 10) for r in self.rects]
        self.status_rect = pygame.Rect(0, 0, status_w, self.H)
        self.lo, self.hi = depth
        self.flips = []  # perf_counter after every flip
        self.flicker_log = []  # (flip index, frame-in-flicker index i) while flickering
        self._cache = {}
        self.keep_on_top()

    def set_freqs(self, freqs, phases):
        self.freqs, self.phases = np.asarray(freqs, float), np.asarray(phases, float)

    def keep_on_top(self):
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            return
        x = shutil.which("xdotool")
        if x:
            subprocess.run(f'{x} search --name "{TITLE}" windowstate --add ABOVE', shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ------------------------------------------------------------------ helpers
    def luminance(self, i):
        return luminance(self.freqs, self.phases, i, self.R)

    def _text(self, text, font, color):
        key = (text, id(font), color)
        if key not in self._cache:
            self._cache[key] = font.render(text, True, color)
        return self._cache[key]

    def fit_lines(self, text, font, max_w, max_lines=2):
        """Word-wrap into <= max_lines lines that fit max_w pixels; the last line ends with '…' if cut."""
        words, lines, cur = text.split(), [], ""
        for w in words:
            t = f"{cur} {w}".strip()
            if font.size(t)[0] <= max_w:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] += "…"
        out = []
        for ln in lines:
            while font.size(ln)[0] > max_w and len(ln) > 1:
                ln = ln[:-2].rstrip() + "…"
            out.append(ln)
        return out

    def events(self):
        """Returns 'stop' on Esc, 'enter' on Enter, 'quit' on window close, else None."""
        pg = self.pg
        out = None
        for e in pg.event.get():
            if e.type == pg.QUIT:
                out = "quit"
            elif e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                out = "stop"
            elif e.type == pg.KEYDOWN and e.key in (pg.K_RETURN, pg.K_KP_ENTER):
                out = out or "enter"
        return out

    def flip(self):
        self.pg.display.flip()
        if self.headless:
            self._clock.tick(60)
        self.flips.append(time.perf_counter())

    # ------------------------------------------------------------------ screens
    def warning(self):
        """First start only: photosensitivity warning. Returns True if acknowledged."""
        if ACK.exists():
            return True
        pg = self.pg
        while True:
            ev = self.events()
            if ev in ("stop", "quit"):
                return False
            if ev == "enter":
                ACK.write_text(time.strftime("%Y-%m-%d %H:%M:%S acknowledged\n"))
                return True
            self.scr.fill((40, 0, 0))
            for k, line in enumerate(WARNING):
                f = self.f_big if k == 0 else self.small
                self.scr.blit(f.render(line, True, (255, 230, 120) if k == 0 else (255, 255, 255)), (40, 16 + k * 44))
            self.flip()

    def draw(self, labels, flicker_i=None, active=None, highlight=None, info=None):
        """labels: 6 strings ('' = hidden). flicker_i: frames since flicker start, or None (static).
        info: dict for the status panel and overlays: state, state_color, lines (list of str), conf, thr, combined,
        evidence (6 relative scores or None), gaze (button index the simulated user looks at, replay only), banner."""
        pg, info = self.pg, info or {}
        self.scr.fill((16, 16, 20))
        lum = self.luminance(flicker_i) if flicker_i is not None else None
        ev, gaze = info.get("evidence"), info.get("gaze")
        thr_hit = info.get("conf") is not None and info.get("thr") and info["conf"] >= info["thr"]
        for k, (r, sq, lb, eb, lab) in enumerate(zip(self.rects, self.squares, self.label_box, self.ev_bars, labels)):
            if not lab:
                continue
            on = lum is not None and (active is None or k in active)
            pg.draw.rect(self.scr, (40, 40, 50), r, border_radius=12)
            g = int(255 * (self.lo + (self.hi - self.lo) * (lum[k] if on else 0.5)))
            pg.draw.rect(self.scr, (g, g, g), sq, border_radius=8)
            col = (255, 210, 0) if highlight == k else (238, 238, 238)
            for j, ln in enumerate(self.fit_lines(lab, self.font, lb.w)):
                t = self._text(ln, self.font, col)
                self.scr.blit(t, t.get_rect(midtop=(lb.centerx, lb.y + j * 28)))
            if ev is not None:  # relative evidence per button (decoder scores, min..max of the 6)
                pg.draw.rect(self.scr, (60, 60, 72), eb, border_radius=5)
                top = int(np.argmax(ev))
                c = (90, 210, 120) if (k == top and thr_hit) else (110, 150, 235) if k == top else (95, 110, 150)
                pg.draw.rect(self.scr, c, pg.Rect(eb.x, eb.y, max(4, int(eb.w * ev[k])), eb.h), border_radius=5)
            if gaze == k:
                pg.draw.rect(self.scr, (60, 220, 255), r, 3, border_radius=12)
                t = self._text("simulated gaze", self.tiny, (60, 220, 255))
                self.scr.blit(t, (r.x + 10, r.y + 4))
            if highlight == k:
                pg.draw.rect(self.scr, (255, 210, 0), r, 7, border_radius=12)
        # ---- status panel
        S = self.status_rect
        pg.draw.rect(self.scr, (28, 28, 36), S)
        title = self.fit_lines(info.get("state", ""), self.f_big, S.w - 36, 1)[0] if info.get("state") else ""
        self.scr.blit(self._text(title, self.f_big, info.get("state_color", (120, 220, 255))), (18, 10))
        y = 54
        for ln in info.get("lines", []):
            for part in self.fit_lines(ln, self.small, S.w - 36, 1):
                self.scr.blit(self._text(part, self.small, (205, 205, 215)), (18, y))
            y += 27
        conf, thr = info.get("conf"), info.get("thr")
        if conf is not None and thr:
            bar = pg.Rect(18, S.h - 52, S.w - 36, 18)
            pg.draw.rect(self.scr, (60, 60, 72), bar, border_radius=6)
            frac = max(0.0, min(1.0, conf / (1.5 * thr)))
            pg.draw.rect(self.scr, (90, 210, 120) if conf >= thr else (215, 165, 60),
                         pg.Rect(bar.x, bar.y, int(bar.w * frac), bar.h), border_radius=6)
            tx = bar.x + int(bar.w / 1.5)
            pg.draw.line(self.scr, (255, 255, 255), (tx, bar.y - 5), (tx, bar.bottom + 5), 3)
            comb = info.get("combined", 1)
            lbl = f"confidence {conf:.2f} / needed {thr:.2f}" + (f"  (over {comb} stares)" if comb > 1 else "")
            self.scr.blit(self._text(lbl, self.small, (205, 205, 215)), (18, S.h - 30))
        if info.get("banner"):
            t = self._text(info["banner"], self.font, (255, 110, 110))
            self.scr.blit(t, (S.w + 20, 2))
        if flicker_i is not None:
            self.flicker_log.append((len(self.flips), flicker_i))
        self.flip()

    def close(self):
        self.pg.quit()


def frame_report(flips, flicker_log, freqs, phases, R):
    """Dropped frames and achieved frequency per button from the logged flip times.
    dropped: a flip interval longer than 1.5 frame periods (one or more refreshes missed).
    achieved frequency: the button's luminance L_k(i) as actually shown at flip times t_i is fitted with a
    sinusoid (Lomb-Scargle peak, 0.0005 Hz grid around f). With index-based stimulation it equals
    f * R_measured / R_nominal when no frame is dropped (e.g. 10 Hz * 59.94 / 60 = 9.99 Hz)."""
    from scipy.signal import lombscargle
    t = np.asarray(flips)
    idx = np.array([a for a, _ in flicker_log])
    fi = np.array([b for _, b in flicker_log])
    tf = t[idx]
    dt = np.diff(tf)
    contiguous = np.diff(fi) == 1
    dt = dt[contiguous]
    period = 1.0 / R
    dropped = int(np.sum(dt > 1.5 * period))
    R_meas = 1.0 / np.median(dt)
    out = dict(frames=int(len(fi)), seconds=float(tf[-1] - tf[0]), refresh_nominal=R, refresh_measured=float(R_meas),
               mean_interval_ms=float(1e3 * dt.mean()), p99_interval_ms=float(1e3 * np.percentile(dt, 99)),
               max_interval_ms=float(1e3 * dt.max()), dropped_frames=dropped,
               dropped_pct=float(100 * dropped / max(1, len(dt))), buttons=[])
    for f, ph in zip(freqs, phases):
        L = 0.5 * (1 + np.sin(2 * np.pi * f * fi / R + ph))
        grid = np.arange(f - 0.2, f + 0.2, 0.0005)
        p = lombscargle(tf - tf[0], L - L.mean(), 2 * np.pi * grid)
        out["buttons"].append(dict(f_target=float(f), f_achieved=float(grid[np.argmax(p)]),
                                   f_expected_from_refresh=float(f * R_meas / R)))
    return out
