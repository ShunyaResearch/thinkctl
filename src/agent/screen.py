"""Screenshots and coordinate mapping.

The flicker panel (a strip along one screen edge) is cropped out before the model sees the screen, so the model
can never act on it. The model answers in 0..1000 on both axes of the image it saw (Holo3.1 convention), so:
    x_screen = x0 + x_model / 1000 * w_region        (x0, w_region: left edge and width of the visible region)
    e.g. region = whole 2560-px width, x_model = 375  ->  0 + 375 / 1000 * 2560 = 960
The same applies to y with the region's top edge and height.
"""
import shutil, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class Region:
    x0: int
    y0: int
    w: int
    h: int

    def to_screen(self, xm, ym):
        return round(self.x0 + xm / 1000 * self.w), round(self.y0 + ym / 1000 * self.h)

    def contains(self, x, y):
        return self.x0 <= x < self.x0 + self.w and self.y0 <= y < self.y0 + self.h


def visible_region(screen_w, screen_h, panel):
    """Screen area outside the flicker panel. panel = {"edge": top|bottom|left|right, "size_px": n}."""
    n, e = int(panel.get("size_px", 0)), panel.get("edge", "bottom")
    if n <= 0:
        return Region(0, 0, screen_w, screen_h)
    return {"bottom": Region(0, 0, screen_w, screen_h - n), "top": Region(0, n, screen_w, screen_h - n),
            "left": Region(n, 0, screen_w - n, screen_h), "right": Region(0, 0, screen_w - n, screen_h)}[e]


def grab():
    """Full-screen screenshot via maim (X11). Returns a PIL RGB image."""
    exe = shutil.which("maim")
    if exe is None:
        raise RuntimeError("maim not found: run inside `nix develop path:.`")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        subprocess.run([exe, "--hidecursor", f.name], check=True)
        return Image.open(f.name).convert("RGB")


@dataclass
class Shot:
    full: Image.Image  # full screen
    region: Region  # what the model is allowed to see
    model_img: Image.Image  # region, downscaled for the model

    @classmethod
    def make(cls, full, panel, model_width):
        region = visible_region(*full.size, panel)
        crop = full.crop((region.x0, region.y0, region.x0 + region.w, region.y0 + region.h))
        h = round(model_width * region.h / region.w)
        return cls(full, region, crop.resize((model_width, h), Image.LANCZOS))

    @classmethod
    def from_file(cls, path, panel, model_width):
        return cls.make(Image.open(path).convert("RGB"), panel, model_width)

    @classmethod
    def capture(cls, panel, model_width):
        return cls.make(grab(), panel, model_width)

    def crop_around(self, x, y, w=420, h=200, scale=2):
        """Screen-pixel crop centred on (x, y) with a red marker, upscaled: for 'what is under this point?'."""
        from PIL import ImageDraw
        box = (max(0, x - w // 2), max(0, y - h // 2), min(self.full.width, x + w // 2), min(self.full.height, y + h // 2))
        im = self.full.crop(box).resize(((box[2] - box[0]) * scale, (box[3] - box[1]) * scale), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        cx, cy = (x - box[0]) * scale, (y - box[1]) * scale
        d.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], outline=(255, 0, 0), width=5)
        return im

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.full.save(path)
