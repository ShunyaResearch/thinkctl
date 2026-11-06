"""Dry-run marker: a red ring with a label at (x, y) on the real screen, always on top, auto-closes.
Usage: python -I agent/overlay.py X Y LABEL [SECONDS]"""
import sys
import tkinter as tk

x, y, label = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
R, W, H = 34, 360, 110
root = tk.Tk()
root.overrideredirect(True)
root.attributes("-topmost", True)
try:
    root.wait_visibility(root)
    root.attributes("-alpha", 0.85)
except tk.TclError:
    pass
root.geometry(f"{W}x{H}+{x - R - 4}+{y - R - 4}")
c = tk.Canvas(root, width=W, height=H, bg="black", highlightthickness=0)
c.pack()
c.create_oval(4, 4, 4 + 2 * R, 4 + 2 * R, outline="#ff3030", width=5)
c.create_line(4 + R, 4 + R - 10, 4 + R, 4 + R + 10, fill="#ff3030", width=3)
c.create_line(4 + R - 10, 4 + R, 4 + R + 10, 4 + R, fill="#ff3030", width=3)
c.create_text(2 * R + 14, R + 4, text=label, fill="#ffffff", anchor="w", font=("Sans", 13, "bold"))
root.after(int(secs * 1000), root.destroy)
root.mainloop()
