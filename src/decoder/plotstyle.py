"""Shared matplotlib style: validated categorical palette (fixed order), recessive axes, thin marks."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_CMAP = "Blues"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.markersize": 6, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "figure.dpi": 110,
    })


def line_with_label(ax, x, y, label, i, yerr=None, direct=True, **kw):
    """Line in series slot i, plus a direct label at the line end (text stays in text ink)."""
    c = SERIES[i]
    ax.plot(x, y, "-o", color=c, label=label, **kw)
    if yerr is not None:
        ax.fill_between(x, [a - e for a, e in zip(y, yerr)], [a + e for a, e in zip(y, yerr)], color=c, alpha=0.12,
                        linewidth=0)
    if direct:
        ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=TEXT2)
