# viz_style.py
import contextlib
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter, PercentFormatter

@contextlib.contextmanager
def paper_style():
    rc = {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.2,
    }
    with plt.rc_context(rc):
        yield

def format_kb_axis(ax):
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel("KB transferred")

def format_pct_axis(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Prefetch hit-rate (%)")

def add_bar_labels(ax, values, is_percent=False):
    for p, v in zip(ax.patches, values):
        y = p.get_height()
        txt = f"{v:,.0f}" if not is_percent else f"{v:.0f}%"
        ax.text(p.get_x() + p.get_width()/2, y, txt,
                ha="center", va="bottom", fontsize=11)
