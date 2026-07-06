import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


SYSTEMS = ["33-bus", "69-bus", "119-bus"]
EXACT = np.array([72.10, 40.20, 52.80])
NRE = np.array([68.58, 36.05, 41.90])
SPEEDUP = np.array([1176, 3302, 3762])

COLORS = {
    "exact": "#767676",
    "nre": "#0F4D92",
    "speedup": "#5D9730",
    "grid": "#D9D9D9",
    "text": "#272727",
}

ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig2_ip1_accuracy_speed"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
        }
    )


def annotate_bars(ax, bars, labels, dy):
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=COLORS["text"],
        )


def main() -> None:
    apply_style()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.8),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
        constrained_layout=True,
    )

    x = np.arange(len(SYSTEMS))
    width = 0.36

    bars_exact = ax1.bar(
        x - width / 2,
        EXACT / 100.0,
        width=width,
        color=COLORS["exact"],
        label="Exact",
    )
    bars_nre = ax1.bar(
        x + width / 2,
        NRE / 100.0,
        width=width,
        color=COLORS["nre"],
        label="NRE",
    )
    ax1.set_title("Posterior inference accuracy", pad=6)
    ax1.set_ylabel("Top-1 accuracy")
    ax1.set_xticks(x, SYSTEMS)
    ax1.set_ylim(0.0, 0.82)
    ax1.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", handlelength=1.4)
    annotate_bars(ax1, bars_exact, [f"{v:.1f}%" for v in EXACT], dy=0.015)
    annotate_bars(ax1, bars_nre, [f"{v:.1f}%" for v in NRE], dy=0.015)
    ax1.text(-0.14, 1.02, "a", transform=ax1.transAxes, fontweight="bold")

    bars_speed = ax2.bar(x, SPEEDUP, width=0.62, color=COLORS["speedup"])
    ax2.set_title("Amortized online inference", pad=6)
    ax2.set_ylabel("Speedup vs exact")
    ax2.set_xticks(x, SYSTEMS)
    ax2.set_ylim(0, 4100)
    ax2.set_yticks([1000, 2000, 3000, 4000], ["1x10$^3$", "2x10$^3$", "3x10$^3$", "4x10$^3$"])
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax2.set_axisbelow(True)
    annotate_bars(ax2, bars_speed, [f"{v}x" for v in SPEEDUP], dy=75)
    ax2.text(-0.14, 1.02, "b", transform=ax2.transAxes, fontweight="bold")

    for suffix in (".svg", ".pdf", ".png"):
        dpi = 600 if suffix == ".png" else None
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=dpi, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
