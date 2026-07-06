import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig5_scalability"

COLORS = {
    "curve": "#0F4D92",
    "exact": "#767676",
    "nre": "#0F4D92",
    "stress": "#C95B45",
    "grid": "#D9D9D9",
    "text": "#272727",
}

K_300 = np.array([45, 80, 120, 150, 180, 220])
H_300 = np.array([2.49, 1.72, 1.45, 1.27, 1.20, 1.06])
H_ERR = np.array([0.84, 0.88, 0.87, 0.87, 0.88, 0.85])

CLEAN_VALUES = {"Exact": 53.60, "NRE": 49.38}
MISS_VALUES = {"Exact": 42.60, "NRE": 26.87}


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


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=7)


def main() -> None:
    apply_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.1, 2.55),
        gridspec_kw={"width_ratios": [1.15, 0.95, 0.95]},
        constrained_layout=True,
    )
    ax1, ax2, ax3 = axes

    ax1.errorbar(
        K_300,
        H_300,
        yerr=H_ERR,
        color=COLORS["curve"],
        marker="o",
        markersize=4.6,
        linewidth=1.8,
        capsize=3,
    )
    ax1.set_title("Identifiability", pad=5)
    ax1.set_xlabel("Installed sensors K")
    ax1.set_ylabel("Posterior entropy H(K)")
    ax1.set_xlim(35, 225)
    ax1.set_ylim(0.0, 3.45)
    ax1.set_xticks([50, 75, 100, 125, 150, 175, 200, 225])
    ax1.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax1.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.text(-0.18, 1.02, "a", transform=ax1.transAxes, fontweight="bold")

    x2 = np.arange(2)
    bars2 = ax2.bar(x2, list(CLEAN_VALUES.values()), color=[COLORS["exact"], COLORS["nre"]], width=0.62)
    ax2.set_title("300-bus clean deployment", pad=5)
    ax2.set_xticks(x2, list(CLEAN_VALUES.keys()))
    ax2.set_ylabel("Top-1 accuracy (%)")
    ax2.set_ylim(0, 62)
    ax2.set_yticks([0, 20, 40, 60])
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax2.set_axisbelow(True)
    annotate_bars(ax2, bars2)
    ax2.text(-0.18, 1.02, "b", transform=ax2.transAxes, fontweight="bold")

    x3 = np.arange(2)
    bars3 = ax3.bar(x3, list(MISS_VALUES.values()), color=[COLORS["exact"], COLORS["stress"]], width=0.62)
    ax3.set_title("30% missing boundary; speedup 40.2x", pad=5)
    ax3.set_xticks(x3, list(MISS_VALUES.keys()))
    ax3.set_ylabel("Top-1 accuracy (%)")
    ax3.set_ylim(0, 52)
    ax3.set_yticks([0, 10, 20, 30, 40, 50])
    ax3.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax3.set_axisbelow(True)
    annotate_bars(ax3, bars3)
    ax3.text(-0.18, 1.02, "c", transform=ax3.transAxes, fontweight="bold")

    for suffix in (".svg", ".pdf", ".png"):
        dpi = 600 if suffix == ".png" else None
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=dpi, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
