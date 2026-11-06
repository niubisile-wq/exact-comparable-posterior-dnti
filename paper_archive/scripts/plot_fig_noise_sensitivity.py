import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig_noise_sensitivity"

COLORS = {
    "direct": "#C95B45",
    "rerank": "#0F4D92",
    "exact": "#767676",
    "grid": "#D9D9D9",
    "text": "#272727",
}

NOISE_LABELS_119 = ["1x", "2x", "3x"]
DIRECT_119 = np.array([64.28, 47.88, 36.38])
RERANK_119 = np.array([np.nan, 62.00, 50.48])
EXACT_119 = np.array([np.nan, 62.55, 51.68])

NOISE_LABELS_300 = ["1x", "2x"]
DIRECT_300 = np.array([32.02, 20.48])
RERANK_300 = np.array([np.nan, 34.72])
EXACT_300 = np.array([np.nan, 35.34])


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


def add_series(ax, x, values, width, offset, color, label):
    valid = ~np.isnan(values)
    bars = ax.bar(x[valid] + offset, values[valid], width=width, color=color, label=label)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    return bars


def main() -> None:
    apply_style()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.95),
        gridspec_kw={"width_ratios": [1.2, 0.8]},
        constrained_layout=True,
    )

    width = 0.22

    x1 = np.arange(len(NOISE_LABELS_119))
    add_series(ax1, x1, DIRECT_119, width, -width, COLORS["direct"], "Direct NRE")
    add_series(ax1, x1, RERANK_119, width, 0.0, COLORS["rerank"], "Rerank@20")
    add_series(ax1, x1, EXACT_119, width, width, COLORS["exact"], "Exact")
    ax1.set_title("119-bus K60", pad=6)
    ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.set_xlabel("Noise multiplier")
    ax1.set_xticks(x1, NOISE_LABELS_119)
    ax1.set_ylim(0, 72)
    ax1.set_yticks([0, 20, 40, 60])
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", handlelength=1.4)
    ax1.text(-0.14, 1.02, "a", transform=ax1.transAxes, fontweight="bold")

    x2 = np.arange(len(NOISE_LABELS_300))
    add_series(ax2, x2, DIRECT_300, width, -width, COLORS["direct"], "Direct NRE")
    add_series(ax2, x2, RERANK_300, width, 0.0, COLORS["rerank"], "Rerank@20")
    add_series(ax2, x2, EXACT_300, width, width, COLORS["exact"], "Exact")
    ax2.set_title("300-bus, 30% missing", pad=6)
    ax2.set_xlabel("Noise multiplier")
    ax2.set_xticks(x2, NOISE_LABELS_300)
    ax2.set_ylim(0, 40)
    ax2.set_yticks([0, 10, 20, 30, 40])
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.text(-0.14, 1.02, "b", transform=ax2.transAxes, fontweight="bold")

    for suffix in (".svg", ".pdf", ".png"):
        dpi = 600 if suffix == ".png" else None
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=dpi, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
