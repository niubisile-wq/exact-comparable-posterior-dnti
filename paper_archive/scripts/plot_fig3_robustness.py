import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig3_robustness"

COLORS = {
    "robust": "#0F4D92",
    "naive": "#C95B45",
    "grid": "#D9D9D9",
    "text": "#272727",
}

RANDOM_LABELS = ["33-bus\n10%", "33-bus\n30%", "69-bus\n10%", "69-bus\n30%"]
RANDOM_ROBUST = np.array([65.18, 61.01, 33.02, 30.56])
RANDOM_NAIVE = np.array([40.10, 10.10, 20.80, 7.30])

OUTAGE_LABELS = ["Naive", "IP-C robust"]
OUTAGE_VALUES = np.array([32.35, 62.97])
OUTAGE_COLORS = [COLORS["naive"], COLORS["robust"]]


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


def annotate_bars(ax, bars, dy: float = 1.2) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def main() -> None:
    apply_style()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.9),
        gridspec_kw={"width_ratios": [1.45, 0.85]},
        constrained_layout=True,
    )

    x = np.arange(len(RANDOM_LABELS))
    width = 0.35
    bars_naive = ax1.bar(x - width / 2, RANDOM_NAIVE, width=width, color=COLORS["naive"], label="Naive")
    bars_robust = ax1.bar(x + width / 2, RANDOM_ROBUST, width=width, color=COLORS["robust"], label="IP-C robust")
    ax1.set_title("Random missing measurements", pad=6)
    ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.set_xticks(x, RANDOM_LABELS)
    ax1.set_ylim(0, 76)
    ax1.set_yticks([0, 20, 40, 60, 80])
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", handlelength=1.4)
    annotate_bars(ax1, bars_naive)
    annotate_bars(ax1, bars_robust)
    ax1.text(-0.14, 1.02, "a", transform=ax1.transAxes, fontweight="bold")

    x2 = np.arange(len(OUTAGE_LABELS))
    bars_outage = ax2.bar(x2, OUTAGE_VALUES, width=0.62, color=OUTAGE_COLORS)
    ax2.set_title("Outage-induced missing", pad=6)
    ax2.set_ylabel("Top-1 accuracy (%)")
    ax2.set_xticks(x2, OUTAGE_LABELS)
    ax2.set_ylim(0, 76)
    ax2.set_yticks([0, 20, 40, 60, 80])
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax2.set_axisbelow(True)
    annotate_bars(ax2, bars_outage)
    ax2.text(-0.14, 1.02, "b", transform=ax2.transAxes, fontweight="bold")

    for suffix in (".svg", ".pdf", ".png"):
        dpi = 600 if suffix == ".png" else None
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=dpi, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
