import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig4_boed_sensor_placement"

COLORS = {
    "baseline": "#767676",
    "other": "#C88D18",
    "mvg": "#5A9632",
    "boed": "#0F4D92",
    "grid": "#D9D9D9",
    "text": "#272727",
}

PANELS = [
    {
        "title": "33-bus K=7",
        "labels": ["Random", "MVG", "BOED"],
        "values": [41.80, 75.40, 75.20],
        "colors": [COLORS["baseline"], COLORS["mvg"], COLORS["boed"]],
    },
    {
        "title": "69-bus K=7",
        "labels": ["Random", "GreedyLoop", "MVG", "BOED"],
        "values": [24.00, 24.00, 66.50, 71.50],
        "colors": [COLORS["baseline"], COLORS["other"], COLORS["mvg"], COLORS["boed"]],
    },
    {
        "title": "119-bus K=7",
        "labels": ["AdaptiveMVG", "AdaptiveFisher", "AdaptiveBOED"],
        "values": [56.70, 58.30, 70.00],
        "colors": [COLORS["mvg"], COLORS["other"], COLORS["boed"]],
    },
]


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


def main() -> None:
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharey=True, constrained_layout=True)

    for idx, (ax, panel) in enumerate(zip(axes, PANELS)):
        x = np.arange(len(panel["labels"]))
        bars = ax.bar(x, panel["values"], color=panel["colors"], width=0.72)
        ax.set_title(panel["title"], pad=5)
        ax.set_xticks(x, panel["labels"], rotation=23, ha="right")
        ax.set_ylim(0, 82)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.set_ylabel("Top-1 accuracy (%)")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.text(-0.16, 1.02, chr(ord("a") + idx), transform=ax.transAxes, fontweight="bold")

    for suffix in (".svg", ".pdf", ".png"):
        dpi = 600 if suffix == ".png" else None
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=dpi, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
