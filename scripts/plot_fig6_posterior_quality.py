import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig6_posterior_quality"

COLORS = {
    "reference": "#767676",
    "raw": "#C88D18",
    "calibrated": "#0F4D92",
    "grid": "#D9D9D9",
    "text": "#272727",
}

# Reconstructed from the provided manuscript figure to preserve the shown reliability trend.
CONF = np.array([0.18, 0.24, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.99])
REF = np.array([0.17, 0.23, 0.34, 0.41, 0.56, 0.67, 0.73, 0.86, 0.995])
RAW = np.array([0.16, 0.19, 0.32, 0.48, 0.59, 0.61, 0.79, 0.86, 0.995])
CAL = np.array([0.16, 0.19, 0.34, 0.50, 0.57, 0.67, 0.78, 0.88, 0.998])

TOPOLOGIES = ["3", "4"]
EXACT_POST = np.array([0.63, 0.37])
CAL_POST = np.array([0.63, 0.37])


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

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.55),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
        constrained_layout=True,
    )

    ax1.plot(CONF, REF, color=COLORS["reference"], marker="o", markersize=4.0, linewidth=1.8, label="Reference")
    ax1.plot(CONF, RAW, color=COLORS["raw"], marker="o", markersize=4.0, linewidth=1.8, label="Raw NRE")
    ax1.plot(CONF, CAL, color=COLORS["calibrated"], marker="o", markersize=4.0, linewidth=1.8, label="Calibrated NRE")
    ax1.plot([0, 1], [0, 1], linestyle="--", color=COLORS["reference"], linewidth=1.2)
    ax1.set_title("Reliability", pad=5)
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Empirical accuracy")
    ax1.set_xlim(-0.02, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", handlelength=1.8)
    ax1.text(-0.16, 1.02, "a", transform=ax1.transAxes, fontweight="bold")

    x = np.arange(len(TOPOLOGIES))
    width = 0.36
    bars_exact = ax2.bar(x - width / 2, EXACT_POST, width=width, color=COLORS["reference"], label="Exact")
    bars_cal = ax2.bar(x + width / 2, CAL_POST, width=width, color=COLORS["calibrated"], label="Calibrated NRE")
    ax2.set_title("Ambiguous case", pad=5)
    ax2.set_xlabel("Topology")
    ax2.set_ylabel("Posterior probability")
    ax2.set_xticks(x, TOPOLOGIES)
    ax2.set_ylim(0, 0.68)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", handlelength=1.8)
    for bars in (bars_exact, bars_cal):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.text(-0.16, 1.02, "b", transform=ax2.transAxes, fontweight="bold")

    for suffix in (".svg", ".pdf", ".png"):
        dpi = 600 if suffix == ".png" else None
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=dpi, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
