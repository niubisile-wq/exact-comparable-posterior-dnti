import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig1_identifiability"

COLORS = {
    "33": "#0F4D92",
    "69": "#5A9632",
    "119": "#C88D18",
    "300": "#C95B45",
    "grid": "#D9D9D9",
    "text": "#272727",
}

# Reconstructed from the manuscript figure to preserve the published trend.
K_33 = [1, 2, 10, 12, 15, 22, 27, 32]
H_33 = [2.70, 2.16, 1.04, 0.95, 0.81, 0.66, 0.56, 0.47]

K_69 = [1, 2, 3, 5, 7, 10, 13, 16, 20, 27, 32]
H_69 = [3.44, 3.07, 2.82, 2.45, 2.10, 1.94, 1.81, 1.64, 1.44, 1.26, 1.12]

K_119 = [1, 2, 3, 5, 7, 10, 12, 14, 17, 21, 27, 32]
H_119 = [4.21, 3.86, 3.55, 3.12, 2.61, 2.31, 2.11, 1.89, 1.58, 1.38, 1.20, 1.18]

K_300 = [45, 80, 120, 150, 180, 220]
H_300 = [2.49, 1.72, 1.45, 1.27, 1.20, 1.06]


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

    fig, ax = plt.subplots(figsize=(7.1, 3.2), constrained_layout=True)

    ax.plot(K_33, H_33, color=COLORS["33"], marker="o", markersize=4.2, linewidth=1.8, label="33-bus")
    ax.plot(K_69, H_69, color=COLORS["69"], marker="o", markersize=4.2, linewidth=1.8, label="69-bus")
    ax.plot(K_119, H_119, color=COLORS["119"], marker="o", markersize=4.2, linewidth=1.8, label="119-bus")
    ax.plot(K_300, H_300, color=COLORS["300"], marker="s", markersize=4.4, linewidth=1.8, label="300-bus synthetic")

    ax.set_title("Identifiability improves with measurements", pad=6)
    ax.set_xlabel("Installed sensors K")
    ax.set_ylabel("Posterior entropy H(K)")
    ax.set_xlim(-5, 225)
    ax.set_ylim(0.3, 4.35)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, columnspacing=1.6, handlelength=1.8)

    ax.text(K_33[-1] + 2.0, H_33[-1] - 0.02, "33-bus", color=COLORS["33"], fontsize=7, va="center")
    ax.text(K_69[-1] + 3.0, H_69[-1] - 0.13, "69-bus", color=COLORS["69"], fontsize=7, va="center")
    ax.text(K_119[-1] + 2.0, H_119[-1] + 0.20, "119-bus", color=COLORS["119"], fontsize=7, va="center")
    ax.text(K_300[-1] - 6.0, H_300[-1] + 0.10, "300-bus", color=COLORS["300"], fontsize=7, ha="right")

    for suffix in (".svg", ".pdf", ".png"):
        dpi = 600 if suffix == ".png" else None
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=dpi, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
