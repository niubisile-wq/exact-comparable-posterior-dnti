import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig9_ablation_controls"

COLORS = {
    "nre": "#244A73",
    "exact": "#8A8F97",
    "support": "#5E9B96",
    "gain": "#C79118",
    "control": "#B76652",
    "muted": "#66707A",
    "grid": "#D9E0E8",
    "ink": "#232323",
    "panel": "#F8FAFD",
}


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
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": COLORS["panel"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        }
    )


def beautify(ax, axis="x") -> None:
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#A8B2BE")
    ax.spines["bottom"].set_color("#A8B2BE")


def panel_label(ax, letter, title):
    ax.text(-0.08, 1.08, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold")
    ax.text(0.02, 1.08, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, fontweight="semibold")


def save_all(fig) -> None:
    for suffix, dpi in ((".svg", None), (".pdf", None), (".png", 600)):
        fig.savefig(
            OUT_STEM.with_suffix(suffix),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # A. Point-estimate and negative-control baselines on the 69-bus setting.
    methods = ["Exact", "NRE", "Edge-GNN", "GraphSAGE", "DGS-lite"]
    values = np.array([40.20, 36.05, 34.35, 29.27, 12.20])
    colors = [COLORS["exact"], COLORS["nre"], COLORS["support"], "#AEBBD0", COLORS["control"]]
    y = np.arange(len(methods))
    ax1.barh(y, values, color=colors, edgecolor="white", linewidth=0.6)
    ax1.set_yticks(y, methods)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 45)
    ax1.set_xlabel("Top-1 accuracy (%)")
    beautify(ax1, "x")
    panel_label(ax1, "A", "69-bus point-estimate controls")
    for yi, v in zip(y, values):
        ax1.text(v + 0.7, yi, f"{v:.2f}", va="center", fontsize=6.8)

    # B. Robust training ablation under missing measurements.
    labels = ["33 10%", "33 30%", "69 10%", "69 30%"]
    gains = np.array([25.08, 50.91, 12.22, 23.26])
    x = np.arange(len(labels))
    bars = ax2.bar(x, gains, color=COLORS["gain"], edgecolor="white", linewidth=0.6)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 56)
    ax2.set_ylabel("Gain over naive model (pp)")
    beautify(ax2, "y")
    panel_label(ax2, "B", "Missing-measurement ablation")
    for bar, v in zip(bars, gains):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.1, f"+{v:.2f}", ha="center", fontsize=6.8)

    # C. Bayesian approximation controls: quality and real timing are separated.
    ax3.set_facecolor("white")
    ax3.axis("off")
    panel_label(ax3, "C", "Approximate Bayesian controls")
    cards = [
        ("AIS vs NRE", "18/20 vs 17/20\nmean KL 0.0082", COLORS["support"]),
        ("Real timing", "NRE 0.728 ms\nAIS 1309 ms", COLORS["nre"]),
        ("SMC 33-bus", "exact 0.697\nSMC 0.693", COLORS["exact"]),
        ("SMC 69-bus", "exact 0.412\nSMC 0.401", COLORS["exact"]),
    ]
    for i, (title, body, color) in enumerate(cards):
        row, col = divmod(i, 2)
        x0 = 0.02 + col * 0.49
        y0 = 0.54 - row * 0.44
        ax3.add_patch(plt.Rectangle((x0, y0), 0.45, 0.34, transform=ax3.transAxes,
                                    facecolor=COLORS["panel"], edgecolor=COLORS["grid"], linewidth=0.8))
        ax3.add_patch(plt.Rectangle((x0, y0 + 0.29), 0.45, 0.05, transform=ax3.transAxes,
                                    facecolor=color, edgecolor="none"))
        ax3.text(x0 + 0.025, y0 + 0.23, title, transform=ax3.transAxes,
                 ha="left", va="center", fontsize=7.2, fontweight="semibold")
        ax3.text(x0 + 0.025, y0 + 0.10, body, transform=ax3.transAxes,
                 ha="left", va="center", fontsize=7.0, color=COLORS["muted"])

    # D. Posterior-enabled decision value and downstream value.
    labels = ["BOED AUC\nvs MVG", "DSSE MAE\nimprovement", "Weighted\nbetter rate"]
    vals = np.array([12.86, 5.37, 60.00])
    units = ["pp", "%", "%"]
    x = np.arange(len(labels))
    bars = ax4.bar(x, vals, color=[COLORS["nre"], COLORS["support"], COLORS["support"]],
                   edgecolor="white", linewidth=0.6)
    ax4.set_xticks(x, labels)
    ax4.set_ylim(0, 68)
    ax4.set_ylabel("Reported effect size")
    beautify(ax4, "y")
    panel_label(ax4, "D", "Posterior value beyond top-1")
    for bar, v, unit in zip(bars, vals, units):
        ax4.text(bar.get_x() + bar.get_width() / 2, v + 1.4, f"{v:.2f}{unit}", ha="center", fontsize=6.8)
    ax4.text(0.02, -0.25, "Panels use their native metrics; values summarize Table 7 evidence.",
             transform=ax4.transAxes, ha="left", va="top", fontsize=6.6, color=COLORS["muted"])

    save_all(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
