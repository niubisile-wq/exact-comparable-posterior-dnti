from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fig7_evidence_board"


PALETTE = {
    "ink": "#1F2A35",
    "muted": "#657384",
    "grid": "#D9E1EA",
    "card": "#FFFFFF",
    "paper": "#F6F8FB",
    "exact": "#8C949E",
    "nre": "#3F73B7",
    "naive": "#C7D1DE",
    "robust": "#C99116",
    "teal": "#5D9C96",
    "red": "#B76550",
    "steel": "#98A9C6",
    "steel2": "#B9C5D8",
    "light": "#D6DDE6",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["card"],
            "legend.frameon": False,
        }
    )


def save_pub(fig: plt.Figure) -> None:
    for ext, dpi in [(".pdf", None), (".svg", None), (".png", 600), (".tiff", 600)]:
        path = OUT.with_suffix(ext)
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "edgecolor": "none"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        if ext == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        print(f"{path.name}: {path.stat().st_size}")


def card(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    bg = FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        edgecolor="#D7DFE9",
        facecolor=PALETTE["card"],
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(bg)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=PALETTE["muted"], labelsize=7.5, length=0, pad=3)
    ax.tick_params(axis="y", colors=PALETTE["muted"], labelsize=7.5, length=0, pad=8)


def panel_header(ax: plt.Axes, letter: str, title: str, subtitle: str) -> None:
    ax.text(
        -0.045,
        1.095,
        letter,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="circle,pad=0.25", facecolor=PALETTE["ink"], edgecolor=PALETTE["ink"]),
        clip_on=False,
    )
    ax.text(
        0.065,
        1.115,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.8,
        fontweight="bold",
        color=PALETTE["ink"],
        clip_on=False,
    )
    ax.text(
        0.065,
        1.045,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=PALETTE["muted"],
        clip_on=False,
    )


def badge(ax: plt.Axes, x: float, y: float, text: str, color: str, *, fontsize: float = 7.2) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.12", facecolor="white", edgecolor=color, linewidth=0.75),
        zorder=5,
    )


def label_bar(ax: plt.Axes, value: float, y: float, color: str, *, unit: str = "%", dx: float = 1.1) -> None:
    ax.text(value + dx, y, f"{value:.2f}{unit}", ha="left", va="center", fontsize=7.2, color=color)


def panel_a(ax: plt.Axes) -> None:
    card(ax)
    panel_header(ax, "A", "Exact-comparable audit", "NRE is checked against the exact posterior on the same finite support.")

    systems = ["33-bus", "69-bus", "119-bus"]
    exact = np.array([72.10, 40.20, 52.80])
    nre = np.array([68.58, 36.05, 41.90])
    y = np.arange(len(systems))[::-1]
    h = 0.28

    ax.barh(y + h / 2, exact, height=h, color=PALETTE["exact"], edgecolor="white", linewidth=0.8, label="Exact")
    ax.barh(y - h / 2, nre, height=h, color=PALETTE["nre"], edgecolor="white", linewidth=0.8, label="NRE")
    ax.set_yticks(y)
    ax.set_yticklabels(systems)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Top-1 accuracy (%)", fontsize=7.8, color=PALETTE["muted"])
    ax.legend(
        handles=[Patch(color=PALETTE["exact"], label="Exact"), Patch(color=PALETTE["nre"], label="NRE")],
        loc="lower right",
        bbox_to_anchor=(0.99, 1.035),
        ncol=2,
        fontsize=7.2,
        columnspacing=1.0,
        handlelength=1.4,
    )

    for yi, e, q, sp in zip(y, exact, nre, ["1176x", "3302x", "3762x"]):
        label_bar(ax, e, yi + h / 2, PALETTE["exact"], dx=0.9)
        label_bar(ax, q, yi - h / 2, PALETTE["nre"], dx=0.9)
        badge(ax, 77.5, yi, sp, PALETTE["nre"], fontsize=6.8)
    ax.text(0.02, -0.20, "Speedup badges are online NRE time relative to exact enumeration.", transform=ax.transAxes, fontsize=7.0, color=PALETTE["muted"])


def panel_b(ax: plt.Axes) -> None:
    card(ax)
    panel_header(ax, "B", "Missing-data robustness", "Mask-aware training repairs severe random telemetry loss.")

    systems = ["33-bus", "69-bus"]
    naive = np.array([10.10, 7.30])
    robust = np.array([61.01, 30.56])
    gains = robust - naive
    y = np.arange(len(systems))[::-1]
    h = 0.30

    ax.barh(y + h / 2, naive, height=h, color=PALETTE["naive"], edgecolor="white", linewidth=0.8, label="Naive")
    ax.barh(y - h / 2, robust, height=h, color=PALETTE["robust"], edgecolor="white", linewidth=0.8, label="Robust")
    ax.set_yticks(y)
    ax.set_yticklabels(systems)
    ax.set_xlim(0, 72)
    ax.set_xlabel("Top-1 accuracy (%)", fontsize=7.8, color=PALETTE["muted"])
    ax.legend(
        handles=[Patch(color=PALETTE["naive"], label="Naive"), Patch(color=PALETTE["robust"], label="Robust")],
        loc="lower right",
        bbox_to_anchor=(0.99, 1.035),
        ncol=2,
        fontsize=7.2,
        columnspacing=1.0,
        handlelength=1.4,
    )

    for yi, n, r, g in zip(y, naive, robust, gains):
        label_bar(ax, n, yi + h / 2, PALETTE["muted"], dx=0.8)
        label_bar(ax, r, yi - h / 2, "#8B630B", dx=0.8)
        badge(ax, min(r + 8.0, 66.0), yi, f"+{g:.2f} pp", PALETTE["red"])
    ax.text(0.02, -0.20, "Large-system repair: 202-bus top-20 91.04%; 417-bus top-20 80.69%.", transform=ax.transAxes, fontsize=7.0, color=PALETTE["muted"])


def panel_c(ax: plt.Axes) -> None:
    card(ax)
    panel_header(ax, "C", "Realism and three-phase stress", "Public synchronized replay anchors the posterior under bounded realism.")

    labels = ["SoCal majority", "SoCal posterior"]
    vals = np.array([36.10, 86.64])
    y = np.arange(len(labels))[::-1]
    colors = [PALETTE["naive"], PALETTE["teal"]]

    ax.barh(y, vals, height=0.42, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)", fontsize=7.8, color=PALETTE["muted"])
    for yi, v, c in zip(y, vals, colors):
        label_bar(ax, v, yi, c if c != PALETTE["naive"] else PALETTE["muted"], dx=1.0)
    badge(ax, 72, y[1] + 0.34, "95% CI [85.18, 88.11]", PALETTE["teal"], fontsize=6.8)
    ax.text(
        0.02,
        -0.20,
        "IEEE123 stress: 20% dropout + 2x noise rerank@20 79.28 vs exact 79.22; 40% dropout rerank@20 72.83.",
        transform=ax.transAxes,
        fontsize=7.0,
        color=PALETTE["muted"],
    )


def panel_d(ax: plt.Axes) -> None:
    card(ax)
    panel_header(ax, "D", "Decision value and boundary repair", "Posterior mass improves sensing decisions and defines fallback use.")

    labels = ["BOED", "MVG", "Fisher", "Random"]
    vals = np.array([54.60, 41.75, 38.89, 11.90])
    y = np.arange(len(labels))[::-1]
    colors = [PALETTE["red"], PALETTE["steel"], PALETTE["steel2"], PALETTE["light"]]

    ax.barh(y, vals, height=0.45, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 62)
    ax.set_xlabel("Accuracy AUC (%)", fontsize=7.8, color=PALETTE["muted"])
    for yi, v, c in zip(y, vals, colors):
        label_bar(ax, v, yi, c if c != PALETTE["light"] else PALETTE["muted"], dx=0.9)
    badge(ax, 48.2, y[0] - 0.45, "+12.86 pp vs MVG", PALETTE["red"])
    ax.text(
        0.02,
        -0.20,
        "Boundary repair: 300-bus clean 49.38 vs exact 53.60; 30% missing 26.87 -> 42.60 after reranking.",
        transform=ax.transAxes,
        fontsize=7.0,
        color=PALETTE["muted"],
    )
    ax.text(
        0.02,
        -0.29,
        "Posterior quality: KL 0.0453; ECE 0.0181; coverage 97.62%.",
        transform=ax.transAxes,
        fontsize=7.0,
        color=PALETTE["muted"],
    )


def main() -> None:
    style()
    fig = plt.figure(figsize=(13.8, 8.9))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.98, top=0.84, bottom=0.15, hspace=0.62, wspace=0.23)

    fig.text(0.075, 0.975, "Condensed evidence board", fontsize=14.0, fontweight="bold", color=PALETTE["ink"], ha="left")
    fig.text(
        0.075,
        0.938,
        "The strongest result families are grouped by claim role rather than by dataset chronology.",
        fontsize=8.5,
        color=PALETTE["muted"],
        ha="left",
    )

    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[1, 0]))
    panel_d(fig.add_subplot(gs[1, 1]))

    save_pub(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
