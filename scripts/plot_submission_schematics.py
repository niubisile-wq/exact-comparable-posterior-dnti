from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle


ROOT = Path(__file__).resolve().parent

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "ink": "#1f2a35",
    "muted": "#607080",
    "line": "#31465a",
    "blue": "#dfeaf7",
    "blue2": "#b9d4ef",
    "green": "#dff1e6",
    "green2": "#a9d9bc",
    "gold": "#f8ebcf",
    "gold2": "#e8c66d",
    "rose": "#f6dfdc",
    "rose2": "#e6a5a0",
    "slate": "#edf1f4",
    "white": "#ffffff",
}


def save_all(fig: plt.Figure, name: str, *, dpi: int = 600) -> None:
    base = ROOT / name
    metadata = {"Creator": "matplotlib", "Producer": "matplotlib"}
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec=None, fontsize=8.0, lw=1.1, radius=0.08):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec or COLORS["line"],
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        zorder=3,
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, *, color=None, lw=1.2, rad=0.0, style="-|>", ms=11):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def badge(ax, x, y, text, fc, r=0.17, fontsize=8):
    circ = Circle((x, y), r, facecolor=fc, edgecolor=COLORS["line"], linewidth=0.9, zorder=4)
    ax.add_patch(circ)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight="bold", color=COLORS["ink"], zorder=5)


def figure_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(0.2, 5.82, "End-to-end audited posterior workflow", fontsize=12, weight="bold", color=COLORS["ink"])
    ax.text(
        0.2,
        5.42,
        "The fast posterior is used only inside a reference-checkable operating contract.",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    stages = [
        ("Enumerable\ntopology library", COLORS["blue"]),
        ("Exact posterior\nreference", COLORS["green"]),
        ("NRE posterior\nsame support", COLORS["gold"]),
        ("Decision layer\nand fallback", COLORS["rose"]),
    ]
    xs = [0.52, 2.86, 5.20, 7.54]
    w, h, y_stage = 1.72, 0.88, 4.02
    for i, ((label, fc), x) in enumerate(zip(stages, xs), 1):
        box(ax, x, y_stage, w, h, label, fc, fontsize=8.5)
        badge(ax, x + 0.13, y_stage + h + 0.06, str(i), COLORS["white"], r=0.145, fontsize=7)
        if i < len(stages):
            arrow(ax, (x + w + 0.07, y_stage + h / 2), (xs[i] - 0.08, y_stage + h / 2), lw=1.3)

    ax.text(0.55, 3.37, "Contract checkpoints", fontsize=8.6, weight="bold", color=COLORS["ink"])
    checks = [
        ("finite candidate\nsupport", COLORS["blue"]),
        ("exact posterior\naudit", COLORS["green"]),
        ("amortized\nonline query", COLORS["gold"]),
        ("bounded-loss\npolicy", COLORS["rose"]),
    ]
    y_check, h_check = 2.63, 0.58
    for i, ((label, fc), x) in enumerate(zip(checks, xs)):
        box(ax, x, y_check, w, h_check, label, fc, fontsize=7.5, lw=0.9, radius=0.06)
        if i < len(checks) - 1:
            arrow(ax, (x + w + 0.07, y_check + h_check / 2), (xs[i + 1] - 0.08, y_check + h_check / 2), lw=0.9, ms=8)

    def panel(x, y, width, height, fc):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            linewidth=1.0,
            edgecolor=COLORS["line"],
            facecolor=fc,
            zorder=2,
        )
        ax.add_patch(patch)

    panel(0.55, 0.62, 4.25, 1.46, COLORS["white"])
    ax.text(0.80, 1.79, "Audit outputs", fontsize=8.9, weight="bold", color=COLORS["ink"])
    audit = [
        "top-1 and top-K recovery",
        "KL / ECE / credible sets",
        "mask-aware missing-data tests",
        "confidence-gated exact reranking",
    ]
    for j, item in enumerate(audit):
        y = 1.50 - 0.24 * j
        ax.add_patch(Rectangle((0.83, y - 0.045), 0.08, 0.08, facecolor=COLORS["green2"], edgecolor="none", zorder=3))
        ax.text(1.00, y, item, fontsize=7.5, va="center", color=COLORS["ink"])

    panel(5.20, 0.62, 4.15, 1.46, COLORS["slate"])
    ax.text(5.45, 1.79, "Operating gate", fontsize=8.9, weight="bold", color=COLORS["ink"])
    ax.text(5.45, 1.45, "calibrated posterior -> act / measure / rerank", fontsize=7.6, color=COLORS["ink"], va="center")
    ax.text(5.45, 1.16, "low confidence -> exact or high-fidelity review", fontsize=7.6, color="#8b5e2b", va="center")
    ax.text(5.45, 0.88, "fallback is explicit rather than hidden in the fast path", fontsize=7.2, color=COLORS["muted"], va="center")
    arrow(ax, (8.40, y_check), (8.40, 2.12), color="#8b5e2b", lw=0.9, ms=8)
    fig.subplots_adjust(0, 0, 1, 1)
    save_all(fig, "fig_workflow_3d")


def figure_principle():
    fig, ax = plt.subplots(figsize=(7.2, 4.65))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.25)
    ax.axis("off")

    ax.text(0.2, 5.93, "Exact-comparable posterior principle", fontsize=12, weight="bold", color=COLORS["ink"])
    ax.text(0.2, 5.52, "Both paths score the same finite candidate support; only the timing differs.", fontsize=8.5, color=COLORS["muted"])

    box(ax, 0.55, 3.08, 1.55, 0.74, "measurement\nvector $y$", COLORS["slate"], fontsize=8.3)
    box(ax, 3.0, 4.23, 2.0, 0.84, "Exact evaluator\n$p(t\\mid y)$", COLORS["green"], fontsize=8.7)
    box(ax, 3.0, 2.55, 2.0, 0.84, "Learned NRE\n$q_\\theta(t\\mid y)$", COLORS["gold"], fontsize=8.7)

    arrow(ax, (2.10, 3.45), (3.0, 4.65), lw=1.1)
    arrow(ax, (2.10, 3.45), (3.0, 2.97), lw=1.1)

    for x, y, color in [(5.48, 4.65, COLORS["green2"]), (5.48, 2.97, COLORS["gold2"])]:
        for k, h in enumerate([0.48, 0.36, 0.23, 0.15, 0.08]):
            ax.add_patch(Rectangle((x + k * 0.18, y - h / 2), 0.11, h, facecolor=color, edgecolor=COLORS["line"], lw=0.4, zorder=3))
    ax.text(5.36, 5.15, "shared support", fontsize=7.4, color=COLORS["muted"])
    ax.text(5.36, 2.39, "same candidates", fontsize=7.4, color=COLORS["muted"])

    arrow(ax, (5.0, 4.65), (5.46, 4.65), lw=1.0, ms=8)
    arrow(ax, (5.0, 2.97), (5.46, 2.97), lw=1.0, ms=8)

    box(ax, 6.78, 2.88, 1.37, 1.84, "posterior\ncomparison", COLORS["blue"], fontsize=8.4)
    arrow(ax, (6.34, 4.65), (6.78, 4.28), lw=1.0)
    arrow(ax, (6.34, 2.97), (6.78, 3.32), lw=1.0)

    box(ax, 8.62, 4.18, 1.02, 0.62, "KL / ECE", COLORS["slate"], fontsize=7.3)
    box(ax, 8.62, 3.49, 1.02, 0.62, "credible\nsets", COLORS["slate"], fontsize=7.3)
    box(ax, 8.62, 2.80, 1.02, 0.62, "rerank\nvalue", COLORS["slate"], fontsize=7.3)
    for y in [4.49, 3.80, 3.11]:
        arrow(ax, (8.15, y), (8.60, y), lw=0.9, ms=8)

    box(
        ax,
        1.0,
        0.55,
        8.0,
        0.9,
        "Decision implication: if $q_\\theta$ preserves posterior shape,\nbounded-loss decisions inherit a bounded regret penalty.",
        COLORS["rose"],
        fontsize=8.0,
        lw=1.0,
    )
    arrow(ax, (7.46, 2.88), (7.46, 1.47), lw=1.0, ms=9)
    fig.subplots_adjust(0, 0, 1, 1)
    save_all(fig, "fig_principle_3d")


def figure_benchmark():
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(0.2, 5.63, "Benchmark roles and claim boundaries", fontsize=12, weight="bold", color=COLORS["ink"])
    ax.text(
        0.2,
        5.25,
        "Each system supports a specific part of the posterior contract; no row is treated as private utility deployment.",
        fontsize=8.3,
        color=COLORS["muted"],
    )

    rows = [
        ("Core exact audit", "33 / 69 / 119 bus", "accuracy, speed, posterior quality", COLORS["blue"]),
        ("Realism anchor", "SoCal replay", "blocked public synchronized measurements", COLORS["green"]),
        ("Three-phase stress", "IEEE123 + 37 bus", "unbalance, phase ambiguity, dropout", COLORS["gold"]),
        ("Scale and fallback", "202 / 417 / 300 bus", "top-20 recall and gated reranking", COLORS["rose"]),
    ]
    y0 = 4.28
    for i, (role, systems, evidence, fc) in enumerate(rows):
        y = y0 - i * 0.92
        box(ax, 0.55, y, 1.75, 0.56, role, fc, fontsize=7.7, lw=1.0)
        box(ax, 2.55, y, 1.72, 0.56, systems, COLORS["white"], fontsize=7.7, lw=1.0)
        box(ax, 4.55, y, 3.0, 0.56, evidence, COLORS["slate"], fontsize=7.3, lw=0.9)
        arrow(ax, (2.32, y + 0.28), (2.53, y + 0.28), lw=0.9, ms=8)
        arrow(ax, (4.29, y + 0.28), (4.53, y + 0.28), lw=0.9, ms=8)
        ax.add_patch(Circle((8.18, y + 0.28), 0.16, facecolor=fc, edgecolor=COLORS["line"], lw=0.8))
        ax.text(8.5, y + 0.28, "bounded interpretation", fontsize=7.4, color=COLORS["ink"], va="center")

    box(ax, 0.9, 0.5, 8.2, 0.74, "Submission claim: decision-ready posterior screening under stated enumerable-library conditions.", COLORS["slate"], fontsize=8.2, lw=1.0)
    fig.subplots_adjust(0, 0, 1, 1)
    save_all(fig, "fig_benchmark_3d")


def figure_graphical_abstract():
    # IJEPES-compatible wide graphical abstract, close to 13 x 5 cm.
    fig, ax = plt.subplots(figsize=(13.28 / 2.54, 5.31 / 2.54))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(0.15, 3.65, "Exact-comparable posterior contract for topology identification", fontsize=9.3, weight="bold", color=COLORS["ink"])
    ax.text(0.15, 3.28, "Fast NRE inference is audited against exact support and gated for safe decision use.", fontsize=6.7, color=COLORS["muted"])

    steps = [
        ("Library", "enumerable\ncandidates", COLORS["blue"]),
        ("Reference", "exact posterior\np(t|y)", COLORS["green"]),
        ("Fast model", "NRE posterior\nq(t|y)", COLORS["gold"]),
        ("Stress tests", "missingness\nnoise scale", COLORS["rose"]),
        ("Operation", "BOED\nrerank audit", COLORS["slate"]),
    ]
    x0 = 0.25
    w = 2.05
    gap = 0.28
    for i, (title, subtitle, fc) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, 1.45, w, 1.05, f"{title}\n{subtitle}", fc, fontsize=6.7, lw=0.9, radius=0.06)
        badge(ax, x + 0.15, 2.62, str(i + 1), COLORS["white"], r=0.12, fontsize=6.0)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.02, 1.98), (x + w + gap - 0.04, 1.98), lw=0.9, ms=8)

    box(ax, 0.35, 0.52, 3.0, 0.52, "Core exact audit\n33/69/119-bus", COLORS["white"], fontsize=4.9, lw=0.7, radius=0.035)
    box(ax, 4.05, 0.52, 3.1, 0.52, "Stress chain\nSoCal, IEEE123, 202/417/300-bus", COLORS["white"], fontsize=4.8, lw=0.7, radius=0.035)
    box(ax, 8.4, 0.52, 2.7, 0.52, "Operating rule\nfast posterior or defer", COLORS["white"], fontsize=4.9, lw=0.7, radius=0.035)
    ax.plot([0.35, 11.5], [0.55, 0.55], color=COLORS["line"], lw=0.8, alpha=0.7)

    metadata = {"Creator": "matplotlib", "Producer": "matplotlib"}
    for name in ["graphical_abstract_ijepes", "涓€鍖烘帰绱graphical_abstract", "ijepes_graphical_abstract"]:
        base = ROOT / name
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main():
    figure_workflow()
    figure_principle()
    figure_benchmark()
    figure_graphical_abstract()


if __name__ == "__main__":
    main()
