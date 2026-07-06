import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "fig8_experiment_landscape"

COLORS = {
    "P": "#244A73",
    "S": "#5E9B96",
    "B": "#C79118",
    "C": "#8A8F97",
    "bg": "#F8FAFD",
    "grid": "#D9E0E8",
    "ink": "#232323",
    "muted": "#66707A",
}

ROWS = [
    "33/69/119 exact audit",
    "GraphSAGE + edge-GNN",
    "Random/structured missing",
    "Noise-compounded fallback",
    "Posterior-aware BOED",
    "Posterior calibration",
    "SoCal public replay",
    "IEEE123 3-phase stress",
    "202/417-bus dropout",
    "300-bus boundary",
    "AIS/SMC controls",
    "DSSE downstream value",
]

COLS = [
    "Exact\nref.",
    "Baseline\nctrl.",
    "Missing\nnoise",
    "Posterior\nquality",
    "Decision\nvalue",
    "Realism\nscale",
    "Bounded\nrisk",
]

MATRIX = [
    ["P", "", "", "S", "", "", "B"],
    ["", "P", "", "", "", "", "C"],
    ["", "S", "P", "", "", "", "B"],
    ["S", "", "P", "", "S", "S", "B"],
    ["", "S", "", "S", "P", "", "B"],
    ["P", "", "", "P", "", "", "B"],
    ["", "S", "", "", "", "P", "B"],
    ["P", "", "S", "", "S", "P", "B"],
    ["", "", "P", "", "S", "P", "B"],
    ["P", "", "P", "", "S", "S", "B"],
    ["S", "P", "", "S", "", "", "C"],
    ["", "", "", "S", "P", "", "B"],
]

HEADLINES = [
    "NRE 68.58/36.05/41.90%; speedup 1176x-3762x",
    "69-bus: NRE 36.05 vs edge-GNN 34.35 vs GraphSAGE 29.27",
    "30% missing gains: +50.91 pp and +23.26 pp",
    "119-bus rerank@20 tracks exact under 2x/3x noise",
    "119-bus AUC: BOED 54.60 vs MVG 41.75",
    "KL 0.0453, ECE 0.0181, coverage90 97.62%",
    "Blocked CV accuracy 86.64%; majority 36.10%",
    "Rerank@20 79.28 vs exact 79.22 at composite stress",
    "40% dropout top20: 91.04 and 80.69",
    "Direct top-1 boundary, rerank@20 restores exact-level MAP",
    "AIS KL 0.0082; SMC close to exact on 33/69-bus",
    "Unobserved-bus MAE gain up to 5.37%",
]

LEGEND = [
    ("P", "primary"),
    ("S", "supporting"),
    ("B", "bounded risk"),
    ("C", "control"),
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": COLORS["ink"],
        }
    )


def save_all(fig) -> None:
    for suffix, dpi in ((".svg", None), ((".pdf", None)), ((".png", 600))):
        fig.savefig(
            OUT_STEM.with_suffix(suffix),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 5.75))
    ax.set_xlim(-3.65, len(COLS) + 6.25)
    ax.set_ylim(-1.45, len(ROWS) + 2.05)
    ax.axis("off")

    ax.text(
        -3.55,
        len(ROWS) + 1.72,
        "Experiment landscape",
        fontsize=12.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        -3.55,
        len(ROWS) + 1.26,
        "Visual index of which evidence line supports each manuscript claim.",
        fontsize=7.8,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )

    for j, col in enumerate(COLS):
        ax.text(j + 0.5, len(ROWS) + 0.12, col, ha="center", va="bottom", fontsize=6.9, fontweight="bold")

    for i, row in enumerate(ROWS):
        y = len(ROWS) - i - 1
        fill = COLORS["bg"] if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((-3.55, y - 0.03), len(COLS) + 9.45, 0.86, facecolor=fill, edgecolor="none"))
        ax.text(-3.45, y + 0.38, row, ha="left", va="center", fontsize=7.1, fontweight="semibold")
        ax.text(len(COLS) + 0.35, y + 0.38, HEADLINES[i], ha="left", va="center", fontsize=6.7, color=COLORS["muted"])

        for j, code in enumerate(MATRIX[i]):
            ax.add_patch(
                Rectangle(
                    (j + 0.08, y + 0.07),
                    0.84,
                    0.62,
                    facecolor="white",
                    edgecolor=COLORS["grid"],
                    linewidth=0.6,
                )
            )
            if code:
                ax.add_patch(
                    Rectangle(
                        (j + 0.21, y + 0.18),
                        0.58,
                        0.40,
                        facecolor=COLORS[code],
                        edgecolor="none",
                        linewidth=0.0,
                    )
                )
                ax.text(j + 0.5, y + 0.38, code, ha="center", va="center", fontsize=7.4, color="white", fontweight="bold")

    ax.text(len(COLS) + 0.35, len(ROWS) + 0.10, "headline result or role", ha="left", va="bottom", fontsize=7.2, fontweight="bold")

    lx = -3.55
    ly = -0.92
    for k, (code, label) in enumerate(LEGEND):
        x = lx + k * 2.25
        ax.add_patch(Rectangle((x, ly), 0.28, 0.18, facecolor=COLORS[code], edgecolor="none"))
        ax.text(x + 0.36, ly + 0.09, f"{code}: {label}", ha="left", va="center", fontsize=6.5, color=COLORS["muted"])

    ax.text(
        len(COLS) + 0.35,
        -0.92,
        "This figure is a visual index, not an additional metric.",
        ha="left",
        va="center",
        fontsize=6.5,
        color=COLORS["muted"],
    )

    save_all(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
