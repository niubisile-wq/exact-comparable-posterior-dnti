# -*- coding: utf-8 -*-
"""Plot Step 5 synthetic 300-bus scalability evidence."""
import os

import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
PALETTE = {
    "navy": "#355C7D",
    "green": "#5B8E2D",
    "gold": "#C48A1D",
    "red": "#C65A46",
    "gray": "#7A7A7A",
}


def read_ip4():
    ks, hs, ss = [], [], []
    with open(os.path.join(SAVE_DIR, "ip4_300bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3 and parts[0].isdigit():
                ks.append(int(parts[0]))
                hs.append(float(parts[1]))
                ss.append(float(parts[2]))
    return np.array(ks), np.array(hs), np.array(ss)


def read_ip1():
    acc, exact, speed = [], [], []
    with open(os.path.join(SAVE_DIR, "ip1_300bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 9 and parts[0].isdigit():
                acc.append(float(parts[1]))
                exact.append(float(parts[4]))
                speed.append(float(parts[8]))
    return np.array(acc), np.array(exact), np.array(speed)


def read_ipc():
    acc30, exact30 = [], []
    with open(os.path.join(SAVE_DIR, "ipc_300bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 11 and parts[0].isdigit():
                acc30.append(float(parts[4]))
                exact30.append(float(parts[6]))
    return np.array(acc30), np.array(exact30)


def main():
    ks, hs, ss = read_ip4()
    acc, exact, speed = read_ip1()
    acc30, exact30 = read_ipc()

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#fbfbfa",
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), constrained_layout=True)

    axes[0].errorbar(ks, hs, yerr=ss, marker="o", linewidth=1.8, capsize=3, color=PALETTE["navy"])
    axes[0].set_xlabel("Installed sensors K")
    axes[0].set_ylabel("Posterior entropy H(K)")
    axes[0].set_title("Identifiability")
    axes[0].grid(alpha=0.25)

    labels = ["Exact", "NRE"]
    clean_vals = [float(np.mean(exact)), float(np.mean(acc))]
    clean_err = [float(np.std(exact)), float(np.std(acc))]
    axes[1].bar(labels, clean_vals, yerr=clean_err, capsize=4, color=[PALETTE["gray"], PALETTE["navy"]])
    axes[1].set_ylim(0, 0.65)
    axes[1].set_ylabel("Top-1 accuracy")
    axes[1].set_title("300-bus clean deployment")
    axes[1].grid(axis="y", alpha=0.25)

    x = np.arange(2)
    miss_vals = [float(np.mean(exact30)), float(np.mean(acc30))]
    miss_err = [float(np.std(exact30)), float(np.std(acc30))]
    axes[2].bar(x, miss_vals, yerr=miss_err, capsize=4, color=[PALETTE["gray"], PALETTE["red"]])
    axes[2].set_xticks(x, ["Exact", "NRE"])
    axes[2].set_ylim(0, 0.55)
    axes[2].set_ylabel("Top-1 accuracy")
    axes[2].set_title(f"30% missing boundary; speedup {np.mean(speed):.1f}x")
    axes[2].grid(axis="y", alpha=0.25)

    out = os.path.join(SAVE_DIR, "fig_scalability.png")
    fig.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
