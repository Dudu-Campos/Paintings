"""Log-log plot of the PCA-component LDA curves for subsets 1, 3, 6 with HOG
folded into Descriptors -- solid lines only."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
HOG_DIR = ROOT / "outputs/lda_pca_curve_by_subset_136_hog"
OUTPUT_PNG = ROOT / "results/lda_pca_curve_hog_only.png"

CATEGORICAL = {1: "#2a78d6", 3: "#1baf7a", 6: "#4a3aa7"}
LABELS = {1: "1: Descriptors + HOG", 3: "3: + SLIC", 6: "6: + Hilbert"}
GRID = "#d8d7d0"
TEXT_SECONDARY = "#52514e"


def main() -> None:
    hog = pd.read_csv(HOG_DIR / "curve_summary.csv")

    figure, axis = plt.subplots(figsize=(11, 7.5))
    for n_representations, color in CATEGORICAL.items():
        rows = hog[hog.n_representations.eq(n_representations)].sort_values("k")
        axis.plot(
            rows["k"], rows["bacc_mean"], "o-", color=color, linewidth=2.3, markersize=6.5,
            label=LABELS[n_representations], zorder=3,
        )
        axis.fill_between(
            rows["k"], rows["bacc_mean"] - rows["bacc_std"], rows["bacc_mean"] + rows["bacc_std"],
            color=color, alpha=0.10, zorder=2,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("k (componentes PCA, escala log)")
    axis.set_ylabel("Balanced accuracy (escala log)")
    axis.set_title("WikiArt (27 estilos, LDA) — PCA em Descriptors + HOG")
    axis.grid(True, which="both", color=GRID, alpha=0.5, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(
        loc="lower right", frameon=False, fontsize=9.5,
        title="Representações combinadas", title_fontsize=9.5,
    )
    legend.get_title().set_color(TEXT_SECONDARY)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
