"""Compare subsets 1, 4, 8 with vs without HOG folded into Descriptors."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OLD_DIR = ROOT / "outputs/lda_topk_curve_by_subset"
NEW_DIR = ROOT / "outputs/lda_topk_curve_subset148_hog"
OUTPUT_PNG = ROOT / "results/lda_topk_curve_hog_comparison_loglog.png"

CATEGORICAL = {1: "#2a78d6", 4: "#eda100", 8: "#e34948"}
LABELS = {1: "1: Descriptors", 4: "4: + Text", 8: "8: + Multiscale (tudo)"}
GRID = "#d8d7d0"
TEXT_SECONDARY = "#52514e"


def main() -> None:
    new = pd.read_csv(NEW_DIR / "curve_summary.csv")

    figure, axis = plt.subplots(figsize=(11, 7.5))
    for n_representations, color in CATEGORICAL.items():
        rows = new[new.n_representations.eq(n_representations)].sort_values("k")
        axis.plot(
            rows["k"], rows["bacc_mean"], "o-", color=color, linewidth=2.3, markersize=6.5,
            label=f"{LABELS[n_representations]} (+ HOG em Descriptors)", zorder=3,
        )
        axis.fill_between(
            rows["k"], rows["bacc_mean"] - rows["bacc_std"], rows["bacc_mean"] + rows["bacc_std"],
            color=color, alpha=0.10, zorder=2,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("k (features selecionadas, escala log)")
    axis.set_ylabel("Balanced accuracy (escala log)")
    axis.set_title(
        "WikiArt (27 estilos, LDA) — HOG dentro de Descriptors"
    )
    axis.grid(True, which="both", color=GRID, alpha=0.5, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(
        loc="lower right", frameon=False, fontsize=9.5,
        title="Representações combinadas",
        title_fontsize=9.5,
    )
    legend.get_title().set_color(TEXT_SECONDARY)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
