"""Compare PCA-component curves (unsupervised, variance-ordered) against
top-k feature curves (supervised, structure-coefficient-ordered) for the
same three nested subsets."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TOPK_DIR = ROOT / "outputs/lda_topk_curve_by_subset"
PCA_DIR = ROOT / "outputs/lda_pca_curve_by_subset"
OUTPUT_PNG = ROOT / "results/lda_pca_vs_topk_curve.png"

CATEGORICAL = {1: "#2a78d6", 4: "#eda100", 8: "#e34948"}
LABELS = {1: "1: Descriptors", 4: "4: + Text", 8: "8: + Multiscale (tudo)"}
GRID = "#d8d7d0"
TEXT_SECONDARY = "#52514e"


def main() -> None:
    topk = pd.read_csv(TOPK_DIR / "curve_summary.csv")
    topk = topk[topk["n_representations"].isin(CATEGORICAL)]
    pca = pd.read_csv(PCA_DIR / "curve_summary.csv")

    figure, axis = plt.subplots(figsize=(11, 7.5))
    for n_representations, color in CATEGORICAL.items():
        rows = topk[topk.n_representations.eq(n_representations)].sort_values("k")
        axis.plot(
            rows["k"], rows["bacc_mean"], "o-", color=color, linewidth=2.3, markersize=6.5,
            label=f"{LABELS[n_representations]} — top-k features (supervisionado)", zorder=3,
        )
        axis.fill_between(
            rows["k"], rows["bacc_mean"] - rows["bacc_std"], rows["bacc_mean"] + rows["bacc_std"],
            color=color, alpha=0.10, zorder=2,
        )

        rows = pca[pca.n_representations.eq(n_representations)].sort_values("k")
        axis.plot(
            rows["k"], rows["bacc_mean"], "o--", color=color, linewidth=2.0, markersize=5.5,
            alpha=0.75, label=f"{LABELS[n_representations]} — componentes PCA (não-supervisionado)",
            zorder=3,
        )
        axis.fill_between(
            rows["k"], rows["bacc_mean"] - rows["bacc_std"], rows["bacc_mean"] + rows["bacc_std"],
            color=color, alpha=0.06, zorder=1,
        )

    axis.set_xscale("log")
    axis.set_xlabel("k (features ou componentes, escala log)")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title(
        "WikiArt (27 estilos, LDA) — top-k supervisionado vs. componentes PCA"
    )
    axis.grid(True, which="both", color=GRID, alpha=0.5, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(
        loc="lower right", frameon=False, fontsize=8.5,
        title="Linha cheia = top-k supervisionado, tracejada = componentes PCA",
        title_fontsize=8.5,
    )
    legend.get_title().set_color(TEXT_SECONDARY)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
