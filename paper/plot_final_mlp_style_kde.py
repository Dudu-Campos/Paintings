"""Plot the distribution of per-style OOF performance for the final MLP.

Styles are categorical, so a KDE over their names would not be meaningful.
Instead this plot estimates the density of the 27 class-wise recalls: each
point is the OOF recall of one WikiArt style.  The accompanying CSV preserves
the values behind the smoothed curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("Visual GIST", "Visual GIST + TF-IDF")
COLORS = {"Visual GIST": "#2a78d6", "Visual GIST + TF-IDF": "#e8782b"}


def class_metrics(predictions: pd.DataFrame, condition: str) -> pd.DataFrame:
    subset = predictions.loc[predictions["condition"].eq(condition)].copy()
    if subset["stem"].duplicated().any():
        raise ValueError(f"{condition} has repeated OOF predictions")
    labels = np.sort(subset["true_style"].unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        subset["true_style"], subset["predicted_style"], labels=labels, zero_division=0
    )
    return pd.DataFrame({
        "condition": condition,
        "style": labels,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


def bounded_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE with reflection at 0 and 1 to avoid boundary leakage."""
    # Estimate bandwidth from the original 27 values.  Estimating it after
    # reflection would make the artificial copies near -x and 2-x inflate it.
    bandwidth = max(0.04, values.std(ddof=1) * len(values) ** (-1 / 5))
    distance = lambda locations: (grid[:, None] - locations[None, :]) / bandwidth
    kernel = lambda locations: np.exp(-0.5 * distance(locations) ** 2)
    normalizer = len(values) * bandwidth * np.sqrt(2 * np.pi)
    return (kernel(values).sum(axis=1) + kernel(-values).sum(axis=1)
            + kernel(2 - values).sum(axis=1)) / normalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp/predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.input, dtype=str)
    missing = set(CONDITIONS) - set(predictions["condition"])
    if missing:
        raise ValueError(f"Missing conditions: {sorted(missing)}")
    metrics = pd.concat([class_metrics(predictions, condition) for condition in CONDITIONS])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "per_style_oof_metrics.csv", index=False, float_format="%.6f")

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    grid = np.linspace(0, 1, 500)
    for level, condition in enumerate(CONDITIONS):
        values = metrics.loc[metrics.condition.eq(condition), "recall"].to_numpy()
        density = bounded_kde(values, grid)
        color = COLORS[condition]
        axis.plot(grid, density, color=color, linewidth=2.4, label=condition)
        axis.fill_between(grid, density, color=color, alpha=0.16)
        axis.scatter(
            values, np.full(len(values), -0.08 - level * 0.08), s=25,
            color=color, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3,
        )
        axis.axvline(values.mean(), color=color, linestyle="--", linewidth=1.2, alpha=0.9)

    axis.set(
        xlim=(0, 1), ylim=(-0.23, None), xlabel="Recall OOF por estilo",
        ylabel="Densidade", title="Distribuição do desempenho entre estilos (27 classes)",
    )
    axis.text(0.01, -0.205, "Pontos: um estilo; linhas tracejadas: média", fontsize=9, color="#555555")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "per_style_recall_kde.png", dpi=180)
    print(metrics.groupby("condition")["recall"].agg(["mean", "std", "min", "max"]).to_string())
    print(f"Saved {args.output_dir / 'per_style_recall_kde.png'}")


if __name__ == "__main__":
    main()
