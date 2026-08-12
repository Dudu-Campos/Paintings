"""Class-conditional distributions of the SLIC multigraph feature family.

Seventy-one variables across 27 styles cannot be read in one panel.  This
script ranks graph features by one-way ANOVA across styles, then plots the top
ones as class-wise boxplots after global z-normalisation.  A long CSV with
count, mean, standard deviation and quantiles for *all* graph features is
saved alongside the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/wikiart_slic_features/combined_features.parquet"
OUTPUT_PNG = ROOT / "paper/graph_feature_distributions_by_style.png"
OUTPUT_RANKING = ROOT / "paper/graph_feature_anova_ranking.csv"
OUTPUT_SUMMARY = ROOT / "paper/graph_feature_class_quantiles.csv"
N_FEATURES_TO_PLOT = 12


METADATA = {
    "stem", "style", "segmentation", "reference_n_nodes", "slic_requested_segments",
    "slic_actual_segments", "slic_compactness", "slic_sigma", "slic_max_side",
    "slic_enforce_connectivity", "elapsed_seconds", "elapsed_seconds_all_graphs",
}


def family(name: str) -> str:
    if name.startswith("knn_"):
        return "k-NN"
    if name.startswith("del_"):
        return "Delaunay"
    if name.startswith("mst_"):
        return "MST"
    return "RAG"


def display(name: str) -> str:
    return name.replace("knn_", "k-NN · ").replace("del_", "Delaunay · ").replace("mst_", "MST · ").replace("_", " ")


def main() -> None:
    frame = pd.read_parquet(INPUT)
    columns = [column for column in frame if column not in METADATA]
    if len(columns) != 71:
        raise ValueError(f"Expected 71 graph features, found {len(columns)}")
    values = frame[columns].replace([np.inf, -np.inf], np.nan)
    keep = ~values.isna().any(axis=1)
    if not keep.all():
        print(f"Dropping {(~keep).sum():,} incomplete rows")
        frame, values = frame.loc[keep].copy(), values.loc[keep].copy()
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["style"].to_numpy(dtype=str))
    variable_columns = [column for column in columns if values[column].nunique() > 1]
    scores, pvalues = f_classif(values[variable_columns].to_numpy(dtype=np.float64), labels)
    ranking = pd.DataFrame({
        "feature": variable_columns, "family": [family(column) for column in variable_columns],
        "anova_f": scores, "anova_pvalue": pvalues,
    }).sort_values("anova_f", ascending=False, ignore_index=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(OUTPUT_RANKING, index=False, float_format="%.8g")

    summary = (
        frame[["style"] + columns]
        .melt(id_vars="style", var_name="feature", value_name="value")
        .groupby(["feature", "style"], as_index=False)["value"]
        .agg(n="count", mean="mean", std="std", q05=lambda x: x.quantile(.05),
             q25=lambda x: x.quantile(.25), median="median", q75=lambda x: x.quantile(.75),
             q95=lambda x: x.quantile(.95))
    )
    summary = summary.merge(ranking[["feature", "rank", "family"]], on="feature", how="left")
    summary.to_csv(OUTPUT_SUMMARY, index=False, float_format="%.8g")

    top = ranking.head(N_FEATURES_TO_PLOT).feature.tolist()
    styles = sorted(frame["style"].unique())
    standardized = (values - values.mean()) / values.std(ddof=0).replace(0, 1)
    colors = {"RAG": "#2a78c8", "k-NN": "#68a9d6", "Delaunay": "#7353a8", "MST": "#e57a44"}
    figure, axes = plt.subplots(4, 3, figsize=(20, 17), sharex=True)
    for axis, feature in zip(axes.flat, top, strict=True):
        groups = [standardized.loc[frame["style"].eq(style), feature].to_numpy() for style in styles]
        box = axis.boxplot(groups, patch_artist=True, showfliers=False, widths=.62,
                           medianprops={"color": "#202020", "linewidth": 1.0})
        color = colors[family(feature)]
        for patch in box["boxes"]:
            patch.set(facecolor=color, alpha=.72, edgecolor="#333333", linewidth=.55)
        axis.axhline(0, color="#777777", linewidth=.75, linestyle="--", zorder=0)
        rank = ranking.index[ranking.feature.eq(feature)][0] + 1
        axis.set_title(f"#{rank}  {display(feature)}", loc="left", fontsize=11, color=color, fontweight="bold")
        axis.set_ylabel("z global")
        axis.grid(axis="y", alpha=.20)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat:
        axis.set_xticks(np.arange(1, len(styles) + 1), styles, rotation=90, fontsize=6.4)
    figure.suptitle(
        "Distribuições dos atributos de grafo SLIC por estilo\n"
        "Top-12 por separação ANOVA; boxplots sem outliers, normalização z global por atributo",
        fontsize=16, y=.995,
    )
    figure.text(.5, .008, f"{len(frame):,} pinturas · 27 estilos · atributos RAG, k-NN, Delaunay e MST", ha="center", color="#555555")
    figure.tight_layout(rect=(0, .025, 1, .965))
    figure.savefig(OUTPUT_PNG, dpi=190)
    print(ranking.head(N_FEATURES_TO_PLOT).to_string(index=False))
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
