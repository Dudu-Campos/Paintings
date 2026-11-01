"""Unsupervised pairwise redundancy audit of canonical visual features.

No class labels enter any statistic.  For each pair of non-constant features,
we compute Pearson's r and r².  With an intercept, this r² is exactly the
coefficient of determination of a univariate linear regression.  The analysis
uses the canonical 79,207-image cohort and excludes all 16 sentiment/text
features, leaving 71 SLIC multigraph + 130 classical descriptor + 57 Hilbert
features.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

from slic_oof_stacking import load_dataset  # noqa: E402


OUT_PREFIX = PAPER / "visual_feature_redundancy_r2"
THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99, 0.999)
CLUSTER_THRESHOLDS = (0.90, 0.95, 0.99)
CONSTANT_TOLERANCE = 1e-12


def descriptor_name(index: int) -> tuple[str, str]:
    if index < 90:
        channel = ("L", "a", "b")[index // 30]
        return f"{channel}_hist_{index % 30}", f"Lab histogram {channel}"
    if index < 99:
        offset = index - 90
        channel = ("L", "a", "b")[offset // 3]
        statistic = ("mean", "std", "skew")[offset % 3]
        return f"{channel}_{statistic}", "Lab statistics"
    if index < 117:
        return f"lbp_{index - 99}", "LBP"
    haralick = [
        "contrast_mean", "contrast_std",
        "dissimilarity_mean", "dissimilarity_std",
        "homogeneity_mean", "homogeneity_std",
        "energy_mean", "energy_std",
        "correlation_mean", "correlation_std",
        "ASM_mean", "ASM_std", "correlation_max",
    ]
    return haralick[index - 117], "Haralick/GLCM"


def feature_metadata(columns: list[str], groups: dict[str, list[str]]) -> pd.DataFrame:
    main_group = {
        feature: group for group, features in groups.items() for feature in features
    }
    rows = []
    for feature in columns:
        group = main_group[feature]
        display = feature
        subgroup = group
        if group == "Descriptors":
            display, subgroup = descriptor_name(int(feature.rsplit("_", 1)[1]))
        elif group == "SLIC multigraph":
            if feature.startswith("knn_"):
                subgroup = "SLIC kNN"
            elif feature.startswith("del_"):
                subgroup = "SLIC Delaunay"
            elif feature.startswith("mst_"):
                subgroup = "SLIC MST"
            else:
                subgroup = "SLIC RAG/global"
        rows.append(
            {
                "feature": feature,
                "display_name": display,
                "group": group,
                "subgroup": subgroup,
            }
        )
    return pd.DataFrame(rows)


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def redundancy_clusters(
    columns: list[str], r2: np.ndarray, thresholds: tuple[float, ...]
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        sets = DisjointSet(len(columns))
        left, right = np.where(np.triu(r2 >= threshold, k=1))
        for a, b in zip(left.tolist(), right.tolist()):
            sets.union(a, b)
        members = defaultdict(list)
        for index, feature in enumerate(columns):
            members[sets.find(index)].append(feature)
        clusters = sorted(
            (items for items in members.values() if len(items) > 1),
            key=lambda items: (-len(items), items),
        )
        for cluster_index, items in enumerate(clusters, 1):
            indices = [columns.index(item) for item in items]
            block = r2[np.ix_(indices, indices)]
            values = block[np.triu_indices(len(indices), 1)]
            rows.append(
                {
                    "threshold": threshold,
                    "cluster_id": cluster_index,
                    "size": len(items),
                    "min_within_cluster_r2": float(values.min()),
                    "max_within_cluster_r2": float(values.max()),
                    "members": " | ".join(items),
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap(
    r2: np.ndarray,
    columns: list[str],
    metadata: pd.DataFrame,
    output: Path,
) -> list[int]:
    distance = 1.0 - np.sqrt(np.clip(r2, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    ordered = r2[np.ix_(order, order)]
    ordered_groups = metadata.iloc[order]["group"].to_numpy()

    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(ordered, vmin=0, vmax=1, cmap="magma", interpolation="nearest")
    boundaries = np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 0.5
    for boundary in boundaries:
        axis.axhline(boundary, color="white", alpha=0.28, linewidth=0.4)
        axis.axvline(boundary, color="white", alpha=0.28, linewidth=0.4)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Features ordenadas por clustering de |r|")
    axis.set_ylabel("Features ordenadas por clustering de |r|")
    axis.set_title(
        "Redundância linear das 256 features visuais não constantes — r² pareado\n"
        "79.207 imagens; nenhuma informação de classe"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("r²")
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)
    return order.tolist()


def plot_top_pairs(pairs: pd.DataFrame, output: Path, count: int = 35) -> None:
    top = pairs.head(count).iloc[::-1]
    labels = [
        f"{a}  ↔  {b}" for a, b in zip(top["display_a"], top["display_b"])
    ]
    colors = ["#d73027" if value < 0 else "#2878b5" for value in top["pearson_r"]]
    figure, axis = plt.subplots(figsize=(13, 12))
    bars = axis.barh(np.arange(len(top)), top["r2"], color=colors, alpha=0.88)
    axis.set_yticks(np.arange(len(top)), labels, fontsize=8)
    axis.set_xlim(max(0, float(top["r2"].min()) - 0.02), 1.005)
    axis.set_xlabel("r² pareado")
    axis.set_title("Pares visuais mais redundantes (azul: r positivo; vermelho: r negativo)")
    axis.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, top["pearson_r"]):
        axis.text(
            min(1.001, bar.get_width() + 0.001),
            bar.get_y() + bar.get_height() / 2,
            f"r={value:+.4f}",
            va="center",
            fontsize=7,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    frame, all_groups = load_dataset()
    groups = {name: values for name, values in all_groups.items() if name != "Sentiment"}
    columns = [feature for features in groups.values() for feature in features]
    if len(columns) != 258 or len(columns) != len(set(columns)):
        raise RuntimeError(f"Expected 258 unique visual features, found {len(columns)}")
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Canonical visual matrix contains NaN or infinity")

    metadata = feature_metadata(columns, groups)
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0)
    constant_mask = standard_deviations < CONSTANT_TOLERANCE
    constants = metadata[constant_mask].copy()
    constants["mean"] = means[constant_mask]
    constants["std"] = standard_deviations[constant_mask]
    constants.to_csv(f"{OUT_PREFIX}_constant_features.csv", index=False)

    active_indices = np.flatnonzero(~constant_mask)
    active_columns = [columns[index] for index in active_indices]
    active_metadata = metadata.iloc[active_indices].reset_index(drop=True)
    standardized = (
        values[:, active_indices] - means[active_indices]
    ) / standard_deviations[active_indices]
    correlation = (standardized.T @ standardized) / len(standardized)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    r2 = correlation ** 2

    left, right = np.triu_indices(len(active_columns), k=1)
    lookup = active_metadata.set_index("feature")
    pairs = pd.DataFrame(
        {
            "feature_a": np.asarray(active_columns, dtype=object)[left],
            "feature_b": np.asarray(active_columns, dtype=object)[right],
            "pearson_r": correlation[left, right],
            "r2": r2[left, right],
        }
    )
    pairs["display_a"] = pairs["feature_a"].map(lookup["display_name"])
    pairs["display_b"] = pairs["feature_b"].map(lookup["display_name"])
    pairs["group_a"] = pairs["feature_a"].map(lookup["group"])
    pairs["group_b"] = pairs["feature_b"].map(lookup["group"])
    pairs["subgroup_a"] = pairs["feature_a"].map(lookup["subgroup"])
    pairs["subgroup_b"] = pairs["feature_b"].map(lookup["subgroup"])
    pairs["same_group"] = pairs["group_a"] == pairs["group_b"]
    pairs = pairs.sort_values(["r2", "feature_a", "feature_b"], ascending=[False, True, True])
    pairs.to_csv(f"{OUT_PREFIX}_pairs.csv", index=False)
    pairs[pairs["r2"] >= 0.80].to_csv(
        f"{OUT_PREFIX}_high_pairs.csv", index=False
    )

    feature_rows = []
    for index, feature in enumerate(active_columns):
        other = r2[index].copy()
        other[index] = -1
        partner_index = int(np.argmax(other))
        row = active_metadata.iloc[index].to_dict()
        row.update(
            {
                "mean": means[active_indices[index]],
                "std": standard_deviations[active_indices[index]],
                "max_r2": float(other[partner_index]),
                "max_abs_r": float(abs(correlation[index, partner_index])),
                "best_partner": active_columns[partner_index],
                "best_partner_display": active_metadata.iloc[partner_index]["display_name"],
                **{
                    f"partners_r2_ge_{str(threshold).replace('.', '_')}": int(
                        np.sum(other >= threshold)
                    )
                    for threshold in THRESHOLDS
                },
            }
        )
        feature_rows.append(row)
    feature_summary = pd.DataFrame(feature_rows).sort_values("max_r2", ascending=False)
    feature_summary.to_csv(f"{OUT_PREFIX}_feature_summary.csv", index=False)

    group_rows = []
    normalized_groups = sorted(set(active_metadata["group"]))
    for group_index, group_a in enumerate(normalized_groups):
        indices_a = np.flatnonzero(active_metadata["group"].to_numpy() == group_a)
        for group_b in normalized_groups[group_index:]:
            indices_b = np.flatnonzero(active_metadata["group"].to_numpy() == group_b)
            if group_a == group_b:
                a, b = np.triu_indices(len(indices_a), 1)
                block_values = r2[indices_a[a], indices_a[b]]
            else:
                block_values = r2[np.ix_(indices_a, indices_b)].ravel()
            group_rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_pairs": len(block_values),
                    "mean_r2": float(block_values.mean()),
                    "median_r2": float(np.median(block_values)),
                    "p95_r2": float(np.quantile(block_values, 0.95)),
                    "max_r2": float(block_values.max()),
                    **{
                        f"pairs_r2_ge_{str(threshold).replace('.', '_')}": int(
                            np.sum(block_values >= threshold)
                        )
                        for threshold in THRESHOLDS
                    },
                }
            )
    pd.DataFrame(group_rows).to_csv(f"{OUT_PREFIX}_group_summary.csv", index=False)

    clusters = redundancy_clusters(active_columns, r2, CLUSTER_THRESHOLDS)
    clusters.to_csv(f"{OUT_PREFIX}_clusters.csv", index=False)
    order = plot_heatmap(
        r2,
        active_columns,
        active_metadata,
        Path(f"{OUT_PREFIX}_heatmap.png"),
    )
    ordered_metadata = active_metadata.iloc[order].copy()
    ordered_metadata.insert(0, "heatmap_position", np.arange(len(order)))
    ordered_metadata.to_csv(f"{OUT_PREFIX}_heatmap_order.csv", index=False)
    plot_top_pairs(pairs, Path(f"{OUT_PREFIX}_top_pairs.png"))

    pair_values = pairs["r2"].to_numpy()
    summary = {
        "method": "pairwise Pearson r squared; no class labels used",
        "cohort_images": len(frame),
        "input_visual_features": len(columns),
        "nonconstant_features": len(active_columns),
        "constant_features": constants["feature"].tolist(),
        "analyzed_pairs": len(pairs),
        "r2_distribution": {
            "mean": float(pair_values.mean()),
            "median": float(np.median(pair_values)),
            "p90": float(np.quantile(pair_values, 0.90)),
            "p95": float(np.quantile(pair_values, 0.95)),
            "p99": float(np.quantile(pair_values, 0.99)),
            "maximum": float(pair_values.max()),
        },
        "threshold_counts": {
            str(threshold): int(np.sum(pair_values >= threshold))
            for threshold in THRESHOLDS
        },
        "features_with_partner_above_threshold": {
            str(threshold): int(np.sum(feature_summary["max_r2"] >= threshold))
            for threshold in THRESHOLDS
        },
        "connected_clusters": {
            str(threshold): int(np.sum(clusters["threshold"] == threshold))
            for threshold in CLUSTER_THRESHOLDS
        },
        "top_pairs": pairs.head(20)[
            [
                "feature_a", "feature_b", "display_a", "display_b",
                "group_a", "group_b", "pearson_r", "r2",
            ]
        ].to_dict(orient="records"),
    }
    Path(f"{OUT_PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
