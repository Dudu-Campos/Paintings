"""Visual-only cumulative LDA curve comparing Fourier v1 and Fourier-v2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))

from registry import feature_columns, load_blocks, log, sanitize  # noqa: E402
from view_names import display  # noqa: E402

SEED = 42
PREFIX = ["Felzenszwalb", "SLIC", "Descriptors", "L(G_felz)",
          "Line graph", "Gabor", "Hilbert"]
ORDER_V1 = PREFIX + ["Fourier"]
ORDER_V2 = PREFIX + ["Fourier v2"]
HAND8 = ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
         "Hilbert", "Fourier", "Gabor"]


def load_fourier_v2(path: Path, stems: np.ndarray) -> np.ndarray:
    table = pd.read_parquet(path)
    columns = feature_columns(table)
    index = pd.Series(np.arange(len(table)), index=table["stem"].astype(str)).reindex(stems)
    if index.isna().any():
        raise ValueError(f"Fourier-v2: {int(index.isna().sum())} stems missing")
    return sanitize(table.iloc[index.to_numpy(dtype=int)][columns].to_numpy(np.float32),
                    "Fourier-v2")


def evaluate(blocks: dict[str, np.ndarray], order: list[str], labels: np.ndarray,
             version: str, folds: int) -> pd.DataFrame:
    n_classes = len(np.unique(labels))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:folds]
    rows = []
    for fold, (train, test) in enumerate(splits, start=1):
        scaled = {
            name: StandardScaler().fit(blocks[name][train]).transform(blocks[name]).astype(np.float32)
            for name in order
        }
        for k in range(1, len(order) + 1):
            names = order[:k]
            values = np.hstack([scaled[n] for n in names])
            model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            model.fit(values[train], labels[train])
            pred = model.predict(values[test])
            bacc = 100 * balanced_accuracy_score(labels[test], pred)
            rows.append({
                "version": version,
                "k": k,
                "fold": fold,
                "added": names[-1],
                "blocks": "|".join(names),
                "dimensions": values.shape[1],
                "bacc": bacc,
            })
            log(f"{version:<10} fold {fold} k={k}: BAcc={bacc:.2f} ({values.shape[1]}d)")
    return pd.DataFrame(rows)


def plot(frame: pd.DataFrame, output: Path) -> None:
    surface, grid, ink, muted = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
    colours = {"Fourier v1": "#7c3fa8", "Fourier v2": "#eb6834"}
    markers = {"Fourier v1": "s", "Fourier v2": "o"}
    fig, ax = plt.subplots(figsize=(8.8, 4.9), facecolor="white")
    ax.set_facecolor(surface)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for version in ["Fourier v1", "Fourier v2"]:
        sub = frame[frame.version.eq(version)]
        mean = sub.groupby("k").bacc.mean()
        sd = sub.groupby("k").bacc.std(ddof=1)
        ax.fill_between(mean.index, mean - sd, mean + sd,
                        color=colours[version], alpha=0.13, linewidth=0)
        ax.plot(mean.index, mean, color=colours[version], marker=markers[version],
                markersize=6.2, linewidth=2.2, markeredgecolor="white",
                markeredgewidth=0.9, label=version)
    names = frame[frame.version.eq("Fourier v2")].drop_duplicates("k").set_index("k").added
    ax.set_xticks(sorted(names.index))
    ax.set_xticklabels([display(v.replace(" v2", ""), "en") for v in names],
                       rotation=28, ha="right", color=muted)
    ax.set_ylabel("balanced accuracy (%)", color=ink)
    ax.set_title("Cumulative visual LDA with Fourier-v2", loc="left",
                 fontweight="bold", fontsize=13.5, color=ink)
    ax.legend(frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor="white")


def summarize(frame: pd.DataFrame) -> None:
    print("\nMean BAcc")
    print(frame.groupby(["version", "k", "added"]).bacc.agg(["mean", "std"]).round(3).to_string())
    pivot = frame.pivot_table(index=["fold", "k"], columns="version", values="bacc")
    both = pivot.dropna()
    full = both.xs(8, level="k")
    delta = full["Fourier v2"] - full["Fourier v1"]
    _, p = ttest_rel(full["Fourier v2"], full["Fourier v1"])
    print(f"\nFinal step v2 - v1: {delta.mean():+.3f} pp "
          f"(sd {delta.std(ddof=1):.3f}, p={p:.4f}, +folds {(delta > 0).sum()}/{len(delta)})")
    for version in ["Fourier v1", "Fourier v2"]:
        sub = frame[frame.version.eq(version)].pivot_table(index="fold", columns="k", values="bacc")
        step = sub[8] - sub[7]
        _, p = ttest_rel(sub[8], sub[7])
        print(f"{version} step 8 - 7: {step.mean():+.3f} pp "
              f"(sd {step.std(ddof=1):.3f}, p={p:.4f}, +folds {(step > 0).sum()}/{len(step)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--fourier-v2", type=Path,
                        default=ROOT / "results/fourier_v2_multitask_features.parquet")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulative_fourier_v2_lda.csv")
    parser.add_argument("--plot", type=Path,
                        default=ROOT / "results/cumulative_fourier_v2_lda.png")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=False)
    labels = LabelEncoder().fit_transform(cohort.labels)
    blocks = {name: cohort.blocks[name].astype(np.float32) for name in HAND8}
    blocks["Fourier v2"] = load_fourier_v2(args.fourier_v2, cohort.stems)
    log(f"{args.dataset}: {len(labels):,} images, {len(np.unique(labels))} classes")

    rows = [
        evaluate(blocks, ORDER_V1, labels, "Fourier v1", args.folds),
        evaluate(blocks, ORDER_V2, labels, "Fourier v2", args.folds),
    ]
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(args.output, index=False, float_format="%.6f")
    plot(frame, args.plot)
    summarize(frame)
    log(f"saved: {args.output}")
    log(f"saved: {args.plot}")


if __name__ == "__main__":
    main()
