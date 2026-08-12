"""Confusion analysis for the standalone ArtEmis caption representation.

This script reuses the already-computed out-of-fold predictions from
``compare_text_controls.py``.  Consequently, the confusion matrices have the
same cohort, folds, TF-IDF preprocessing and classifier as the text-control
comparison; no model is fitted again here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "paper/text_controls_standalone_predictions.csv"
AUDIT = ROOT / "paper/artemis_model_leakage_audit.csv"
PREDICTION_COLUMN = "prediction_artemis_fine_tuned"

OUTPUT_PNG = ROOT / "paper/artemis_text_confusion.png"
OUTPUT_SUMMARY = ROOT / "paper/artemis_text_confusion_summary.csv"
OUTPUT_RAW_ALL = ROOT / "paper/artemis_text_confusion_raw_all.csv"
OUTPUT_NORM_ALL = ROOT / "paper/artemis_text_confusion_normalized_all.csv"
OUTPUT_RAW_HELDOUT = ROOT / "paper/artemis_text_confusion_raw_heldout.csv"
OUTPUT_NORM_HELDOUT = ROOT / "paper/artemis_text_confusion_normalized_heldout.csv"
OUTPUT_BY_CLASS = ROOT / "paper/artemis_text_confusion_by_class.csv"


def matrix_frame(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "true_style"
    frame.columns.name = "predicted_style"
    return frame


def summarize_subset(
    frame: pd.DataFrame, subset: str, labels: list[str]
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    true = frame["label"].to_numpy(dtype=str)
    predicted = frame[PREDICTION_COLUMN].to_numpy(dtype=str)
    raw = confusion_matrix(true, predicted, labels=labels)
    normalized = confusion_matrix(
        true, predicted, labels=labels, normalize="true"
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=labels, zero_division=0
    )

    class_rows = []
    for index, style in enumerate(labels):
        errors = raw[index].copy()
        errors[index] = 0
        target = int(np.argmax(errors))
        class_rows.append(
            {
                "subset": subset,
                "style": style,
                "support": int(support[index]),
                "recall": float(recall[index]),
                "precision": float(precision[index]),
                "f1": float(f1[index]),
                "top_confusion": labels[target],
                "top_confusion_count": int(errors[target]),
                "top_confusion_rate": (
                    float(errors[target] / support[index])
                    if support[index]
                    else 0.0
                ),
            }
        )

    summary = {
        "subset": subset,
        "n": len(frame),
        "n_styles": len(labels),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro")),
    }
    return summary, raw, normalized, class_rows


def short_label(label: str) -> str:
    return label.replace("_", " ")


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS, dtype=str)
    audit = pd.read_csv(AUDIT, usecols=["stem", "source_split"], dtype=str)
    if predictions["stem"].duplicated().any() or audit["stem"].duplicated().any():
        raise ValueError("Duplicate stems in prediction or audit inputs")
    frame = predictions.merge(audit, on="stem", how="left", validate="one_to_one")
    if frame["source_split"].isna().any():
        raise ValueError("Some OOF predictions have no seen/held-out assignment")

    labels = sorted(frame["label"].unique().tolist())
    subsets = {
        "all": frame,
        "finetune_seen": frame[frame["source_split"] == "finetune_seen"],
        "heldout": frame[frame["source_split"] == "heldout"],
    }
    summaries = []
    class_rows = []
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, subset in subsets.items():
        summary, raw, normalized, rows = summarize_subset(subset, name, labels)
        summaries.append(summary)
        class_rows.extend(rows)
        matrices[name] = (raw, normalized)

    pd.DataFrame(summaries).to_csv(OUTPUT_SUMMARY, index=False)
    pd.DataFrame(class_rows).to_csv(OUTPUT_BY_CLASS, index=False)
    matrix_frame(matrices["all"][0], labels).to_csv(OUTPUT_RAW_ALL)
    matrix_frame(matrices["all"][1], labels).to_csv(OUTPUT_NORM_ALL)
    matrix_frame(matrices["heldout"][0], labels).to_csv(OUTPUT_RAW_HELDOUT)
    matrix_frame(matrices["heldout"][1], labels).to_csv(OUTPUT_NORM_HELDOUT)

    figure, axes = plt.subplots(1, 2, figsize=(25, 11), constrained_layout=True)
    shown = (
        ("all", "Todas as 12.210 imagens"),
        ("heldout", "Somente held-out (8.474 imagens)"),
    )
    image = None
    tick_labels = [short_label(label) for label in labels]
    for axis, (name, title) in zip(axes, shown):
        normalized = matrices[name][1]
        image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
        metric = next(row for row in summaries if row["subset"] == name)
        axis.set_title(
            f"{title}\nBalanced accuracy = {metric['balanced_accuracy']:.3f}",
            fontsize=15,
        )
        axis.set_xlabel("Estilo predito")
        axis.set_ylabel("Estilo verdadeiro")
        axis.set_xticks(np.arange(len(labels)), tick_labels, rotation=60, ha="right")
        axis.set_yticks(np.arange(len(labels)), tick_labels)
        axis.tick_params(axis="both", labelsize=7)
        axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.15, alpha=0.25)
        axis.tick_params(which="minor", bottom=False, left=False)

    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.75, pad=0.01)
    colorbar.set_label("Proporção dentro da classe verdadeira")
    figure.suptitle(
        "ArtEmis: matriz de confusão da representação textual isolada (OOF, 5 folds)",
        fontsize=18,
    )
    figure.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary_frame = pd.DataFrame(summaries)
    print(summary_frame.to_string(index=False))
    heldout_rows = pd.DataFrame(class_rows)
    heldout_rows = heldout_rows[heldout_rows["subset"] == "heldout"]
    print("\nBest held-out recalls:")
    print(
        heldout_rows.nlargest(8, "recall")[
            ["style", "support", "recall", "top_confusion", "top_confusion_rate"]
        ].to_string(index=False)
    )
    print("\nWorst held-out recalls:")
    print(
        heldout_rows.nsmallest(8, "recall")[
            ["style", "support", "recall", "top_confusion", "top_confusion_rate"]
        ].to_string(index=False)
    )
    print(f"\nSaved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
