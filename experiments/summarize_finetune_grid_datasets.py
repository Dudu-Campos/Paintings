"""Resumo de tamanho/desbalanceamento dos 7 datasets do grid de fine-tuning
(`run_finetune_grid_resilient.sh`), aplicando o mesmo filtro `--min-per-class`
que o driver usa por dataset (`dataset_args()`), para que os números batam com
a coorte que o treino realmente ve -- nao a contagem bruta do metadado.

Saida: results/finetune_grid_datasets_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    class_column: str
    min_per_class: int
    train_only: bool = False  # datasets de split oficial: contar so o `train`,
    # que e o que caracteriza o desbalanceamento real (val/test costumam ser
    # balanceados por construcao -- ver CLAUDE.md sobre o 996x do Places-LT).


DATASETS = (
    DatasetSpec("Multitask (125cl)", ROOT / "data/multitask_meta.csv", "style", 5),
    DatasetSpec("OilPainting", ROOT / "data/oilpainting_meta.csv", "style", 50),
    DatasetSpec("Rijksmuseum (drawings)", ROOT / "data/rijksdraw_labels_clean.csv", "period", 50),
    DatasetSpec("Food-101", ROOT / "data/food101_meta.csv", "style", 5, train_only=True),
    DatasetSpec("Places-LT", ROOT / "data/places_lt_meta.csv", "style", 5, train_only=True),
    DatasetSpec("NCT-CRC", ROOT / "data/nct_crc_meta.csv", "style", 5, train_only=True),
    DatasetSpec("MINC-2500", ROOT / "data/minc2500_meta.csv", "style", 5, train_only=True),
)


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def summarize(spec: DatasetSpec) -> dict[str, object]:
    frame = pd.read_csv(spec.path)
    if spec.train_only:
        frame = frame[frame["split"].eq("train")]
    labels = frame[spec.class_column].map(normalize_label)
    labels = labels[labels.ne("")]
    counts = labels.value_counts()

    retained = counts[counts >= spec.min_per_class]
    dropped_classes = int(len(counts) - len(retained))
    values = retained.to_numpy(dtype=np.int64)
    total = int(values.sum())
    majority = int(values.max())
    minority = int(values.min())

    return {
        "dataset": spec.name,
        "class_column": spec.class_column,
        "counted_on": "train split" if spec.train_only else "full corpus",
        "min_per_class": spec.min_per_class,
        "n_images": total,
        "n_classes": int(len(retained)),
        "classes_dropped_below_min": dropped_classes,
        "min_class_size": minority,
        "mean_class_size": round(float(np.mean(values)), 1),
        "median_class_size": round(float(np.median(values)), 1),
        "max_class_size": majority,
        "imbalance_ratio_max_min": round(float(majority / minority), 2) if minority else float("inf"),
    }


def main() -> None:
    rows = [summarize(spec) for spec in DATASETS]
    summary = pd.DataFrame(rows)
    output_path = ROOT / "results/finetune_grid_datasets_summary.csv"
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSalvo em: {output_path}")


if __name__ == "__main__":
    main()
