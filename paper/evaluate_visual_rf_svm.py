"""Classical RF and RBF-SVM controls for the canonical visual GIST features.

Both controls use the 79,207-painting visual cohort and the same five outer
stratified folds as the final visual MLP.  They deliberately use no captions,
CNN embeddings, data augmentation or validation-label tuning.  The RBF-SVM
standardises each fold from its outer-training partition only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset


ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_classifier(name: str, seed: int, workers: int):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced_subsample", n_jobs=workers,
            random_state=seed,
        )
    if name == "rbf_svm":
        return SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", cache_size=12_000)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True, choices=("random_forest", "rbf_svm"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12, help="Parallel trees for RF; ignored by SVM.")
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / f"outputs/visual_{args.classifier}"

    frame, groups = load_dataset()
    columns = groups["SLIC"] + groups["Descriptors"] + groups["Hilbert"]
    values = frame[columns].to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    log(
        f"{args.classifier}: {len(frame):,} images, {len(encoder.classes_)} styles, "
        f"{values.shape[1]} visual features"
    )
    rows, predictions = [], []
    for fold, (train, test) in enumerate(splitter.split(values, y), start=1):
        train_values, test_values = values[train], values[test]
        if args.classifier == "rbf_svm":
            scaler = StandardScaler()
            train_values = scaler.fit_transform(train_values)
            test_values = scaler.transform(test_values)
        classifier = make_classifier(args.classifier, SEED + fold, args.workers)
        started = time.perf_counter()
        classifier.fit(train_values, y[train])
        prediction = classifier.predict(test_values)
        metrics = {
            "bacc": balanced_accuracy_score(y[test], prediction),
            "accuracy": accuracy_score(y[test], prediction),
            "macro_f1": f1_score(y[test], prediction, average="macro"),
        }
        rows.append({
            "fold": fold, "classifier": args.classifier, "n_images": len(frame),
            "n_train": len(train), "n_test": len(test), "n_features": values.shape[1],
            "elapsed_seconds": time.perf_counter() - started, **metrics,
        })
        predictions.extend({
            "fold": fold, "stem": stem, "true_style": encoder.classes_[truth],
            "predicted_style": encoder.classes_[pred],
        } for stem, truth, pred in zip(frame.iloc[test]["stem"], y[test], prediction, strict=True))
        log(f"fold {fold}/{N_FOLDS}: BAcc={metrics['bacc']:.4f}; {rows[-1]['elapsed_seconds'] / 60:.1f} min")
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / "fold_results.csv", index=False)
    detail = pd.DataFrame(rows)
    summary = detail.agg({
        "bacc": ["mean", "std"], "accuracy": "mean", "macro_f1": "mean", "elapsed_seconds": "sum",
    })
    summary.to_csv(output_dir / "summary.csv")
    pd.DataFrame(predictions).to_csv(output_dir / "predictions.csv", index=False)
    log(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
