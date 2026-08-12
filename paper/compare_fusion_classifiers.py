"""XGBoost and MLP fusion curves using cached legacy and SLIC features."""

from __future__ import annotations

import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from compare_fusion_curves import load_multigraph
from fusion_curve_slic import (
    BASELINE_DIR, N_FOLDS, ROOT, SEED, SENTIMENT_FEATURES,
    load_npy, log,
)


OUTPUT_CSV = ROOT / "paper/fusion_curves_3clf.csv"
OUTPUT_PNG = ROOT / "paper/fusion_curves_3clf.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 4096


class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_features
        for width in (512, 512, 256, 128):
            layers.extend([
                nn.Linear(previous, width), nn.BatchNorm1d(width),
                nn.GELU(), nn.Dropout(0.35),
            ])
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def load_dataset() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    legacy, legacy_columns = load_multigraph(BASELINE_DIR, "legacy")
    slic, slic_columns = load_multigraph(ROOT / "outputs/wikiart_slic_features", "slic")
    hilbert, hilbert_columns = load_npy(
        "hilbert_features.npy", "hilbert_stems.npy", "hilbert"
    )
    random_walk, random_walk_columns = load_npy(
        "rw_features.npy", "rw_stems.npy", "rw"
    )
    hybrid_walk, hybrid_walk_columns = load_npy(
        "hybrid_walk_features.npy", "hybrid_walk_stems.npy", "hybrid"
    )
    descriptors, descriptor_columns = load_npy(
        "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
    )
    sentiment = pd.read_parquet(
        BASELINE_DIR / "sentiment_features.parquet",
        columns=["stem"] + SENTIMENT_FEATURES,
    ).drop_duplicates("stem")
    frame = legacy.merge(slic, on="stem", how="inner", validate="one_to_one")
    for addition in (hilbert, random_walk, hybrid_walk, descriptors, sentiment):
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
    metadata = pd.read_csv(ROOT / "data/artemis/wikiart_meta.csv")
    label_map = dict(zip(metadata["stem"].astype(str), metadata["style"].astype(str)))
    frame["label"] = frame["stem"].astype(str).map(label_map)
    frame = frame.dropna(subset=["label"]).sort_values("stem").reset_index(drop=True)
    counts = frame["label"].value_counts()
    frame = frame[frame["label"].isin(counts[counts >= N_FOLDS].index)].reset_index(drop=True)
    common = {
        "Descriptors": descriptor_columns,
        "Hilbert": hilbert_columns,
        "Sentiment": SENTIMENT_FEATURES,
    }
    graphs = {"Legacy percolation": legacy_columns, "SLIC-2048": slic_columns}
    return frame, graphs, common


def evaluate_xgb(
    values: np.ndarray, y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    scores = []
    n_classes = len(np.unique(y))
    for fold, (train, test) in enumerate(splits, 1):
        classifier = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.5, reg_lambda=2.0, objective="multi:softmax",
            num_class=n_classes, eval_metric="mlogloss", tree_method="hist",
            n_jobs=16, random_state=SEED,
        )
        weights = compute_sample_weight("balanced", y[train])
        classifier.fit(values[train], y[train], sample_weight=weights)
        score = balanced_accuracy_score(y[test], classifier.predict(values[test]))
        scores.append(float(score))
        log(f"    XGBoost fold {fold}/{N_FOLDS}: {score:.4f}")
    return scores


def evaluate_mlp(
    values: np.ndarray, y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    scores = []
    n_classes = len(np.unique(y))
    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train],
            random_state=SEED + fold,
        )
        scaler = StandardScaler()
        train_values = scaler.fit_transform(values[train]).astype(np.float32)
        validation_values = scaler.transform(values[validation]).astype(np.float32)
        test_values = scaler.transform(values[outer_test]).astype(np.float32)

        torch.manual_seed(SEED + fold)
        model = MLP(values.shape[1], n_classes).to(DEVICE)
        class_counts = np.bincount(y[train], minlength=n_classes).astype(np.float32)
        class_weights = (1 / class_counts) / (1 / class_counts).mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS
        )
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_values),
                torch.from_numpy(y[train]).long(),
            ),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            pin_memory=DEVICE.type == "cuda",
        )
        validation_tensor = torch.from_numpy(validation_values).to(DEVICE)
        best_score = -1.0
        best_state = None
        stale = 0
        for _ in range(MAX_EPOCHS):
            model.train()
            for batch_values, batch_labels in loader:
                batch_values = batch_values.to(DEVICE, non_blocking=True)
                batch_labels = batch_labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(model(batch_values), batch_labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                prediction = model(validation_tensor).argmax(1).cpu().numpy()
            score = balanced_accuracy_score(y[validation], prediction)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= PATIENCE:
                break
        if best_state is None:
            raise RuntimeError("MLP failed to produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(test_values).to(DEVICE)).argmax(1).cpu().numpy()
        score = balanced_accuracy_score(y[outer_test], prediction)
        scores.append(float(score))
        log(f"    MLP fold {fold}/{N_FOLDS}: {score:.4f}")
    return scores


def main() -> None:
    frame, graphs, common = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y)), y))
    log(f"Cached cohort: {len(frame):,} images, device={DEVICE}; no graph extraction")

    stages = [
        ("Multigraph", []), ("Descriptors", common["Descriptors"]),
        ("Hilbert", common["Hilbert"]), ("Sentiment", common["Sentiment"]),
    ]
    rows = []
    for method, graph_columns in graphs.items():
        active = list(graph_columns)
        for step, (group, columns) in enumerate(stages, 1):
            active.extend(columns)
            values = frame[active].to_numpy(dtype=np.float32)
            for classifier, evaluator in (("XGBoost", evaluate_xgb), ("MLP", evaluate_mlp)):
                started = time.perf_counter()
                log(f"{method} | {classifier} | {group} ({len(active)} features)")
                scores = np.asarray(evaluator(values, y, splits))
                rows.append({
                    "method": method, "classifier": classifier, "step": step,
                    "added_group": group, "n_features": len(active),
                    "bacc_mean": scores.mean(), "bacc_std": scores.std(),
                    "elapsed_seconds": time.perf_counter() - started,
                    **{f"fold_{index + 1}": value for index, value in enumerate(scores)},
                })
                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
                log(f"  Result: {scores.mean():.4f} ± {scores.std():.4f}")

    result = pd.DataFrame(rows)
    lda = pd.read_csv(ROOT / "paper/fusion_curves_slic_vs_legacy.csv")
    lda["classifier"] = "LDA"
    combined = pd.concat([lda, result], ignore_index=True, sort=False)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = {"Legacy percolation": "#777777", "SLIC-2048": "#2a78d6"}
    for axis, classifier in zip(axes, ("LDA", "XGBoost", "MLP")):
        subset = combined[combined["classifier"] == classifier]
        for method, group in subset.groupby("method", sort=False):
            group = group[group["step"] <= 4]
            axis.plot(group["step"], group["bacc_mean"], "o-", linewidth=2.2,
                      color=colors[method], label=method)
            axis.fill_between(
                group["step"], group["bacc_mean"] - group["bacc_std"],
                group["bacc_mean"] + group["bacc_std"],
                color=colors[method], alpha=0.14,
            )
        axis.set_title(classifier)
        axis.set_xticks(range(1, 5), ["Graph", "+ Desc", "+ Hilbert", "+ Sent"],
                        rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(frameon=False)
    figure.suptitle("Paired feature-fusion curves — cached legacy vs SLIC")
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    log(f"Saved: {OUTPUT_CSV}")
    log(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
