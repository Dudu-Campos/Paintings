"""Shared XGBoost and MLP evaluators for feature-representation experiments."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
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


def evaluate_xgb(
    values: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    scores = []
    n_classes = len(np.unique(y))
    for train, test in splits:
        classifier = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.5, reg_lambda=2.0, objective="multi:softmax",
            num_class=n_classes, eval_metric="mlogloss", tree_method="hist",
            n_jobs=16, random_state=SEED,
        )
        classifier.fit(values[train], y[train], sample_weight=compute_sample_weight("balanced", y[train]))
        scores.append(float(balanced_accuracy_score(y[test], classifier.predict(values[test]))))
    return scores


def evaluate_mlp(
    values: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    scores = []
    n_classes = len(np.unique(y))
    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=SEED + fold
        )
        scaler = StandardScaler()
        train_values = scaler.fit_transform(values[train]).astype(np.float32)
        validation_values = scaler.transform(values[validation]).astype(np.float32)
        test_values = scaler.transform(values[outer_test]).astype(np.float32)
        torch.manual_seed(SEED + fold)
        model = MLP(values.shape[1], n_classes).to(DEVICE)
        counts = np.bincount(y[train], minlength=n_classes).astype(np.float32)
        weights = (1 / counts) / (1 / counts).mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_values), torch.from_numpy(y[train]).long()),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda",
        )
        validation_tensor = torch.from_numpy(validation_values).to(DEVICE)
        best_score, best_state, stale = -1.0, None, 0
        for _ in range(MAX_EPOCHS):
            model.train()
            for batch_values, batch_labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_values.to(DEVICE, non_blocking=True)), batch_labels.to(DEVICE, non_blocking=True))
                loss.backward()
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                score = balanced_accuracy_score(y[validation], model(validation_tensor).argmax(1).cpu().numpy())
            if score > best_score:
                best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
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
        scores.append(float(balanced_accuracy_score(y[outer_test], prediction)))
    return scores
