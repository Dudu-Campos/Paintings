"""Latent-space CKA between PRISM representation families.

Question: do different representation families induce different latent
organizations under the same downstream learner?

For each family we train the same dense MLP, with the same 128-d bottleneck,
loss, folds and seeds. For every test image we save the last hidden layer and
compare latent matrices with linear CKA.

The key contrast is:

    CKA(within family, different seeds) > CKA(between families)

This supports the weaker and defensible claim that representation families
induce systematically different latent geometries under the same learner. It is
not a claim of independent information.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "16")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402
from sweep_theta_hilbert import linear_cka  # noqa: E402

# Procedural families, not the error-pattern clades in view_names.FAMILIES.
# Text and the similarity graph are intentionally excluded: Text is external
# supervision, and the similarity graph is derived from all visual views.
FAMILY_VIEWS = {
    "Image statistics": ["Descriptors"],
    "Region graphs": ["Felzenszwalb", "SLIC", "L(G_felz)", "Line graph"],
    "Space-filling curve": ["Hilbert"],
    "Spectral": ["Fourier", "Gabor"],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class DenseLatentMLP(nn.Module):
    """Same learner for every family: dense MLP with a 128-d bottleneck."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.35),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))

    def latent(self, values: torch.Tensor) -> torch.Tensor:
        return self.features(values)


def metrics_of(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "bacc": 100 * balanced_accuracy_score(truth, prediction),
        "accuracy": 100 * accuracy_score(truth, prediction),
        "macro_f1": 100 * f1_score(truth, prediction, average="macro"),
    }


def train_latent_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    patience: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    seed_everything(seed)
    model = DenseLatentMLP(train_x.shape[1], n_classes).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=2048, shuffle=True, drop_last=False, num_workers=0,
        pin_memory=DEVICE.type == "cuda", generator=torch.Generator().manual_seed(seed),
    )
    validation_tensor = torch.from_numpy(validation_x).to(DEVICE, non_blocking=True)
    test_tensor = torch.from_numpy(test_x).to(DEVICE, non_blocking=True)

    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_tensor).argmax(1).cpu().numpy()
        score = balanced_accuracy_score(validation_y, validation_prediction)
        if score > best_score:
            best_score, best_state, best_epoch, stale = float(score), copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(test_tensor)
        prediction = logits.argmax(1).cpu().numpy()
        latent = model.latent(test_tensor).cpu().numpy().astype(np.float32)
    metrics = metrics_of(test_y, prediction)
    metrics.update({"validation_bacc": 100 * best_score, "best_epoch": best_epoch})
    del model, validation_tensor, test_tensor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, prediction.astype(np.int16), latent


def class_center(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    for label in np.unique(labels):
        mask = labels == label
        out[mask] -= out[mask].mean(axis=0, keepdims=True)
    return out


def load_or_train(
    cache_path: Path,
    values: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    test_index: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    patience: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True)
        metrics = {k: float(blob[k]) for k in ("bacc", "accuracy", "macro_f1", "validation_bacc", "best_epoch")}
        return metrics, blob["prediction"], blob["latent"]

    scaler = StandardScaler().fit(values[train_index])
    scaled = scaler.transform(values).astype(np.float32)
    metrics, prediction, latent = train_latent_mlp(
        scaled[train_index], labels[train_index],
        scaled[validation_index], labels[validation_index],
        scaled[test_index], labels[test_index],
        n_classes, seed, epochs, patience,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, prediction=prediction, latent=latent, **metrics)
    return metrics, prediction, latent


def pairwise_rows(
    fold: int,
    test_labels: np.ndarray,
    predictions: dict[tuple[str, int], np.ndarray],
    latents: dict[tuple[str, int], np.ndarray],
    families: list[str],
    seeds: list[int],
) -> list[dict]:
    rows: list[dict] = []
    correct_stack = np.column_stack([predictions[(family, seed)] == test_labels
                                     for family in families for seed in seeds])
    masks = {
        "all": np.ones(len(test_labels), dtype=bool),
        "contested": ~correct_stack.all(axis=1),
    }
    for scope, mask in masks.items():
        if mask.sum() < 3:
            continue
        for centered in (False, True):
            prepared: dict[tuple[str, int], np.ndarray] = {}
            labels_here = test_labels[mask]
            for family in families:
                for seed in seeds:
                    z = latents[(family, seed)][mask]
                    prepared[(family, seed)] = class_center(z, labels_here) if centered else z
            keys = [(family, seed) for family in families for seed in seeds]
            for i, (family_a, seed_a) in enumerate(keys):
                for family_b, seed_b in keys[i + 1:]:
                    relation = "within" if family_a == family_b else "between"
                    if relation == "within" and seed_a == seed_b:
                        continue
                    rows.append({
                        "fold": fold,
                        "scope": scope,
                        "class_centered": centered,
                        "family_a": family_a,
                        "family_b": family_b,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "relation": relation,
                        "n_samples": int(mask.sum()),
                        "cka": linear_cka(prepared[(family_a, seed_a)], prepared[(family_b, seed_b)]),
                    })
    return rows


def summarize_pairs(pairs: pd.DataFrame, output_prefix: Path, bootstrap: int, seed: int) -> None:
    matrix = (pairs.groupby(["scope", "class_centered", "family_a", "family_b"], as_index=False)
              .agg(cka_mean=("cka", "mean"), cka_std=("cka", "std"), n=("cka", "size")))
    output_prefix.with_name(output_prefix.name + "_matrix.csv").parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_prefix.with_name(output_prefix.name + "_matrix.csv"), index=False)

    rng = np.random.default_rng(seed)
    tests = []
    for (scope, centered), group in pairs.groupby(["scope", "class_centered"]):
        within = group[group.relation == "within"].cka.to_numpy(float)
        between = group[group.relation == "between"].cka.to_numpy(float)
        observed = float(within.mean() - between.mean())
        samples = []
        for _ in range(bootstrap):
            w = rng.choice(within, size=len(within), replace=True)
            b = rng.choice(between, size=len(between), replace=True)
            samples.append(float(w.mean() - b.mean()))
        samples = np.asarray(samples)
        # One-sided probability that the hypothesized positive gap is <= 0.
        p_le_zero = float((np.sum(samples <= 0) + 1) / (bootstrap + 1))
        tests.append({
            "scope": scope,
            "class_centered": centered,
            "within_mean": float(within.mean()),
            "between_mean": float(between.mean()),
            "delta_within_minus_between": observed,
            "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
            "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
            "bootstrap_p_delta_le_0": p_le_zero,
            "n_within_pairs": len(within),
            "n_between_pairs": len(between),
        })
    pd.DataFrame(tests).to_csv(output_prefix.with_name(output_prefix.name + "_bootstrap.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt",
                        help="Pass a missing path such as 'none' for the full multitask125 cohort.")
    parser.add_argument("--families", nargs="+", default=list(FAMILY_VIEWS))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "outputs/family_latent_cka")
    parser.add_argument("--output-prefix", type=Path,
                        default=ROOT / "results/family_latent_cka")
    args = parser.parse_args()

    unknown = sorted(set(args.families) - set(FAMILY_VIEWS))
    if unknown:
        raise ValueError("unknown families: " + ", ".join(unknown))

    needed_views = sorted({view for family in args.families for view in FAMILY_VIEWS[family]},
                          key=(VISUAL + ["Gabor"]).index)
    blocks, raw_labels, _ = load_corpus(args.dataset, needed_views)
    if args.classes_file.exists():
        wanted = {key(line) for line in args.classes_file.read_text().splitlines() if line.strip()}
        mask = np.array([key(value) in wanted for value in raw_labels])
        if len(np.unique(raw_labels[mask])) != len(wanted):
            raise ValueError("casamento de classes do benchmark incompleto")
        blocks = {name: clean(value[mask]) for name, value in blocks.items()}
        raw_labels = raw_labels[mask]
    else:
        blocks = {name: clean(value) for name, value in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    family_values = {
        family: np.hstack([blocks[view] for view in FAMILY_VIEWS[family] if view in blocks]).astype(np.float32)
        for family in args.families
    }

    log(f"{args.dataset}: {len(labels):,} images, {n_classes} classes")
    for family, values in family_values.items():
        log(f"  {family:<20} {values.shape[1]:>4} dims")

    metrics_rows: list[dict] = []
    pair_rows: list[dict] = []
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        validation_stratify = labels[train_index]
        validation_index_train, validation_index = train_test_split(
            train_index, test_size=0.1, stratify=validation_stratify,
            random_state=SEED + fold,
        )
        predictions: dict[tuple[str, int], np.ndarray] = {}
        latents: dict[tuple[str, int], np.ndarray] = {}
        for family, values in family_values.items():
            for seed in args.seeds:
                cache = (args.cache_dir / args.dataset / f"fold_{fold:02d}" /
                         family.replace(" ", "_").replace("/", "-") /
                         f"seed_{seed}.npz")
                metrics, prediction, latent = load_or_train(
                    cache, values, labels, validation_index_train, validation_index, test_index,
                    n_classes, seed, args.epochs, args.patience,
                )
                predictions[(family, seed)] = prediction
                latents[(family, seed)] = latent
                metrics_rows.append({
                    "dataset": args.dataset,
                    "fold": fold,
                    "seed": seed,
                    "family": family,
                    "dims": values.shape[1],
                    **metrics,
                })
                log(f"fold {fold} seed {seed} {family:<20} "
                    f"BAcc {metrics['bacc']:5.2f} latent {latent.shape}")
        pair_rows.extend(pairwise_rows(
            fold, labels[test_index], predictions, latents, args.families, list(args.seeds)
        ))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows).drop_duplicates(["fold", "seed", "family"])
    pairs = pd.DataFrame(pair_rows)
    metrics.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv"), index=False)
    pairs.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_pairs.csv"), index=False)
    summarize_pairs(pairs, args.output_prefix, args.bootstrap, SEED)
    log(f"saved: {args.output_prefix.with_name(args.output_prefix.name + '_pairs.csv')}")
    log(f"saved: {args.output_prefix.with_name(args.output_prefix.name + '_matrix.csv')}")
    log(f"saved: {args.output_prefix.with_name(args.output_prefix.name + '_bootstrap.csv')}")


if __name__ == "__main__":
    main()
