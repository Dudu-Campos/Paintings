#!/usr/bin/env python3
"""Rank 3k line-graph features by univariate signal and XGBoost gain."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("paper/super_resolution_3k_sample.csv"))
    parser.add_argument("--line-features", type=Path, default=Path("paper/line_graph_3k_features.parquet"))
    parser.add_argument("--output", type=Path, default=Path("paper/line_graph_3k_feature_ranking.csv"))
    args = parser.parse_args()
    sample = pd.read_csv(args.sample)[["stem", "style"]]
    line = pd.read_parquet(args.line_features)
    columns = [column for column in line if column not in {"stem", "style"}]
    frame = sample.merge(line[["stem"] + columns], on="stem", validate="one_to_one").sort_values("stem")
    values = frame[columns].to_numpy(dtype=np.float32)
    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    f_values, _ = f_classif(values, labels)
    mutual = mutual_info_classif(values, labels, random_state=42)
    gains = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for train, _ in splitter.split(values, labels):
        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.5, reg_lambda=2.0, objective="multi:softmax",
            num_class=len(np.unique(labels)), eval_metric="mlogloss",
            tree_method="hist", n_jobs=16, random_state=42,
        )
        model.fit(values[train], labels[train], sample_weight=compute_sample_weight("balanced", labels[train]))
        gains.append(model.feature_importances_)
    result = pd.DataFrame({
        "feature": columns,
        "anova_f": f_values,
        "mutual_information": mutual,
        "xgb_gain_mean": np.mean(gains, axis=0),
        "xgb_gain_std": np.std(gains, axis=0),
    }).sort_values("xgb_gain_mean", ascending=False)
    result.to_csv(args.output, index=False)
    print(result.head(20).to_string(index=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
