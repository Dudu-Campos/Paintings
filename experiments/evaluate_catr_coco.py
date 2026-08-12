"""Evaluate the public COCO-trained CATR captions on the fixed 12k cohort.

Outputs include standalone OOF TF-IDF/SVM performance, nested-OOF text scores
fused with the existing numeric branch MLP, lexical leakage checks and a
row-normalized confusion matrix.  Splits and preprocessing match the earlier
paired text-control experiment.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from compare_text_controls import aligned_decision, first_complete_sentence
from compare_text_controls_mlp import (
    nested_text_scores,
    prediction_metrics,
    run_mlp,
)
from fusion_curve_slic import N_FOLDS, ROOT, SEED, log
from slic_oof_stacking import load_dataset
from tfidf_caption_style import make_classifier, make_stop_words, make_vectorizer


CAPTIONS = ROOT / "outputs/affect_captions/catr_coco_12k.jsonl"
COHORT = ROOT / "paper/text_controls_cohort.csv"
QWEN_AUDIT = ROOT / "paper/qwen_leakage_audit.csv"
OLD_MLP_PREDICTIONS = ROOT / "paper/text_controls_mlp_predictions.csv"
OLD_PLACEBOS = ROOT / "paper/text_controls_mlp_placebos.csv"

PREFIX = ROOT / "paper/catr_coco"
OUTPUT_STANDALONE = Path(f"{PREFIX}_standalone_results.csv")
OUTPUT_STANDALONE_FOLDS = Path(f"{PREFIX}_standalone_folds.csv")
OUTPUT_STANDALONE_PREDICTIONS = Path(f"{PREFIX}_standalone_predictions.csv")
OUTPUT_MLP = Path(f"{PREFIX}_mlp_results.csv")
OUTPUT_MLP_FOLDS = Path(f"{PREFIX}_mlp_folds.csv")
OUTPUT_MLP_PREDICTIONS = Path(f"{PREFIX}_mlp_predictions.csv")
OUTPUT_LEAKAGE = Path(f"{PREFIX}_leakage_audit.csv")
OUTPUT_LEAKAGE_SUMMARY = Path(f"{PREFIX}_leakage_summary.csv")
OUTPUT_CONFUSION = Path(f"{PREFIX}_confusion_normalized.csv")
OUTPUT_CONFUSION_PNG = Path(f"{PREFIX}_confusion.png")

STYLE_ALIASES = {
    "Abstract_Expressionism": ("abstract expressionism", "abstract expressionist"),
    "Action_painting": ("action painting",),
    "Analytical_Cubism": ("analytical cubism", "analytic cubism"),
    "Art_Nouveau_Modern": ("art nouveau",),
    "Baroque": ("baroque",),
    "Color_Field_Painting": ("color field painting", "colour field painting"),
    "Contemporary_Realism": ("contemporary realism",),
    "Cubism": ("cubism", "cubist"),
    "Early_Renaissance": ("early renaissance",),
    "Expressionism": ("expressionism", "expressionist"),
    "Fauvism": ("fauvism", "fauvist"),
    "High_Renaissance": ("high renaissance",),
    "Impressionism": ("impressionism", "impressionist"),
    "Mannerism_Late_Renaissance": ("mannerism", "mannerist", "late renaissance"),
    "Minimalism": ("minimalism", "minimalist"),
    "Naive_Art_Primitivism": ("naive art", "primitive art", "primitivism"),
    "New_Realism": ("new realism",),
    "Northern_Renaissance": ("northern renaissance",),
    "Pointillism": ("pointillism", "pointillist"),
    "Pop_Art": ("pop art",),
    "Post_Impressionism": ("post impressionism", "post impressionist"),
    "Realism": ("realism", "realist"),
    "Rococo": ("rococo",),
    "Romanticism": ("romanticism",),
    "Surrealism": ("surrealism", "surrealist"),
    "Symbolism": ("symbolism", "symbolist"),
    "Synthetic_Cubism": ("synthetic cubism",),
    "Ukiyo_e": ("ukiyo e", "ukiyo-e"),
}
ALL_STYLE_TERMS = sorted(
    {alias for aliases in STYLE_ALIASES.values() for alias in aliases},
    key=len,
    reverse=True,
)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(character for character in value if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def load_captions() -> pd.DataFrame:
    rows = []
    with CAPTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if len(frame) != 12_210 or frame["stem"].nunique() != 12_210:
        raise ValueError(
            f"CATR output is incomplete: rows={len(frame):,}, "
            f"unique={frame['stem'].nunique():,}"
        )
    if frame["caption"].astype(str).str.strip().eq("").any():
        raise ValueError("CATR produced empty captions")
    if frame["checkpoint_sha256"].nunique() != 1:
        raise ValueError("Mixed CATR checkpoints in output")
    return frame[["stem", "caption"]]


def load_frame() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT, usecols=["stem", "label", "path"], dtype=str)
    captions = load_captions()
    frame = cohort.merge(captions, on="stem", how="left", validate="one_to_one")
    if frame["caption"].isna().any():
        raise ValueError("Some cohort paintings lack CATR captions")
    frame["text"] = frame["caption"].map(first_complete_sentence)
    return frame


def standalone(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    texts = frame["text"].to_numpy(dtype=str)
    labels = frame["label"].to_numpy(dtype=str)
    classes = np.unique(labels)
    stop_words = make_stop_words(classes)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    predictions = np.empty(len(frame), dtype=object)
    fold_assignment = np.zeros(len(frame), dtype=np.int8)
    fold_rows = []
    for fold, (train, test) in enumerate(
        splitter.split(np.zeros(len(frame), dtype=np.uint8), labels), 1
    ):
        vectorizer = make_vectorizer(stop_words)
        train_values = vectorizer.fit_transform(texts[train])
        test_values = vectorizer.transform(texts[test])
        classifier = make_classifier()
        classifier.fit(train_values, labels[train])
        prediction = classifier.predict(test_values)
        predictions[test] = prediction
        fold_assignment[test] = fold
        fold_rows.append({
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "bacc": balanced_accuracy_score(labels[test], prediction),
            "accuracy": accuracy_score(labels[test], prediction),
            "macro_f1": f1_score(labels[test], prediction, average="macro"),
        })
        log(f"CATR standalone fold {fold}/{N_FOLDS}: BAcc={fold_rows[-1]['bacc']:.4f}")
    folds = pd.DataFrame(fold_rows)
    result = pd.DataFrame([{
        "condition": "CATR COCO human",
        "n_images": len(frame),
        "n_styles": len(classes),
        "bacc_oof": balanced_accuracy_score(labels, predictions),
        "bacc_fold_mean": folds["bacc"].mean(),
        "bacc_fold_std": folds["bacc"].std(ddof=0),
        "accuracy_oof": accuracy_score(labels, predictions),
        "macro_f1_oof": f1_score(labels, predictions, average="macro"),
    }])
    result.to_csv(OUTPUT_STANDALONE, index=False)
    folds.to_csv(OUTPUT_STANDALONE_FOLDS, index=False)
    pd.DataFrame({
        "stem": frame["stem"],
        "label": labels,
        "fold": fold_assignment,
        "prediction": predictions,
        "correct": labels == predictions,
        "caption": texts,
    }).to_csv(OUTPUT_STANDALONE_PREDICTIONS, index=False)
    return labels, predictions, classes.tolist()


def run_numeric_mlp(frame: pd.DataFrame) -> None:
    numeric_frame, groups = load_dataset()
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric_frame = numeric_frame[["stem", "label"] + numeric_columns]
    merged = frame.merge(
        numeric_frame, on="stem", how="inner", validate="one_to_one",
        suffixes=("", "_numeric"),
    )
    if len(merged) != len(frame):
        raise ValueError(
            f"CATR/numeric cohort differs: {len(frame)} -> {len(merged)}"
        )
    label_mismatches = int(merged["label"].ne(merged["label_numeric"]).sum())
    if label_mismatches:
        # Some ArtEmis metadata stems occur under more than one style and the
        # full-dataset loader's stem-only map keeps the last occurrence.  The
        # fixed paired cohort is path-qualified, and its labels agree with the
        # style stored alongside the corresponding SLIC feature rows.
        log(
            f"Keeping {label_mismatches} path-qualified cohort labels over "
            "ambiguous stem-only ArtEmis metadata labels"
        )
    encoder = LabelEncoder()
    labels = encoder.fit_transform(merged["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    stop_words = make_stop_words(encoder.classes_)
    stems = merged["stem"].to_numpy(dtype=str)
    texts = merged["text"].to_numpy(dtype=str)
    numeric_values = merged[numeric_columns].to_numpy(dtype=np.float32)
    branch_sizes = [len(columns) for columns in groups.values()] + [n_classes]
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(merged), dtype=np.uint8), labels))
    predictions = np.full(len(merged), -1, dtype=np.int64)
    fold_rows = []
    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        fit_train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=labels[outer_train],
            random_state=SEED + fold,
        )
        train_text, test_text = nested_text_scores(
            "CATR COCO human", fold, outer_train, outer_test, stems, texts,
            labels, stop_words, n_classes,
        )
        augmented = np.empty(
            (len(merged), numeric_values.shape[1] + n_classes), dtype=np.float32
        )
        augmented[:, :numeric_values.shape[1]] = numeric_values
        augmented[outer_train, numeric_values.shape[1]:] = train_text
        augmented[outer_test, numeric_values.shape[1]:] = test_text
        prediction, best_epoch, elapsed = run_mlp(
            augmented, branch_sizes, labels, fit_train, validation, outer_test,
            n_classes, SEED + fold + 10_000,
        )
        predictions[outer_test] = prediction
        fold_rows.append({
            "fold": fold,
            "bacc": balanced_accuracy_score(labels[outer_test], prediction),
            "best_epoch": best_epoch,
            "elapsed_seconds": elapsed,
        })
        log(f"CATR MLP fold {fold}/{N_FOLDS}: BAcc={fold_rows[-1]['bacc']:.4f}")
    if np.any(predictions < 0):
        raise RuntimeError("Incomplete CATR MLP predictions")

    old = pd.read_csv(OLD_MLP_PREDICTIONS, dtype=str)
    base_map = old.set_index("stem")["prediction_numeric_mlp"]
    base_labels = merged["stem"].map(base_map)
    if base_labels.isna().any():
        raise ValueError("Missing old numeric OOF predictions")
    base_predictions = encoder.transform(base_labels.to_numpy(dtype=str))
    base_metrics = prediction_metrics(labels, base_predictions)
    metrics = prediction_metrics(labels, predictions)
    placebo_deltas = pd.read_csv(OLD_PLACEBOS)["delta_bacc_vs_numeric"].to_numpy()
    base_correct = base_predictions == labels
    text_correct = predictions == labels
    text_only = int(np.sum(text_correct & ~base_correct))
    base_only = int(np.sum(~text_correct & base_correct))
    discordant = text_only + base_only
    delta = metrics["bacc_oof"] - base_metrics["bacc_oof"]
    folds = pd.DataFrame(fold_rows)
    pd.DataFrame([{
        "condition": "CATR COCO human",
        "n_images": len(merged),
        **metrics,
        "numeric_bacc_oof": base_metrics["bacc_oof"],
        "delta_bacc_vs_numeric": delta,
        "bacc_fold_mean": folds["bacc"].mean(),
        "bacc_fold_std": folds["bacc"].std(ddof=0),
        "text_only_correct": text_only,
        "base_only_correct": base_only,
        "mcnemar_exact_p": (
            float(binomtest(min(text_only, base_only), discordant, 0.5).pvalue)
            if discordant else 1.0
        ),
        "placebo_delta_mean": placebo_deltas.mean(),
        "placebo_delta_p975": np.quantile(placebo_deltas, 0.975),
        "placebo_delta_max": placebo_deltas.max(),
        "empirical_p_placebo_ge_real": (
            1 + np.sum(placebo_deltas >= delta)
        ) / (len(placebo_deltas) + 1),
    }]).to_csv(OUTPUT_MLP, index=False)
    folds.to_csv(OUTPUT_MLP_FOLDS, index=False)
    pd.DataFrame({
        "stem": merged["stem"],
        "label": encoder.inverse_transform(labels),
        "prediction_numeric_mlp": encoder.inverse_transform(base_predictions),
        "prediction_catr_coco": encoder.inverse_transform(predictions),
    }).to_csv(OUTPUT_MLP_PREDICTIONS, index=False)


def leakage_audit(frame: pd.DataFrame) -> None:
    metadata = pd.read_csv(
        QWEN_AUDIT,
        usecols=["stem", "artist", "title_normalized"],
        dtype=str,
    ).drop_duplicates("stem")
    audit = frame[["stem", "label", "caption"]].merge(
        metadata, on="stem", how="left", validate="one_to_one"
    )
    rows = []
    for item in audit.itertuples(index=False):
        text = normalize(item.caption)
        true_terms = [
            term for term in STYLE_ALIASES[str(item.label)]
            if contains_phrase(text, normalize(term))
        ]
        style_terms = [
            term for term in ALL_STYLE_TERMS
            if contains_phrase(text, normalize(term))
        ]
        artist = normalize(item.artist)
        artist_parts = artist.split()
        surname = artist_parts[-1] if len(artist_parts) >= 2 else ""
        title = normalize(item.title_normalized)
        rows.append({
            "stem": item.stem,
            "true_style": item.label,
            "caption": item.caption,
            "true_style_mention": bool(true_terms),
            "true_style_terms": "|".join(true_terms),
            "any_style_mention": bool(style_terms),
            "style_terms": "|".join(style_terms),
            "full_artist_mention": bool(artist and contains_phrase(text, artist)),
            "surname_mention": bool(
                surname and len(surname) >= 4 and contains_phrase(text, surname)
            ),
            "exact_title_phrase": bool(
                title and len(title.split()) >= 2 and contains_phrase(text, title)
            ),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_LEAKAGE, index=False)
    count_columns = [
        "true_style_mention", "any_style_mention", "full_artist_mention",
        "surname_mention", "exact_title_phrase",
    ]
    summary = {"n": len(result)}
    for column in count_columns:
        summary[f"{column}_count"] = int(result[column].sum())
        summary[f"{column}_pct"] = 100 * float(result[column].mean())
    pd.DataFrame([summary]).to_csv(OUTPUT_LEAKAGE_SUMMARY, index=False)


def plot_confusion(
    labels: np.ndarray, predictions: np.ndarray, classes: list[str]
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=classes, normalize="true")
    displayed = [label.replace("_", " ") for label in classes]
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(OUTPUT_CONFUSION)
    figure, axis = plt.subplots(figsize=(14, 12), constrained_layout=True)
    image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(classes)), displayed, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(classes)), displayed)
    axis.tick_params(labelsize=7)
    axis.set_xlabel("Estilo predito")
    axis.set_ylabel("Estilo verdadeiro")
    axis.set_title(
        "CATR COCO — texto isolado, matriz OOF normalizada\n"
        f"Balanced accuracy = {balanced_accuracy_score(labels, predictions):.3f}"
    )
    figure.colorbar(image, ax=axis, label="Proporção na classe verdadeira")
    figure.savefig(OUTPUT_CONFUSION_PNG, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    frame = load_frame()
    labels, predictions, classes = standalone(frame)
    leakage_audit(frame)
    plot_confusion(labels, predictions, classes)
    run_numeric_mlp(frame)
    log(f"CATR evaluation completed in {time.perf_counter() - started:.1f}s")
    for path in (
        OUTPUT_STANDALONE, OUTPUT_MLP, OUTPUT_LEAKAGE_SUMMARY,
        OUTPUT_CONFUSION_PNG,
    ):
        log(f"Saved: {path}")


if __name__ == "__main__":
    main()
