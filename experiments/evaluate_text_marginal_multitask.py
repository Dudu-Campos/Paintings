"""§7.2: o canal de texto acrescenta alguma coisa ao pipeline final?

O numero antigo do WikiArt (+6,46pp) veio de um captioner do ArtEmis ajustado
nas anotacoes das proprias pinturas. A versao limpa (CATR/COCO) rendeu +0,16pp
n.s. no MLP do OilPainting, e ficaram duas hipoteses sem separar: ou a base
mudou (o grafo ja ocupa o papel de bloco tipo-posterior), ou as legendas
contaminadas inflavam o numero antigo. Agora ha um terceiro captioner -- o
estudante CLIP+decoder -- que bate o CATR por +7,10pp sozinho (16,08 x 8,98 em
125 classes). Se nem ele mover o marginal, a resposta e a base, nao o captioner.

Base: `hand-8 + grafo`, que e a configuracao de producao do §3 (SEM texto).
Cabeca: BranchMLP estreito, a versao final do §7.8.

**OOF ANINHADO e obrigatorio.** O `build_text_scores.py` produz OOF simples
sobre a coorte inteira: para o fold externo k, os escores das linhas de TREINO
vieram de modelos que viram os rotulos do fold k. O MLP entao aprende sobre
features contaminadas pelo teste. E o mesmo mecanismo que derrubou o SNF (§6),
so que diluido. Aqui cada fold externo refaz o TF-IDF em 5 folds internos.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from build_text_scores import (MAX_FEATURES, NGRAM_RANGE, clean_caption,  # noqa: E402
                               make_stop_words)
from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_preprocess_effect_oilpainting import fit  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS, INNER_FOLDS = 42, 5, 5
CAPTIONS = {
    "estudante": ROOT / "results/multitask_student_captions.jsonl",
    "catr": ROOT / "outputs/affect_captions/catr_coco_multitask.jsonl",
}


def load_captions(path: Path, stems: np.ndarray) -> np.ndarray:
    """Legendas na ordem da coorte. Falha alto se faltar alguma: o §8 registra
    quatro extratores que gravaram menos imagens que o esperado em silencio."""
    table = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            table[str(row["stem"])] = clean_caption(row["caption"])
    texts = np.array([table.get(s, "") for s in stems], dtype=object)
    missing = int((texts == "").sum())
    if missing:
        raise ValueError(f"{path.name}: {missing:,} de {len(stems):,} stems sem legenda")
    return texts


def fit_text(texts, labels, stop_words):
    vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES,
                                 binary=True, sublinear_tf=True,
                                 stop_words=stop_words, min_df=2)
    matrix = vectorizer.fit_transform(texts)
    model = LinearSVC(class_weight="balanced", random_state=SEED)
    model.fit(matrix, labels)
    return vectorizer, model


def align(model, decision, classes, index_of):
    out = np.zeros((decision.shape[0], len(classes)), dtype=np.float32)
    for local, label in enumerate(model.classes_):
        out[:, index_of[label]] = decision[:, local]
    return out


def nested_text_block(texts, labels_str, classes, index_of, stop_words,
                      train_idx, test_idx, outer_fold) -> np.ndarray:
    """Bloco de texto de comprimento total: linhas de treino recebem escore de
    OOF interno, linhas de teste recebem escore do modelo do treino externo."""
    block = np.zeros((len(texts), len(classes)), dtype=np.float32)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + outer_fold)
    for k, (rel_fit, rel_val) in enumerate(
            inner.split(np.zeros(len(train_idx), dtype=np.uint8), labels_str[train_idx]), 1):
        fit_idx, val_idx = train_idx[rel_fit], train_idx[rel_val]
        vectorizer, model = fit_text(texts[fit_idx], labels_str[fit_idx], stop_words)
        block[val_idx] = align(model, model.decision_function(vectorizer.transform(texts[val_idx])),
                               classes, index_of)
        log(f"    inner {k}/{INNER_FOLDS}")
    vectorizer, model = fit_text(texts[train_idx], labels_str[train_idx], stop_words)
    block[test_idx] = align(model, model.decision_function(vectorizer.transform(texts[test_idx])),
                            classes, index_of)
    return block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="multitask")
    ap.add_argument("--captions", nargs="+", default=["estudante", "catr"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path, default=ROOT / "results/multitask_text_marginal.csv")
    args = ap.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels_str = np.asarray(cohort.labels, dtype=str)
    labels = LabelEncoder().fit_transform(labels_str)
    n_classes = len(np.unique(labels))
    hand = [cohort.blocks[n] for n in HAND8]
    graph_source = np.hstack(hand)
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    stop_words = make_stop_words(classes)
    log(f"{args.dataset}: {len(stems):,} imagens, {n_classes} classes, "
        f"hand-8 {graph_source.shape[1]}d, {len(stop_words)} stop words")

    texts = {name: load_captions(CAPTIONS[name], stems) for name in args.captions}
    for name, t in texts.items():
        log(f"  {name}: {len(t):,} legendas, {np.mean([len(x.split()) for x in t]):.1f} tokens")

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))

    def record(condition, number, blocks, source):
        if (condition, number) in done:
            return
        started = time.perf_counter()
        tr, te = folds[number - 1]
        metrics = fit(blocks, labels, tr, te, n_classes, SEED,
                      args.epochs, args.patience, "narrow", source)
        row = {"condition": condition, "fold": number,
               "seconds": time.perf_counter() - started, **metrics}
        frame = pd.DataFrame([row])
        if args.output.exists():
            frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
        frame.to_csv(args.output, mode="a", index=False,
                     header=not args.output.exists(), float_format="%.4f")
        log(f"{condition:28s} fold {number} | {row['dimensions']:5d}d | "
            f"BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    for number, (train_idx, test_idx) in enumerate(folds, 1):
        record("hand-8 + grafo", number, hand, graph_source)
        for name in args.captions:
            condition = f"hand-8 + grafo + texto ({name})"
            if (condition, number) in done:
                continue
            log(f"  fold {number}: OOF aninhado do texto ({name})")
            block = nested_text_block(texts[name], labels_str, classes, index_of,
                                      stop_words, train_idx, test_idx, number)
            record(condition, number, hand + [block], graph_source)


if __name__ == "__main__":
    main()
