"""Escores TF-IDF de legenda por dataset, no protocolo OOF do projeto.

Generaliza `tfidf_caption_style.py`, que tem coorte, caminhos e saídas fixos no
WikiArt. Aqui o dataset é argumento, para que Text exista nos corpora novos.

Protocolo idêntico ao original, e cada peça está lá por um motivo medido:

* **OOF por fold**: cada imagem é escorada por um `LinearSVC` que não a viu.
  Sem isso o bloco Text vaza rótulo para o classificador a jusante.
* **`class_weight="balanced"`**: os escores viram feature sob acurácia
  balanceada; sem a correção o classificador herda o prior e as classes raras
  somem (é o mesmo mecanismo que o log-lift corrige no bloco de grafo).
* **stop words derivadas dos nomes das classes**: sem isso o vetorizador pode
  casar o nome do estilo que aparece por acaso na legenda, o que seria
  vazamento pelo texto.

Ressalva que precisa acompanhar qualquer número que use este bloco: as
legendas vêm de um captioner treinado no MS COCO, então **Text depende de
supervisão externa** e fica fora do número principal do paper.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_external_modalities import LABELS  # noqa: E402

SEED, N_FOLDS = 42, 5
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 60_000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def clean_caption(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def make_stop_words(classes: np.ndarray) -> list[str]:
    """Palavras dos nomes de estilo: casá-las seria vazar o rótulo pelo texto."""
    words = set()
    for label in classes:
        for word in clean_caption(label.replace("_", " ")).split():
            if len(word) > 2:
                words.add(word)
    return sorted(words)


SUFIXOS = ("isms", "ism", "ists", "ist", "esque", "ique", "ic", "al", "e")


def style_roots(stop_words: list[str], min_len: int = 4) -> list[str]:
    """Radicais dos nomes de estilo, por remocao de sufixo declarada.

    As stop words casam **token exato**, entao formas derivadas escapam: o
    §5.18b mediu que `surreal` passa quando `surrealism` e a stop word. O
    Snowball nao resolve -- ele leva `impressionism` a `impression` e
    `impressionistic` a `impressionist`, separando justamente o par `-ism`/
    `-ist` que interessa, e funde `naturalism` com `natural`.

    A regra abaixo erra para o lado de remover DEMAIS (`mannerism` derruba
    `manner`, `naturalism` derruba `natural`), o que e o lado conservador para
    um controle de vazamento: se o numero sobrevive a esta poda, ele sobrevive
    a qualquer poda mais cirurgica.
    """
    roots = set()
    for word in stop_words:
        root = word
        for sufixo in SUFIXOS:
            if word.endswith(sufixo) and len(word) - len(sufixo) >= min_len:
                root = word[: -len(sufixo)]
                break
        roots.add(root)
    return sorted(roots)


def expand_stop_words(stop_words: list[str], texts: np.ndarray) -> list[str]:
    """Stop words + todo token do vocabulario que comece por um radical."""
    roots = tuple(style_roots(stop_words))
    vocab = set()
    for text in texts:
        vocab.update(text.split())
    extra = {token for token in vocab if token.startswith(roots)}
    return sorted(set(stop_words) | extra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(LABELS))
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/text_scores")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES,
                        help="teto de termos do TF-IDF; o default reproduz tudo "
                             "que ja foi medido")
    parser.add_argument("--stopword-mode", choices=("exact", "root"), default="exact",
                        help="exact reproduz todo resultado anterior; root poda "
                             "tambem as formas derivadas (controle de vazamento)")
    args = parser.parse_args()

    meta_path, label_column = LABELS[args.dataset]
    meta = pd.read_csv(ROOT / meta_path, dtype={"stem": str}).dropna(subset=[label_column])
    meta["label"] = meta[label_column].astype(str)

    captions = {}
    with args.captions.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            captions[str(row["stem"])] = row["caption"]
    frame = meta[["stem", "label"]].copy()
    frame["caption"] = frame["stem"].map(captions)
    before = len(frame)
    frame = frame.dropna(subset=["caption"])
    if len(frame) != before:
        log(f"{before - len(frame):,} imagens sem legenda, removidas")

    counts = frame["label"].value_counts()
    if args.classes_file:
        def normalize(value: str) -> str:
            return value.replace("_", " ").replace("-", " ").replace("ï", "i").strip().lower()
        wanted = {normalize(line) for line in args.classes_file.read_text().splitlines() if line.strip()}
        by_norm = {normalize(label): label for label in counts.index}
        frame = frame[frame["label"].isin([by_norm[n] for n in wanted])]
    elif args.min_per_class:
        frame = frame[frame["label"].isin(counts[counts >= args.min_per_class].index)]

    frame = frame.sort_values("stem").reset_index(drop=True)
    frame["clean"] = frame["caption"].map(clean_caption)
    frame = frame[frame["clean"].str.len() > 0].reset_index(drop=True)

    texts = frame["clean"].to_numpy(dtype=str)
    labels = frame["label"].to_numpy(dtype=str)
    classes = np.unique(labels)
    index_of = {label: index for index, label in enumerate(classes)}
    stop_words = make_stop_words(classes)
    log(f"{args.dataset}: {len(frame):,} legendas, {len(classes)} classes, "
        f"{len(stop_words)} stop words de nome de estilo")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(frame), dtype=np.uint8), labels))
    decisions = np.zeros((len(frame), len(classes)), dtype=np.float32)
    scores = []
    for fold, (train, test) in enumerate(splits, 1):
        # a expansao sai do vocabulario de TREINO, nunca da coorte inteira:
        # derivar do corpus todo seria transdutivo (§10.6), ainda que sem rotulo.
        fold_stop = (expand_stop_words(stop_words, texts[train])
                     if args.stopword_mode == "root" else stop_words)
        if args.stopword_mode == "root" and fold == 1:
            log(f"  modo root: {len(stop_words)} -> {len(fold_stop):,} stop words")
        vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=args.max_features,
                                     binary=True, sublinear_tf=True,
                                     stop_words=fold_stop, min_df=2)
        train_x = vectorizer.fit_transform(texts[train])
        model = LinearSVC(class_weight="balanced", random_state=SEED)
        model.fit(train_x, labels[train])
        decision = model.decision_function(vectorizer.transform(texts[test]))
        aligned = np.zeros((len(test), len(classes)), dtype=np.float32)
        for local, label in enumerate(model.classes_):
            aligned[:, index_of[label]] = decision[:, local]
        decisions[test] = aligned
        bacc = balanced_accuracy_score(labels[test], model.predict(vectorizer.transform(texts[test])))
        scores.append(bacc)
        log(f"  fold {fold}: text-only BAcc={100 * bacc:.2f} "
            f"({train_x.shape[1]:,} termos)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.dataset
    np.save(f"{stem}_decisions.npy", decisions)
    np.save(f"{stem}_stems.npy", frame["stem"].to_numpy(dtype=str))
    np.save(f"{stem}_classes.npy", classes)
    log(f"text-only BAcc: {100 * np.mean(scores):.2f} ± {100 * np.std(scores, ddof=1):.2f}")
    log(f"salvo: {stem}_decisions.npy ({decisions.shape})")


if __name__ == "__main__":
    main()
