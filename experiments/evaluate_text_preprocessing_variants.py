"""O canal de texto melhora com pre-processamento melhor? E fecha o gap do bipartido?

Diagnostico que motivou (31/ago, sobre as legendas do estudante no Multitask):

    stop words do ingles:    101 termos,  4,6% da massa TF-IDF   <- pouco
    unigramas:             1.801 termos, 36,4%
    bigramas:             14.715 termos, 63,6%                   <- muito

    top por massa: the / image / the image / with / of / sense /
                   sense of / depicts / image depicts / in / colors

As palavras de TEMPLATE do decoder (`image`, `depicts`, `sense of`) dominam e nao
sao stop words do ingles -- o `build_text_scores` so tem `min_df=2` e NENHUM
`max_df`, entao elas passam inteiras. E bigramas sao 64% da massa: para um
esquema de contagem (bipartido) cada bigrama tem a linha palavra->classe estimada
com poucos documentos, o que e ruido com peso alto. O SVM regulariza; a
propagacao nao.

Roda os DOIS metodos em cada variante, para a comparacao continuar pareada.

Referencia atual (5 folds, pre-processamento de producao):
    LinearSVC          16,30
    bipartido 1 passo   7,46
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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_text_scores import MAX_FEATURES, clean_caption, make_stop_words  # noqa: E402
from evaluate_bipartite_text_graph import bipartite, prior_lift  # noqa: E402
from evaluate_external_modalities import LABELS  # noqa: E402

SEED = 42

# (ngram, min_df, max_df, stop words do ingles, stemming)
VARIANTS = {
    "produção (1-2gram, sem max_df)": dict(ngram=(1, 2), max_df=1.0, english=False, stem=False),
    "+ max_df=0.5":                   dict(ngram=(1, 2), max_df=0.5, english=False, stem=False),
    "só unigrama":                    dict(ngram=(1, 1), max_df=1.0, english=False, stem=False),
    "unigrama + max_df=0.5":          dict(ngram=(1, 1), max_df=0.5, english=False, stem=False),
    "+ stop-EN + max_df=0.5":         dict(ngram=(1, 2), max_df=0.5, english=True, stem=False),
    "+ stem + max_df=0.5":            dict(ngram=(1, 2), max_df=0.5, english=False, stem=True),
    "tudo":                           dict(ngram=(1, 2), max_df=0.5, english=True, stem=True),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="multitask")
    ap.add_argument("--captions", type=Path,
                    default=ROOT / "results/multitask_student_captions.jsonl")
    ap.add_argument("--folds", type=int, default=2, help="folds (dos 5 canônicos)")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/text_preprocessing_variants.csv")
    args = ap.parse_args()

    meta_path, column = LABELS[args.dataset]
    meta = pd.read_csv(ROOT / meta_path, dtype={"stem": str}).dropna(subset=[column])
    meta = meta.drop_duplicates("stem", keep="last")
    table = {}
    with args.captions.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            table[str(row["stem"])] = row["caption"]
    frame = meta[["stem", column]].rename(columns={column: "label"})
    frame["caption"] = frame["stem"].astype(str).map(table)
    frame = frame.dropna(subset=["caption"]).sort_values("stem").reset_index(drop=True)
    frame["clean"] = frame["caption"].map(clean_caption)
    frame = frame[frame["clean"].str.len() > 0].reset_index(drop=True)

    labels_str = frame["label"].to_numpy(dtype=str)
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    labels = np.array([index_of[v] for v in labels_str])
    n_classes = len(classes)
    style_stop = make_stop_words(classes)

    plain = frame["clean"].to_numpy(dtype=str)
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("english")
    cache: dict[str, str] = {}

    def stem_text(text: str) -> str:
        out = []
        for word in text.split():
            hit = cache.get(word)
            if hit is None:
                hit = cache[word] = stemmer.stem(word)
            out.append(hit)
        return " ".join(out)

    t0 = time.perf_counter()
    stemmed = np.array([stem_text(t) for t in plain], dtype=object)
    log(f"{len(frame):,} legendas, {n_classes} classes | stemming {time.perf_counter()-t0:.0f}s, "
        f"{len(cache):,} tipos")

    rows = []
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(plain, labels))
    for name, cfg in VARIANTS.items():
        texts = stemmed if cfg["stem"] else plain
        stop = sorted(set(style_stop) | set(ENGLISH_STOP_WORDS)) if cfg["english"] else style_stop
        if cfg["stem"]:
            stop = sorted({stemmer.stem(w) for w in stop})
        for fold, (train, test) in enumerate(folds[:args.folds], 1):
            vec = TfidfVectorizer(ngram_range=cfg["ngram"], max_features=MAX_FEATURES,
                                  binary=True, sublinear_tf=True, stop_words=stop,
                                  min_df=2, max_df=cfg["max_df"])
            train_x = vec.fit_transform(texts[train].astype(str))
            test_x = vec.transform(texts[test].astype(str))

            t0 = time.perf_counter()
            scores, _ = bipartite(train_x, labels[train], test_x, n_classes, 1)
            t_bip = time.perf_counter() - t0
            bip = 100 * balanced_accuracy_score(
                labels[test], prior_lift(np.maximum(scores, 0), labels[train], n_classes).argmax(1))

            t0 = time.perf_counter()
            model = LinearSVC(class_weight="balanced", random_state=SEED).fit(train_x, labels[train])
            t_svm = time.perf_counter() - t0
            svm = 100 * balanced_accuracy_score(labels[test], model.predict(test_x))

            rows.append(dict(variante=name, fold=fold, termos=train_x.shape[1],
                             bipartido=bip, linearsvc=svm,
                             s_bipartido=t_bip, s_svm=t_svm))
            log(f"{name:32s} fold {fold} | {train_x.shape[1]:6,}t | "
                f"SVM {svm:5.2f} ({t_svm:4.0f}s) | bipartido {bip:5.2f} ({t_bip:4.1f}s)")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    out = pd.DataFrame(rows).groupby("variante").agg(
        termos=("termos", "mean"), LinearSVC=("linearsvc", "mean"),
        bipartido=("bipartido", "mean"), s_svm=("s_svm", "mean"))
    print("\n" + out.round(2).to_string())


if __name__ == "__main__":
    main()
