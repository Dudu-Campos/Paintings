"""Cada modalidade sozinha, sob cinco classificadores.

Primeira tabela da ablação: antes de perguntar o que a fusão acrescenta, quanto
cada representação vale por conta própria -- e se essa ordenação depende do
classificador. O §8 registra que LDA e MLP concordam em <=0,35pp em vetores
densos e divergem ~9pp em multi-bloco; aqui cada bloco é denso e homogêneo, então
a expectativa é concordância, e discordância seria achado.

Cinco classificadores, todos com peso de classe onde o estimador aceita, porque
a métrica é acurácia balanceada:

  LDA   sonda linear; teto de posto em n_classes-1 = 24
  MLP   `PlainMLP` sem branches (§5.14) -- com um bloco só não há o que separar
  XGB   histograma, 200 rodadas (= 200 x 25 árvores), CPU
  RF    300 árvores, class_weight balanceado por subamostra
  SVM   RBF, C=10, gamma="scale"

**O SVM roda em subamostra e isso não é opcional.** O §6 mede 19 minutos para
UM fold sobre o banco de 639d, com 94% dos pontos virando vetores de suporte; o
custo é O(n^2) no treino e O(n) na predição. Com 66k linhas x 8 modalidades x 5
folds seria mais de um dia. A subamostra estratificada é declarada, e para que
a barra dele seja interpretável **LDA e MLP também rodam na mesma subamostra** --
sem esse par a penalidade de subamostragem fica confundida com a diferença entre
classificadores.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_backward_elimination import EXTERNAL  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def metrics_of(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return {"bacc": 100 * balanced_accuracy_score(truth, prediction),
            "accuracy": 100 * accuracy_score(truth, prediction),
            "macro_f1": 100 * f1_score(truth, prediction, average="macro")}


def run_lda(train_x, train_y, test_x, test_y, n_classes, seed):
    model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
    model.fit(train_x, train_y)
    return metrics_of(test_y, model.predict(test_x))


def run_rf(train_x, train_y, test_x, test_y, n_classes, seed):
    model = RandomForestClassifier(n_estimators=300, n_jobs=16, random_state=seed,
                                   class_weight="balanced_subsample")
    model.fit(train_x, train_y)
    return metrics_of(test_y, model.predict(test_x))


def run_xgb(train_x, train_y, test_x, test_y, n_classes, seed):
    import xgboost as xgb
    # Peso por amostra em vez de class_weight: o xgboost não tem o parâmetro, e
    # sem ele a comparação com os outros quatro deixa de ser em acurácia
    # balanceada.
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    weight = (1.0 / np.maximum(counts, 1))[train_y]
    # CPU, não GPU: medido nesta máquina, `device="cuda"` com entrada numpy fica
    # 2x MAIS LENTO (45,2s contra 22,5s em n=100) por transferência host->device
    # a cada iteração, e ainda disputa a placa com a grade de fine-tuning.
    # n_estimators=200 porque com 25 classes cada rodada constrói 25 árvores.
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              tree_method="hist", n_jobs=16, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed,
                              verbosity=0, objective="multi:softmax")
    model.fit(train_x, train_y, sample_weight=weight / weight.mean())
    return metrics_of(test_y, model.predict(test_x))


def run_svm(train_x, train_y, test_x, test_y, n_classes, seed):
    model = SVC(C=10.0, gamma="scale", class_weight="balanced", cache_size=1000)
    model.fit(train_x, train_y)
    return metrics_of(test_y, model.predict(test_x))


def run_knn(train_x, train_y, test_x, test_y, n_classes, seed, k=30):
    """Voto de k vizinhos, ponderado por 1/n_c.

    Duas escolhas que precisam ficar declaradas.

    **k=30 e o mesmo do grafo de similaridade** (§5.7), entao este ponto e
    comparavel a vizinhanca que o operador do §5.2 usa -- so que por bloco, em
    vez de sobre as 617 dims concatenadas. O espaco tambem e o mesmo: o
    `StandardScaler` por fold ja roda antes de chamar o runner.

    **O voto e ponderado por frequencia inversa**, que e o analogo direto do
    `class_weight="balanced"` que os outros quatro runners usam (LDA com priors
    uniformes, RF e SVM com class_weight, XGB com sample_weight). Sem isso, com
    desbalanceamento de 1052x, a BAcc mediria o desbalanceamento e nao o
    estimador -- e a comparacao com os outros deixaria de ser pareada.

    Nao ha ajuste: e o unico dos cinco sem fase de treino, o que amplia o
    intervalo de vies indutivo coberto pela figura.
    """
    from sklearn.neighbors import NearestNeighbors

    counts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    weight = 1.0 / np.maximum(counts, 1.0)
    index = NearestNeighbors(n_neighbors=min(k, len(train_y)), n_jobs=16).fit(train_x)
    prediction = np.empty(len(test_x), dtype=train_y.dtype)
    # Em lotes: a matriz densa consulta x treino nao cabe de uma vez em 125
    # classes (20k x 80k x 8 bytes = 12,8 GB).
    for start in range(0, len(test_x), 4096):
        chunk = slice(start, start + 4096)
        neighbours = index.kneighbors(test_x[chunk], return_distance=False)
        votes = weight[train_y[neighbours]]
        tally = np.zeros((neighbours.shape[0], n_classes), dtype=np.float64)
        np.add.at(tally, (np.arange(neighbours.shape[0])[:, None], train_y[neighbours]), votes)
        prediction[chunk] = tally.argmax(axis=1)
    return metrics_of(test_y, prediction)


def run_mlp(train_x, train_y, test_x, test_y, n_classes, seed):
    from evaluate_redundancy_selection_curve import fit as plain_fit
    # Em 125 classes a subamostra do SVM (15k linhas, desbalanceamento de 1052x)
    # deixa as classes mais raras com UM exemplo, e o split estratificado de
    # validacao falha. Nao acontece em 25 classes. Cair para split simples e
    # correto aqui: a estratificacao e conveniencia da divisao interna de
    # validacao, nao parte do protocolo -- os folds externos continuam
    # estratificados.
    stratify = train_y if np.bincount(train_y, minlength=n_classes).min() >= 2 else None
    fit_idx, val_idx = train_test_split(
        np.arange(len(train_y)), test_size=0.1, stratify=stratify, random_state=seed)
    out = plain_fit(train_x[fit_idx], train_y[fit_idx], train_x[val_idx], train_y[val_idx],
                    test_x, test_y, n_classes, seed=seed)
    return {k: out[k] for k in ("bacc", "accuracy", "macro_f1")}


RUNNERS = {"LDA": run_lda, "MLP": run_mlp, "XGBoost": run_xgb,
           "RandomForest": run_rf, "SVM": run_svm, "kNN": run_knn}
SUBSAMPLED = {"SVM"}

# Views que existem na figura mas NAO na configuracao final, cada uma por um
# motivo diferente.
#
# ATENCAO (§5.9b): `Caption embedding` e o **embedding** da legenda (384d), que o
# §5.9 mede como ortogonal a todo o banco visual (r² max 0,068) e com marginal de
# **-0,09pp**. Ele NAO e o canal de texto do §5.18 (TF-IDF -> LinearSVC -> 125
# escores, +2,31pp). Os dois compartilhavam o rotulo `Text`, e isso fazia o leitor
# concluir que texto e ortogonal e inutil quando o inutil e o embedding.
#
# O canal de texto de verdade NAO entra aqui de proposito: o standalone dele e o
# do proprio LinearSVC (16,08 BAcc, §5.18). Passar os 125 escores OOF por um
# segundo estagio de LDA/MLP/XGB seria outro objeto, e vazaria -- os escores sao
# OOF sobre a coorte INTEIRA, entao as linhas de treino do segundo estagio vem de
# modelos que viram o fold de teste. Reportar 16,08 a parte, com a nota de que a
# sonda dele e outra.
EXTERNAL_SUPERVISION = {
    "Caption embedding": ("outputs/multitask_caption_embeddings/caption_embed_features.npy",
                          "outputs/multitask_caption_embeddings/caption_embed_stems.npy"),
}

# `Similarity graph` nao e bloco estatico: e calculado POR FOLD com OOF aninhado
# sobre o espaco visual concatenado. "Sozinho" significa outra coisa para ele --
# e um operador sobre as outras views, nao uma medida independente da imagem.
# `Similarity graph (class scores)` sao as 2*C primeiras colunas dele (log-lift e
# histograma ponderado, sem as 32 estatisticas); sai de graca porque a construcao
# do grafo ja foi paga, e separa "identidade de classe" de "confianca de
# vizinhanca", que o §6 mede valerem coisas muito diferentes.
PER_FOLD = {"Similarity graph", "Similarity graph (class scores)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--classifiers", nargs="+", default=list(RUNNERS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--svm-train", type=int, default=15000,
                        help="linhas de treino para o SVM e para o par de "
                             "controle LDA/MLP na mesma subamostra")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/standalone_by_classifier.csv")
    args = parser.parse_args()

    blocks, raw_labels, _ = load_corpus(args.dataset, VISUAL + ["Gabor"])
    if args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
    else:
        # `--classes-file none` roda a coorte inteira (multitask125). O recorte
        # de 25 é o benchmark do §7.11; sem ele o LDA deixa de ficar preso em
        # 24 componentes, que é a ressalva do §5.1. `wanted` vira o conjunto
        # completo porque ele é reusado adiante para recortar o metadado.
        wanted = {key(v) for v in raw_labels}
        mask = np.ones(len(raw_labels), dtype=bool)
    if args.classes_file.exists() and len(np.unique(raw_labels[mask])) != len(wanted):
        raise ValueError("casamento de classes do benchmark incompleto")
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels[mask])
    n_classes = len(np.unique(labels))

    # Random walk e HVG não estão no registro; alinhar por stem contra a ordem
    # de linhas da coorte, que é a do metadado filtrado.
    meta = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})
    reference = meta[[key(str(v)) in wanted for v in meta["style"]]].stem.to_numpy()
    if len(reference) != len(labels):
        raise ValueError(f"coorte ({len(labels)}) != metadado ({len(reference)})")
    for name, (values_path, stems_path) in EXTERNAL.items():
        values = np.load(ROOT / values_path)
        stems = np.load(ROOT / stems_path, allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        if index.isna().any():
            raise ValueError(f"{name}: {int(index.isna().sum())} stems ausentes")
        blocks[name] = clean(values[index.to_numpy(dtype=int)])

    for name, (values_path, stems_path) in EXTERNAL_SUPERVISION.items():
        values = np.load(ROOT / values_path)
        stems = np.load(ROOT / stems_path, allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        if index.isna().any():
            raise ValueError(f"{name}: {int(index.isna().sum())} stems ausentes")
        blocks[name] = clean(values[index.to_numpy(dtype=int)])

    visual = [n for n in VISUAL + ["Gabor"] if n in blocks]
    order = visual + list(EXTERNAL) + list(EXTERNAL_SUPERVISION) + sorted(PER_FOLD)
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, {len(order)} blocos")

    rows, done = [], set()
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {(str(r["block"]), str(r["classifier"]), int(r["fold"]), bool(r["subsampled"]))
                for r in rows}

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]

    for name in order:
        for fold, (train, test) in enumerate(splits, start=1):
            if name in PER_FOLD:
                # Construído dentro do fold: o grafo vê só linhas de treino, e
                # as de treino recebem o valor OOF aninhado. Pré-computar uma
                # vez vazaria o teste (§6: o SNF caiu para 3,4% assim).
                from evaluate_production_visual_extended_similarity_graph import (
                    prior_correct_rich, rich_similarity_features)
                space = np.hstack([StandardScaler().fit(blocks[n][train])
                                   .transform(blocks[n]).astype(np.float32) for n in visual])
                oof, target = rich_similarity_features(
                    space[train], labels[train], space[test], n_classes,
                    30, 0.4, 100, SEED + fold)
                values = np.zeros((len(labels), oof.shape[1]), dtype=np.float32)
                values[train] = prior_correct_rich(oof, labels[train], n_classes)
                values[test] = prior_correct_rich(target, labels[train], n_classes)
                if "class scores" in name:
                    values = values[:, :2 * n_classes]
            else:
                values = blocks[name]
            scaler = StandardScaler().fit(values[train])
            scaled = scaler.transform(values).astype(np.float32)
            # Subamostra fixa por fold: o SVM e o par de controle têm de ver
            # exatamente as mesmas linhas, senão a penalidade medida é ruído.
            small, _ = train_test_split(train, train_size=min(args.svm_train, len(train) - 1),
                                        stratify=labels[train], random_state=SEED + fold)
            for classifier in args.classifiers:
                for subsampled in ({True} if classifier in SUBSAMPLED
                                   else {False} | ({True} if classifier in ("LDA", "MLP") else set())):
                    if (name, classifier, fold, subsampled) in done:
                        continue
                    index = small if subsampled else train
                    started = time.perf_counter()
                    result = RUNNERS[classifier](scaled[index], labels[index],
                                                 scaled[test], labels[test],
                                                 n_classes, SEED + fold)
                    rows.append({"block": name, "dimensions": values.shape[1],
                                 "classifier": classifier, "fold": fold,
                                 "subsampled": subsampled, "n_train": len(index),
                                 "seconds": time.perf_counter() - started, **result})
                    log(f"  {name:<13}{classifier:<13}f{fold} "
                        f"{'sub' if subsampled else 'full':<5} "
                        f"BAcc {result['bacc']:5.2f}  ({rows[-1]['seconds']:.0f}s)")
                    pd.DataFrame(rows).to_csv(args.output, index=False)

    frame = pd.DataFrame(rows)
    full = frame[~frame.subsampled]
    log("\nBAcc por modalidade e classificador (coorte inteira):")
    log(full.pivot_table(index="block", columns="classifier", values="bacc").round(2).to_string())
    both = frame[frame.classifier.isin(["LDA", "MLP"])]
    penalty = (both[both.subsampled].groupby("classifier").bacc.mean()
               - both[~both.subsampled].groupby("classifier").bacc.mean())
    log(f"\npenalidade de subamostragem ({args.svm_train:,} linhas), para ler a barra do SVM:")
    log(penalty.round(2).to_string())


if __name__ == "__main__":
    main()
