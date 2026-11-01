"""Grafo ingênuo (uma distância sobre blocos concatenados) contra multiaresta.

A pergunta que isto decide: o kNN do grafo roda hoje sobre os 8 blocos
concatenados e padronizados, com **peso igual por dimensão**. As 130 dimensões
de Descriptors valem 4x as 32 de Gabor só por serem mais numerosas, e
`per_modality_neighbourhoods` já mediu que as vizinhanças das modalidades são
quase disjuntas (Jaccard mediano 0,008). Isso sugere multiaresta -- um grafo
por modalidade -- mas sugestão não é medição.

Duas partes, porque a segunda só interessa se a primeira sobreviver:

**Parte A, o confundidor de k.** O oráculo da união das 8 vizinhanças é 99,8%
contra 91,9% do concatenado, mas a união tem 8*k vizinhos e o concatenado tem k.
Comparar o concatenado em k e em 8k separa "há estrutura complementar" de "há
mais vizinhos". Se o concatenado em 8k empatar com a união, multiaresta não
compra teto nenhum e a Parte B fica sem motivação.

**Parte B, o desfecho.** Três codificações do mesmo grafo, no MLP de produção:

  - `naive`      : um kNN sobre os 639d concatenados -> 2C+32 = 82d (produção)
  - `multi-lean` : um kNN por modalidade, só a identidade de classe
                   (log-lift + weighted_hist = 2C cada) -> 8*50 = 400d
  - `multi-full` : um kNN por modalidade, as 2C+32 completas -> 8*82 = 656d

`multi-lean` existe porque o §6 já mediu que as 32 estatísticas sozinhas não
entregam nada -- o que o grafo dá é identidade de classe, não confiança de
vizinhança --, então replicar 32 estatísticas oito vezes é a parte do custo
dimensional com menor chance de pagar.

O risco declarado é diluição: o §5.6 registra que o bloco do grafo perde efeito
conforme cresce em relação à base (Δ +0,79pp com 82d em 25 classes, -0,03 n.s.
com 282d em 125 classes). 656d contra 639d de base é exatamente esse regime, e
é por isso que a resposta não é dedutível -- multiaresta compra informação e
paga em diluição, e só a medição diz qual domina.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from scipy.sparse import csr_matrix, diags  # noqa: E402

from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    INNER_FOLDS, encode_neighbourhood, prior_correct_rich, rich_similarity_features)


# --------------------------------------------------------------------- fusão
# O multiaresta por concatenação funde os RESULTADOS de oito difusões
# independentes. A literatura funde o GRAFO antes de difundir -- SNF por
# cross-difusão (Wang et al., Nature Methods 2014), difusão multiplex, ou
# combinação de Laplacianos (Karasuyama & Mamitsuka 2013). O que segue é a
# versão sem nenhum parâmetro aprendido: soma das oito afinidades kNN, uma
# difusão só. A saída volta a ser 2*C+32, então o teste separa as duas
# hipóteses do fracasso anterior -- se era diluição dimensional, esta variante
# recupera; se era falta de informação, ela também perde.


def view_affinity(train_x, query_x, k, exclude_self=False):
    """Afinidade esparsa de uma view, com escala do próprio grafo de treino.

    A escala não é cosmética: somar afinidades de um bloco de 130d com as de um
    de 32d sem normalizar deixa o de maior dimensão dominar a soma, que é
    exatamente o peso-igual-por-dimensão que o multiaresta existe para evitar.
    """
    model = NearestNeighbors(n_neighbors=k + (1 if exclude_self else 0)).fit(train_x)
    distances, indices = model.kneighbors(query_x)
    if exclude_self:
        distances, indices = distances[:, 1:], indices[:, 1:]
    train_distances, _ = model.kneighbors(train_x)
    scale = max(float(np.median(train_distances[:, 1:])), 1e-6)
    similarity = np.exp(-distances / scale)
    rows = np.repeat(np.arange(len(query_x)), similarity.shape[1])
    return csr_matrix((similarity.ravel(), (rows, indices.ravel())),
                      shape=(len(query_x), len(train_x)))


def top_k_rows(matrix, k):
    """Top-k por linha da afinidade somada -- a vizinhança fundida."""
    values = np.zeros((matrix.shape[0], k), dtype=np.float32)
    indices = np.zeros((matrix.shape[0], k), dtype=np.int64)
    for row in range(matrix.shape[0]):
        lo, hi = matrix.indptr[row], matrix.indptr[row + 1]
        data, column = matrix.data[lo:hi], matrix.indices[lo:hi]
        if len(data) > k:
            take = np.argpartition(-data, k - 1)[:k]
            data, column = data[take], column[take]
        order = np.argsort(-data)
        values[row, :len(data)] = data[order]
        indices[row, :len(column)] = column[order]
    return values, indices


def fused_diffusion(affinity, train_y, n_classes, alpha, max_iter):
    """Label spreading de Zhou et al. (2004) sobre a afinidade fundida.

    Escrito à mão porque o `LabelSpreading` do sklearn constrói o próprio grafo
    kNN internamente e não aceita afinidade de fora -- é precisamente o ponto
    onde a fusão precisa entrar.
    """
    weights = affinity.maximum(affinity.T)
    degree = np.asarray(weights.sum(axis=1)).ravel()
    inverse = diags(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    normalized = inverse @ weights @ inverse
    initial = np.eye(n_classes, dtype=np.float32)[train_y]
    posterior = initial.copy()
    for _ in range(max_iter):
        posterior = alpha * (normalized @ posterior) + (1 - alpha) * initial
    return posterior / np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)


def fused_query_features(train_blocks, train_y, query_blocks, n_classes, k,
                         alpha, max_iter):
    n_views = len(train_blocks)
    train_affinity = sum(view_affinity(x, x, k, exclude_self=True)
                         for x in train_blocks)
    posterior = fused_diffusion(train_affinity, train_y, n_classes, alpha, max_iter)

    query_affinity = sum(view_affinity(t, q, k)
                         for t, q in zip(train_blocks, query_blocks))
    similarity, indices = top_k_rows(query_affinity.tocsr(), k)
    # Dividir por n_views devolve a similaridade a [0,1], para que a distância
    # implícita -log(sim) seja positiva e as estatísticas de distância das 32
    # continuem com o mesmo significado que têm no grafo único.
    similarity = (similarity / n_views).astype(np.float32)
    distances = -np.log(np.clip(similarity, 1e-12, 1.0))
    propagated = posterior[indices].mean(axis=1).astype(np.float32)
    return encode_neighbourhood(distances, indices, similarity, propagated,
                                train_y, n_classes)


def fused_similarity_features(train_blocks, train_y, query_blocks, n_classes,
                              k, alpha, max_iter, seed):
    """Mesmo OOF aninhado do grafo único -- a trava do §6 vale igual aqui."""
    oof = None
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_train, inner_val in splitter.split(np.zeros(len(train_y), dtype=np.uint8), train_y):
        values = fused_query_features(
            [x[inner_train] for x in train_blocks], train_y[inner_train],
            [x[inner_val] for x in train_blocks], n_classes, k, alpha, max_iter)
        if oof is None:
            oof = np.zeros((len(train_y), values.shape[1]), dtype=np.float32)
        oof[inner_val] = values
    assert oof is not None
    target = fused_query_features(train_blocks, train_y, query_blocks,
                                  n_classes, k, alpha, max_iter)
    return oof, target


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def neighbourhood(train_x, query_x, k):
    return NearestNeighbors(n_neighbors=k).fit(train_x).kneighbors(query_x, return_distance=False)


def oracle_and_purity(indices, train_y, query_y, n_classes):
    """Oráculo = a classe verdadeira aparece entre os vizinhos. Pureza = fração
    dos vizinhos com a classe certa. Com poucas classes o oráculo satura e só a
    pureza informa -- por isso as duas saem juntas, sempre contra o acaso."""
    labels = train_y[indices]
    hit = (labels == query_y[:, None])
    return float(hit.any(axis=1).mean() * 100), float(hit.mean() * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--part-a-folds", type=int, default=1)
    parser.add_argument("--mlp", choices=["wide", "narrow"], default="narrow",
                        help="`narrow` é o MLP de produção adotado no §7.8 "
                             "(enc 128, cabeça 512-256-128); `wide` é o da "
                             "tabela de sensibilidade entre corpora")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results")
    args = parser.parse_args()

    names = VISUAL + ["Gabor"] if "Gabor" not in VISUAL else list(VISUAL)
    blocks, raw_labels, _ = load_corpus(args.dataset, names)
    if args.classes_file and args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
        if len(np.unique(raw_labels[mask])) != len(wanted):
            raise ValueError("casamento de classes do benchmark incompleto")
        blocks = {n: b[mask] for n, b in blocks.items()}
        raw_labels = raw_labels[mask]
    blocks = {n: clean(b) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    order = [n for n in names if n in blocks]
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, "
        f"{len(order)} modalidades ({sum(b.shape[1] for b in blocks.values())}d)")

    splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))

    # ------------------------------------------------------------------ parte A
    rows_a = []
    for fold, (train_index, test_index) in enumerate(splits[:args.part_a_folds], start=1):
        train_y, test_y = labels[train_index], labels[test_index]
        scaled = {}
        for name in order:
            scaler = StandardScaler().fit(blocks[name][train_index])
            scaled[name] = (scaler.transform(blocks[name][train_index]),
                            scaler.transform(blocks[name][test_index]))
        concat_train = np.hstack([scaled[n][0] for n in order])
        concat_test = np.hstack([scaled[n][1] for n in order])

        union = []
        for name in order:
            union.append(neighbourhood(scaled[name][0], scaled[name][1], args.k))
        union_indices = np.hstack(union)

        budget = args.k * len(order)
        for label, indices in (("concat k", neighbourhood(concat_train, concat_test, args.k)),
                               ("concat 8k", neighbourhood(concat_train, concat_test, budget)),
                               ("union of 8", union_indices)):
            oracle, purity = oracle_and_purity(indices, train_y, test_y, n_classes)
            # O acaso do oráculo depende do número de vizinhos, que é justamente
            # o que difere entre as três linhas -- sem ele a comparação mente.
            prior = np.bincount(train_y, minlength=n_classes) / len(train_y)
            chance_oracle = float((1 - (1 - prior[test_y]) ** indices.shape[1]).mean() * 100)
            rows_a.append({"fold": fold, "condition": label,
                           "n_neighbours": indices.shape[1], "oracle": oracle,
                           "chance_oracle": chance_oracle, "purity": purity,
                           "chance_purity": float(prior[test_y].mean() * 100)})
            log(f"  fold {fold}  {label:<11} k={indices.shape[1]:>3}  "
                f"oráculo {oracle:6.2f} (acaso {chance_oracle:6.2f})  "
                f"pureza {purity:6.2f}")
    frame_a = pd.DataFrame(rows_a)
    path_a = args.output_dir / f"multiedge_part_a_{args.dataset}.csv"
    frame_a.to_csv(path_a, index=False)
    log(f"parte A salva em {path_a}")

    # ------------------------------------------------------------------ parte B
    # Import tardio: torch. As duas larguras existem porque o Δ do grafo pode
    # depender da capacidade da cabeça -- o §7.8 já registra que a escolha corta
    # a favor de um lado, e aqui ela é a variável sob teste.
    from sklearn.model_selection import train_test_split
    if args.mlp == "wide":
        from evaluate_fusion_delta_across_corpora import fit_mlp

        def run(chosen, fold, train_index, test_index):
            fit_idx, val_idx = train_test_split(
                train_index, test_size=0.1, stratify=labels[train_index],
                random_state=SEED + fold)
            return fit_mlp(chosen, labels, fit_idx, val_idx, test_index,
                           n_classes, seed=SEED + fold)
    else:
        from evaluate_full_factorial_ablation import evaluate_mlp

        def run(chosen, fold, train_index, test_index):
            metrics = evaluate_mlp(chosen, labels, train_index, test_index,
                                   fold, epochs=150, patience=15)
            return {k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}

    def full_length(train_index, test_index, oof, target):
        """As features do grafo são por fold; o `fit_mlp` quer arrays do tamanho
        da coorte e recorta por índice. Linhas de treino recebem o OOF aninhado,
        linhas de teste o valor ajustado no treino inteiro."""
        out = np.zeros((len(labels), oof.shape[1]), dtype=np.float32)
        out[train_index], out[test_index] = oof, target
        return out

    base_blocks = [blocks[n] for n in order]
    rows_b = []
    suffix = "" if args.mlp == "narrow" else f"_{args.mlp}"
    output_b = args.output_dir / f"multiedge_{args.dataset}{suffix}.csv"
    done = set()
    if output_b.exists():
        rows_b = pd.read_csv(output_b).to_dict("records")
        done = {(int(r["fold"]), str(r["condition"])) for r in rows_b}

    for fold, (train_index, test_index) in enumerate(splits[:args.folds], start=1):
        train_y = labels[train_index]

        scaled = {}
        for name in order:
            scaler = StandardScaler().fit(blocks[name][train_index])
            scaled[name] = (scaler.transform(blocks[name][train_index]),
                            scaler.transform(blocks[name][test_index]))

        started = time.perf_counter()
        oof, target = rich_similarity_features(
            np.hstack([scaled[n][0] for n in order]), train_y,
            np.hstack([scaled[n][1] for n in order]), n_classes,
            args.k, args.alpha, args.max_iter, SEED + fold)
        naive_seconds = time.perf_counter() - started
        naive = full_length(train_index, test_index,
                            prior_correct_rich(oof, train_y, n_classes),
                            prior_correct_rich(target, train_y, n_classes))

        started = time.perf_counter()
        per_modality = []
        for name in order:
            oof, target = rich_similarity_features(
                scaled[name][0], train_y, scaled[name][1], n_classes,
                args.k, args.alpha, args.max_iter, SEED + fold)
            per_modality.append(full_length(
                train_index, test_index,
                prior_correct_rich(oof, train_y, n_classes),
                prior_correct_rich(target, train_y, n_classes)))
        multi_seconds = time.perf_counter() - started
        log(f"  fold {fold}  grafo: ingênuo {naive_seconds:.0f}s, "
            f"multiaresta {multi_seconds:.0f}s ({multi_seconds / naive_seconds:.1f}x)")

        started = time.perf_counter()
        oof, target = fused_similarity_features(
            [scaled[n][0] for n in order], train_y, [scaled[n][1] for n in order],
            n_classes, args.k, args.alpha, args.max_iter, SEED + fold)
        fused_seconds = time.perf_counter() - started
        fused = full_length(train_index, test_index,
                            prior_correct_rich(oof, train_y, n_classes),
                            prior_correct_rich(target, train_y, n_classes))
        log(f"  fold {fold}  grafo fundido (uma difusão): {fused_seconds:.0f}s")

        lean = slice(0, 2 * n_classes)
        conditions = {
            "hand-8": base_blocks,
            "naive": base_blocks + [naive],
            "multi-lean": base_blocks + [x[:, lean] for x in per_modality],
            "multi-full": base_blocks + per_modality,
            "multi-diffusion": base_blocks + [fused],
        }
        graph_cost = {"naive": naive_seconds, "multi-lean": multi_seconds,
                      "multi-full": multi_seconds, "multi-diffusion": fused_seconds}
        for condition, chosen in conditions.items():
            if (fold, condition) in done:
                continue
            started = time.perf_counter()
            metrics = run(chosen, fold, train_index, test_index)
            rows_b.append({"fold": fold, "condition": condition,
                           "dimensions": sum(b.shape[1] for b in chosen),
                           "graph_seconds": graph_cost.get(condition, 0.0),
                           "mlp_seconds": time.perf_counter() - started,
                           **{k: (100 * v if k in ("bacc", "accuracy", "macro_f1") else v)
                              for k, v in metrics.items()}})
            log(f"  fold {fold}  {condition:<11} {rows_b[-1]['dimensions']:>5}d  "
                f"BAcc {rows_b[-1]['bacc']:6.2f}")
            pd.DataFrame(rows_b).to_csv(output_b, index=False)

    frame_b = pd.DataFrame(rows_b)
    log("\n" + frame_b.groupby("condition")[["dimensions", "bacc", "accuracy"]]
        .mean().round(2).to_string())
    from scipy import stats
    base = frame_b[frame_b.condition.eq("naive")].set_index("fold")["bacc"]
    log("\ndelta contra o grafo ingênuo (pareado por fold):")
    for condition in ("hand-8", "multi-lean", "multi-full", "multi-diffusion"):
        other = frame_b[frame_b.condition.eq(condition)].set_index("fold")["bacc"]
        shared = base.index.intersection(other.index)
        if len(shared) > 1:
            delta = other[shared] - base[shared]
            _, p = stats.ttest_rel(other[shared], base[shared])
            log(f"  {condition:<11} {delta.mean():+6.2f}pp  p={p:.4f}  "
                f"{int((delta > 0).sum())}/{len(delta)} folds")


if __name__ == "__main__":
    main()
