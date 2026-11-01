"""Grafo de similaridade fundido por Similarity Network Fusion (Wang et al.
2014, Nature Methods), um grafo k-NN por modalidade (das 8 nativas) na
propria escala de cada uma, fundidos por difusao cruzada iterativa -- em
vez do k-NN unico sobre o vetor concatenado que o projeto usava ate agora
(achado 12 e itens 32-33 do CLAUDE.md), sensivel a desbalanceamento de
dimensao entre blocos (Multiscale ~1000d vs Hilbert 57d).

Protocolo por fold (leak-free, indutivo):
  1. Para cada bloco, kNN local-scale-adaptativo (kernel gaussiano, escala
     por vizinhanca) SO no treino, via GPU. Duas matrizes esparsas por
     bloco: S (top-k=30, "confianca local") e P (top-K=100, "opiniao
     completa"), ambas normalizadas por linha.
  2. Difusao cruzada: a cada iteracao, P do bloco m e atualizado com a
     media das opinioes dos OUTROS blocos, filtrada pela vizinhanca local
     de m (S_m @ media(P_outros) @ S_m^T). ~15 iteracoes.
  3. Fusao final: media das P convergidas dos 8 blocos.
  4. Espalhamento de rotulo (label spreading, alpha=0.4, iterativo) sobre o
     grafo fundido do treino -> posterior por imagem de treino.
  5. Pontos de teste (indutivo, nunca vistos no fit): para cada bloco, os
     k=30 vizinhos mais proximos no TREINO (na escala daquele bloco);
     media do posterior (ja convergido) desses vizinhos; media das 8
     estimativas por bloco. Log-lift com o prior do treino (achado 12: todo
     bloco tipo-posterior nasce com log-lift).
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
import torch
from scipy import sparse
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.registry import load_blocks  # noqa: E402

SEED = 42
N_FOLDS = 5
CORE_BLOCKS = [
    "Descriptors", "SLIC", "Felzenszwalb", "Line graph",
    "L(G_felz)", "Hilbert", "Fourier", "Multiscale",
]
K_LOCAL = 30
K_FULL = 100
SNF_ITERS = 15
SPREAD_ALPHA = 0.4
SPREAD_ITERS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def gpu_knn(query: np.ndarray, reference: np.ndarray, k: int,
            exclude_self: bool, batch: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Top-k vizinhos de `query` em `reference` (distancia euclidiana), via GPU
    em lotes. `exclude_self` descarta o vizinho de distancia 0 na propria
    posicao (usado quando query is reference)."""
    ref = torch.as_tensor(reference, dtype=torch.float32, device=DEVICE)
    ref_sq = (ref ** 2).sum(dim=1)
    n_query = query.shape[0]
    k_query = k + 1 if exclude_self else k
    all_idx = np.empty((n_query, k), dtype=np.int64)
    all_dist = np.empty((n_query, k), dtype=np.float32)
    for start in range(0, n_query, batch):
        end = min(start + batch, n_query)
        chunk = torch.as_tensor(query[start:end], dtype=torch.float32, device=DEVICE)
        chunk_sq = (chunk ** 2).sum(dim=1, keepdim=True)
        dist_sq = chunk_sq + ref_sq.unsqueeze(0) - 2.0 * (chunk @ ref.T)
        dist_sq.clamp_(min=0.0)
        values, indices = torch.topk(dist_sq, k_query, dim=1, largest=False)
        if exclude_self:
            values = values[:, 1:]
            indices = indices[:, 1:]
        all_idx[start:end] = indices.cpu().numpy()
        all_dist[start:end] = torch.sqrt(values).cpu().numpy()
    return all_idx, all_dist


def local_scale_similarity(indices: np.ndarray, dist: np.ndarray, k_scale: int) -> np.ndarray:
    """Kernel gaussiano com escala local (media da distancia aos k_scale
    vizinhos mais proximos de cada ponto), estilo SNF."""
    eps = dist[:, :k_scale].mean(axis=1, keepdims=True)
    eps = np.maximum(eps, 1e-8)
    # eps_ij = (eps_i + eps_j) / 2 -- eps_j exige indexar eps pelo indice do vizinho
    eps_j = eps[indices.ravel(), 0].reshape(indices.shape)
    eps_ij = 0.5 * (eps + eps_j)
    return np.exp(-(dist ** 2) / (2.0 * eps_ij ** 2 + 1e-12))


def build_sparse(n: int, indices: np.ndarray, weights: np.ndarray, top_k: int) -> sparse.csr_matrix:
    rows = np.repeat(np.arange(n), top_k)
    cols = indices[:, :top_k].ravel()
    vals = weights[:, :top_k].ravel()
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    return sparse.diags(1.0 / row_sums) @ matrix


def truncate_topk_per_row(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Mantem so os k maiores valores por linha. Necessario: produtos de
    matriz esparsa "enchem" o padrao de esparsidade a cada iteracao (visto
    na pratica: 3 iteracoes em N=3.000 bastaram pra virar 100% densa) --
    sem truncar, a difusao cruzada explode memoria em N~63.000/fold."""
    matrix = matrix.tocsr()
    data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
    new_data_parts, new_indices_parts, new_indptr = [], [], [0]
    for row in range(matrix.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        row_data = data[start:end]
        row_idx = indices[start:end]
        if len(row_data) > k:
            top = np.argpartition(-row_data, k)[:k]
            row_data = row_data[top]
            row_idx = row_idx[top]
        new_data_parts.append(row_data)
        new_indices_parts.append(row_idx)
        new_indptr.append(new_indptr[-1] + len(row_data))
    new_data = np.concatenate(new_data_parts) if new_data_parts else np.array([], dtype=matrix.dtype)
    new_indices = np.concatenate(new_indices_parts) if new_indices_parts else np.array([], dtype=np.int32)
    return sparse.csr_matrix((new_data, new_indices, np.asarray(new_indptr)), shape=matrix.shape)


def snf_fuse(blocks_train: dict[str, np.ndarray]) -> tuple[dict[str, sparse.csr_matrix], sparse.csr_matrix]:
    """Retorna (S por bloco, P fundido) para o treino."""
    s_mats, p_mats = {}, {}
    n = next(iter(blocks_train.values())).shape[0]
    for name, values in blocks_train.items():
        idx, dist = gpu_knn(values, values, K_FULL, exclude_self=True)
        weights = local_scale_similarity(idx, dist, K_LOCAL)
        s_mats[name] = build_sparse(n, idx, weights, K_LOCAL)
        p_mats[name] = build_sparse(n, idx, weights, K_FULL)
        log(f"  {name}: grafo local construido ({n:,} x {n:,}, k_local={K_LOCAL})")

    names = list(blocks_train)
    for iteration in range(SNF_ITERS):
        updated = {}
        for name in names:
            others = [p_mats[other] for other in names if other != name]
            mean_others = others[0]
            for other in others[1:]:
                mean_others = mean_others + other
            mean_others = mean_others / len(others)
            new_p = s_mats[name] @ mean_others @ s_mats[name].T
            new_p = truncate_topk_per_row(new_p, K_FULL)
            row_sums = np.asarray(new_p.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            updated[name] = sparse.diags(1.0 / row_sums) @ new_p
        p_mats = updated
        log(f"  difusao cruzada: iteracao {iteration + 1}/{SNF_ITERS}")

    fused = p_mats[names[0]]
    for name in names[1:]:
        fused = fused + p_mats[name]
    fused = fused / len(names)
    return s_mats, fused


def spread_labels(fused: sparse.csr_matrix, y_onehot: np.ndarray) -> np.ndarray:
    f = y_onehot.copy()
    y0 = y_onehot
    for _ in range(SPREAD_ITERS):
        f = SPREAD_ALPHA * (fused @ f) + (1 - SPREAD_ALPHA) * y0
    row_sums = f.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return f / row_sums


def log_lift(posterior: np.ndarray, prior: np.ndarray, smoothing: float = 0.01) -> np.ndarray:
    """log p(c|x) - log p(c), com suavizacao aditiva (p + smoothing*prior) em
    vez de clip duro. Um clip duro em 1e-9 faz uma fracao grande dos valores
    (~3,7% no teste com 10k/27 classes) colapsar exatamente no mesmo piso
    numerico -14/-15 -- imagens sem nenhuma relacao viram "identicas" nessa
    coluna so por saturacao, o que corrompe a covariancia do LDA. A
    suavizacao aditiva limita o log-lift minimo a log(smoothing) de forma
    continua, sem criar platos artificiais."""
    smoothed = posterior + smoothing * prior
    return np.log(smoothed) - np.log(prior)


def inductive_test_posterior(blocks_train: dict[str, np.ndarray], blocks_test: dict[str, np.ndarray],
                              train_posterior: np.ndarray) -> np.ndarray:
    estimates = []
    for name in blocks_train:
        idx, _ = gpu_knn(blocks_test[name], blocks_train[name], K_LOCAL, exclude_self=False)
        estimates.append(train_posterior[idx].mean(axis=1))
    return np.mean(estimates, axis=0)


def make_lda(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, n_features),
            priors=np.ones(n_classes) / n_classes,
        )),
    ])


def make_mlp() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-4, max_iter=200,
            early_stopping=True, n_iter_no_change=10, random_state=SEED,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/snf_similarity_graph_results.csv")
    args = parser.parse_args()

    cohort = load_blocks("wikiart", CORE_BLOCKS, allow_external=False)
    y = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(y))
    stems = cohort.stems
    log(f"{len(stems):,} imagens, {n_classes} classes, device={DEVICE}")

    multiscale_desc_only = cohort.select("Multiscale", lambda c: "_Descriptors_" in f"_{c}_")
    base_blocks = [cohort.blocks[name] for name in CORE_BLOCKS if name != "Multiscale"]
    base = np.hstack(base_blocks + [multiscale_desc_only]).astype(np.float32)
    log(f"base de producao: {base.shape[1]}d")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(base, y))
    rows = []
    for fold_index, (train_idx, test_idx) in enumerate(folds, 1):
        log(f"=== fold {fold_index}/{N_FOLDS} ===")
        blocks_train = {name: cohort.blocks[name][train_idx].astype(np.float32) for name in CORE_BLOCKS}
        blocks_test = {name: cohort.blocks[name][test_idx].astype(np.float32) for name in CORE_BLOCKS}
        y_train, y_test = y[train_idx], y[test_idx]

        _s_mats, fused_train = snf_fuse(blocks_train)

        onehot = np.zeros((len(train_idx), n_classes), dtype=np.float64)
        onehot[np.arange(len(train_idx)), y_train] = 1.0
        train_posterior = spread_labels(fused_train, onehot)

        prior = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        prior /= prior.sum()

        test_posterior = inductive_test_posterior(blocks_train, blocks_test, train_posterior)
        train_feature = log_lift(train_posterior, prior).astype(np.float32)
        test_feature = log_lift(test_posterior, prior).astype(np.float32)

        base_train, base_test = base[train_idx], base[test_idx]
        combos = {
            "base": (base_train, base_test),
            "base+snf_graph": (
                np.hstack([base_train, train_feature]),
                np.hstack([base_test, test_feature]),
            ),
        }
        for combo_name, (xtr, xte) in combos.items():
            for clf_name, make in (("lda", make_lda), ("mlp", make_mlp)):
                model = make(n_classes, xtr.shape[1]) if clf_name == "lda" else make()
                model.fit(xtr, y_train)
                pred = model.predict(xte)
                bacc = balanced_accuracy_score(y_test, pred)
                rows.append({"fold": fold_index, "combo": combo_name, "classifier": clf_name,
                             "balanced_accuracy": bacc})
                log(f"  {combo_name:15s} {clf_name:4s}: BAcc={bacc*100:.2f}%")

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print()
    print("=== resumo ===")
    for clf_name in ("lda", "mlp"):
        base_scores = result[(result.classifier == clf_name) & (result.combo == "base")].sort_values("fold")["balanced_accuracy"].to_numpy()
        graph_scores = result[(result.classifier == clf_name) & (result.combo == "base+snf_graph")].sort_values("fold")["balanced_accuracy"].to_numpy()
        delta = graph_scores - base_scores
        stat, pvalue = ttest_rel(graph_scores, base_scores)
        print(f"{clf_name}: base={base_scores.mean()*100:.2f}+/-{base_scores.std(ddof=0)*100:.2f}  "
              f"+SNF={graph_scores.mean()*100:.2f}+/-{graph_scores.std(ddof=0)*100:.2f}  "
              f"delta={delta.mean()*100:+.3f}pp +/- {delta.std(ddof=0)*100:.3f}pp  "
              f"p={pvalue:.4f}  {int((delta>0).sum())}/{N_FOLDS} folds")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
