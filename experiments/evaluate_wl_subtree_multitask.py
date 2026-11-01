"""Weisfeiler-Lehman sobre os RAGs de regiao: a topologia acrescenta algo sobre
o histograma de cor?

Proposta avaliada: rotular os nos do RAG por um codebook de cor e usar o mapa
explicito do kernel WL (um "bag of subtrees") como view nova. O substrato ja
esta em disco (§7ter preservou os RAGs exatamente para isto) e o `x` dos nos ja
traz Lab, entao o codebook nao custa releitura de imagem.

**O TESTE QUE DECIDE E d=0 CONTRA d=2.** `d=0` e so o histograma de rotulos de
cor -- sem vizinhanca nenhuma -- e o `Descriptors` ja carrega 90 dims de
histograma Lab (§10.2). Se d=2 nao passar d=0, o que foi medido e um codebook de
cor novo com outro nome, nao topologia.

**Segundo controle, mais forte: ARESTAS EMBARALHADAS.** Mesmos rotulos, mesmo
numero de arestas, adjacencia aleatoria. Separa "a estrutura real das regioes"
de "agregar vizinho qualquer suaviza cor de um jeito util". O §5.6 mede que um
bloco com informacao destruida ainda marca +0,22pp so de arquitetura, entao o
placebo nao e opcional.

**Reducao: TF-IDF -> LinearSVC -> escore por classe**, a mesma do canal de texto
(`build_text_scores.py`). Um bag of subtrees E um bag of words, e esta e a
reducao que o projeto tem medida para sinal esparso de vocabulario grande
(§5.9b: a MESMA legenda vale -0,09pp como embedding de 384d e +2,31pp como
TF-IDF de 125d). Concatenar milhares de dims esparsas no `BranchMLP`, que manda
todo bloco para 128, e a patologia do §5.17c (-1,75pp no pipeline contra -0,76
sozinho).

**O codebook e o vocabulario sao ajustados SO NO TREINO de cada fold.** Ajustar
na coorte inteira e transdutivo (§10.6) e, com o SVM em cima, e o mecanismo que
derrubou o SNF (§6).

Uso:
    python experiments/evaluate_wl_subtree_multitask.py --k 64 --folds 5
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import HAND8                              # noqa: E402
from protocol import SEED, prediction_metrics          # noqa: E402
from registry import load_blocks                       # noqa: E402

P1, P2, P3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0x165667B19E3779F9)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_cache(cache_dir: Path, stems: list[str]) -> dict:
    """Le os shards e devolve arrays concatenados na ORDEM de cohort.stems."""
    shards = sorted(cache_dir.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"nenhum shard em {cache_dir}")
    labs, edges, n_nodes, n_edges, order = [], [], [], [], []
    for path in shards:
        blob = np.load(path, allow_pickle=False)
        node_off, edge_off = blob["node_offsets"], blob["edge_offsets"]
        lab_all, edge_all = blob["lab"], blob["edges"]
        for i, stem in enumerate(blob["stems"]):
            labs.append(lab_all[node_off[i]:node_off[i + 1]])
            edges.append(edge_all[:, edge_off[i]:edge_off[i + 1]])
            n_nodes.append(node_off[i + 1] - node_off[i])
            n_edges.append(edge_off[i + 1] - edge_off[i])
            order.append(str(stem))
    if order != stems:
        index = {s: i for i, s in enumerate(order)}
        missing = [s for s in stems if s not in index]
        if missing:
            raise SystemExit(f"{len(missing)} stems da coorte sem grafo (ex.: {missing[:3]})")
        perm = [index[s] for s in stems]
        labs = [labs[i] for i in perm]
        edges = [edges[i] for i in perm]
        n_nodes = [n_nodes[i] for i in perm]
        n_edges = [n_edges[i] for i in perm]
        log("  cache reindexado para a ordem de cohort.stems")
    node_offsets = np.concatenate([[0], np.cumsum(n_nodes)]).astype(np.int64)
    return {
        "lab": np.concatenate(labs, axis=0).astype(np.float32),
        "edges": edges,                       # por imagem, indices LOCAIS
        "node_offsets": node_offsets,
        "n_nodes": np.asarray(n_nodes, dtype=np.int64),
    }


def global_edges(edges_per_image: list[np.ndarray], node_offsets: np.ndarray,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Concatena as arestas num grafo gigante desconexo, com indices globais.

    `rng` != None => ARESTAS EMBARALHADAS: mesma contagem por imagem, pontas
    sorteadas dentro da propria imagem (placebo de topologia).
    """
    src, dst = [], []
    for i, e in enumerate(edges_per_image):
        base, n = node_offsets[i], node_offsets[i + 1] - node_offsets[i]
        if e.shape[1] == 0 or n < 2:
            continue
        if rng is None:
            s, d = e[0].astype(np.int64), e[1].astype(np.int64)
        else:
            s = rng.integers(0, n, size=e.shape[1])
            d = rng.integers(0, n, size=e.shape[1])
        src.append(s + base)
        dst.append(d + base)
    return np.concatenate(src), np.concatenate(dst)


def wl_labels(colour_labels: np.ndarray, src: np.ndarray, dst: np.ndarray,
              depth: int, seed: int) -> list[np.ndarray]:
    """Rotulos WL por iteracao. Hash comutativo exato: o aleatorio de 64 bits e
    partido em duas metades de 32, somadas em float64 (soma de ~2k valores
    < 2^32 da < 2^53, entao e exata) e recombinadas."""
    rng = np.random.default_rng(seed)
    n = colour_labels.shape[0]
    out = [colour_labels]
    labels = colour_labels
    for _ in range(depth):
        n_labels = int(labels.max()) + 1
        table = rng.integers(0, 2 ** 64, size=n_labels, dtype=np.uint64)
        lo = (table & np.uint64(0xFFFFFFFF)).astype(np.float64)
        hi = (table >> np.uint64(32)).astype(np.float64)
        w = labels[dst]
        lo_sum = np.bincount(src, weights=lo[w], minlength=n)
        hi_sum = np.bincount(src, weights=hi[w], minlength=n)
        key = (labels.astype(np.uint64) * P1
               ^ lo_sum.astype(np.uint64) * P2
               ^ hi_sum.astype(np.uint64) * P3)
        _, labels = np.unique(key, return_inverse=True)
        labels = labels.astype(np.int64)
        out.append(labels)
    return out


def counts_matrix(label_sets: list[np.ndarray], node_offsets: np.ndarray,
                  n_images: int, min_df: int) -> sp.csr_matrix:
    """Bag of subtrees: uma coluna por (profundidade, rotulo), podada por
    document-frequency. A poda e a diferenca entre ~10^6 colunas e ~10^4."""
    image_of_node = np.repeat(np.arange(n_images), np.diff(node_offsets))
    blocks, offset = [], 0
    for labels in label_sets:
        n_labels = int(labels.max()) + 1
        key = image_of_node.astype(np.int64) * n_labels + labels
        uniq, counts = np.unique(key, return_counts=True)
        rows, cols = uniq // n_labels, uniq % n_labels
        mat = sp.coo_matrix((counts, (rows, cols)),
                            shape=(n_images, n_labels)).tocsr()
        df = np.asarray((mat > 0).sum(axis=0)).ravel()
        keep = np.flatnonzero(df >= min_df)
        blocks.append(mat[:, keep])
        offset += keep.size
    return sp.hstack(blocks).tocsr()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/prism_data/multitask_slic_rag"))
    parser.add_argument("--k", type=int, default=64, help="tamanho do codebook de cor")
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "multitask_wl_subtree.csv")
    args = parser.parse_args()

    log(f"coorte de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset, list(HAND8))
    labels_y = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    stems = [str(s) for s in cohort.stems]
    n_images = len(stems)
    log(f"  {n_images} imagens, {len(np.unique(labels_y))} classes")

    log(f"lendo cache de RAG em {args.cache_dir}")
    started = time.time()
    cache = load_cache(args.cache_dir, stems)
    log(f"  {cache['lab'].shape[0]:,} nos no total "
        f"({cache['n_nodes'].mean():.0f} por imagem), {time.time()-started:.0f}s")

    src_real, dst_real = global_edges(cache["edges"], cache["node_offsets"])
    rng_edges = np.random.default_rng(SEED)
    src_fake, dst_fake = global_edges(cache["edges"], cache["node_offsets"], rng=rng_edges)
    log(f"  {src_real.size:,} arestas (ambos os sentidos)")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(n_images), labels_y))[:args.folds]

    ARMS = [("d0 (so cor)", 0, False), ("d1", 1, False), ("d2", 2, False),
            ("d2 arestas embaralhadas", 2, True)]

    rows: list[dict] = []
    for fold, (train, test) in enumerate(splits, start=1):
        node_is_train = np.repeat(np.isin(np.arange(n_images), train),
                                  np.diff(cache["node_offsets"]))
        sample = cache["lab"][node_is_train]
        if sample.shape[0] > 400_000:
            idx = np.random.default_rng(SEED + fold).choice(sample.shape[0], 400_000,
                                                            replace=False)
            sample = sample[idx]
        km = MiniBatchKMeans(n_clusters=args.k, random_state=SEED + fold,
                             n_init=3, batch_size=4096).fit(sample)
        colour_labels = km.predict(cache["lab"]).astype(np.int64)
        log(f"  fold {fold}: codebook K={args.k} ajustado em {sample.shape[0]:,} nos de treino")

        for name, depth, shuffled in ARMS:
            t0 = time.perf_counter()
            s, d = (src_fake, dst_fake) if shuffled else (src_real, dst_real)
            label_sets = wl_labels(colour_labels, s, d, depth, SEED + fold)
            X = counts_matrix(label_sets, cache["node_offsets"], n_images, args.min_df)
            tfidf = TfidfTransformer(sublinear_tf=True)
            Xtr = tfidf.fit_transform(X[train])
            Xte = tfidf.transform(X[test])
            clf = LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=3000)
            clf.fit(Xtr, labels_y[train])
            metrics = prediction_metrics(labels_y[test], clf.predict(Xte))
            rows.append({"fold": fold, "arm": name, "depth": depth,
                         "shuffled_edges": shuffled, "vocab": X.shape[1],
                         "seconds": time.perf_counter() - t0,
                         "bacc": 100 * metrics["bacc"],
                         "accuracy": 100 * metrics["accuracy"],
                         "macro_f1": 100 * metrics["macro_f1"]})
            log(f"    {name:26s} vocab={X.shape[1]:6d}  BAcc {rows[-1]['bacc']:5.2f}  "
                f"({rows[-1]['seconds']:.0f}s)")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows)
    summary = table.groupby("arm", sort=False).agg(
        vocab=("vocab", "mean"), bacc=("bacc", "mean"), sd=("bacc", "std"),
        acc=("accuracy", "mean"), f1=("macro_f1", "mean"), s=("seconds", "mean"))
    log("\n" + summary.round(2).to_string())

    base = table[table.arm.eq("d0 (so cor)")].set_index("fold")["bacc"]
    log("\ndelta contra d0 (so cor), pareado por fold:")
    for name in ["d1", "d2", "d2 arestas embaralhadas"]:
        other = table[table.arm.eq(name)].set_index("fold")["bacc"]
        delta = other - base
        log(f"  {name:26s} {delta.mean():+6.2f} +- {delta.std():4.2f}  "
            f"{int((delta > 0).sum())}/{len(delta)}")

    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
