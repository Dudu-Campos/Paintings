"""Shapley EXATO das dez views (8 visuais + grafo + texto), com sonda LDA.

O §5.10 tirou Shapley e LOO das 511 máscaras do fatorial de 2^9, mas aquele
fatorial tem **Multiscale** (aposentado no §3) e **não tem nem o grafo nem o
texto**. Aqui são 2^10 = 1.024 máscaras sobre as dez views correntes.

**Por que Shapley e não LOO.** O LOO mede o marginal na coalizão CHEIA, e o
`evaluate_loo_all_views.py` mostrou que isso é instável justamente onde as views
se substituem: o SLIC deu +2,31 num fold e −2,19 no outro. Shapley promedia o
marginal sobre TODAS as coalizões, então um bloco redundante no conjunto cheio
ainda recebe crédito das coalizões pequenas. E `substituição = Shapley − LOO`
(§5.10) sai de graça, que é a quantidade que responde "quanto deste bloco os
outros cobrem".

**Três decisões de protocolo que mudam o que o número significa:**

1. **Sonda LDA, não MLP** -- Shapley exato exige 1.024 ajustes por fold, e o LDA
   ajusta em segundos. O §8 registra SEIS casos de "LDA extrai, MLP destrói" em
   vetores multi-bloco. **Ler como ordenação de importância, não como previsão do
   que acontece no MLP.**
2. **O grafo entra PRÉ-COMPUTADO a partir do espaço hand-8 completo**, igual em
   toda máscara. Reconstruí-lo por máscara custaria 133 s x 5.120 -- inviável.
   Então o Shapley dele mede o valor do bloco de features, NÃO "o grafo que
   aquele subconjunto produziria". É limitação real e tem de ser declarada.
3. **Texto com OOF aninhado**, uma vez por fold (§5.18): sem isso os escores das
   linhas de treino vêm de modelos que viram o fold de teste.

`v(vazio)` = 1/C (acaso), porque o LDA não ajusta com zero colunas.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import itertools
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from build_text_scores import make_stop_words  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features)
from evaluate_text_marginal_multitask import CAPTIONS, load_captions, nested_text_block  # noqa: E402
from evaluate_backward_elimination import EXTERNAL  # noqa: E402
from registry import load_blocks, log  # noqa: E402
from view_names import display  # noqa: E402

SEED, N_FOLDS = 42, 5
GRAPH_PARAMS = dict(n_neighbors=30, alpha=0.4, max_iter=100)
# DOZE views: as 8 de produção, as duas DORMENTES que existem extraídas para o
# multitask (§6), o grafo e o texto. As dormentes entram porque esta tabela é da
# ABLAÇÃO -- o argumento para removê-las tem de sair daqui, e uma tabela só com
# as que sobreviveram assume a conclusão. As outras cinco dormentes do §6 não
# foram extraídas para este corpus.
DORMANT = list(EXTERNAL)
VIEWS = list(HAND8) + DORMANT + ["Similarity graph", "Text"]
_TODAS = list(VIEWS)                      # o default publicado

_STATE: dict = {}


def _init(matrix, y, spans, train, test, n_classes):
    _STATE.update(matrix=matrix, y=y, spans=spans, train=train, test=test,
                  n_classes=n_classes)


def _evaluate(mask: int) -> tuple[int, float, int]:
    if mask == 0:
        return mask, 100.0 / _STATE["n_classes"], 0
    cols = np.concatenate([np.arange(*_STATE["spans"][i])
                           for i in range(len(VIEWS)) if mask >> i & 1])
    X = _STATE["matrix"][:, cols]
    tr, te = _STATE["train"], _STATE["test"]
    n = _STATE["n_classes"]
    model = LDA(priors=np.ones(n) / n).fit(X[tr], _STATE["y"][tr])
    score = balanced_accuracy_score(_STATE["y"][te], model.predict(X[te]))
    return mask, 100.0 * score, len(cols)


def build_fold(cohort, labels, labels_str, texts, classes, index_of, stop_words,
               train_idx, test_idx, fold, cache, extras):
    """Matriz do fold com as doze views, padronizadas no treino.

    O cache guarda SO grafo e texto -- as duas pecas caras (~9 min por fold). A
    matriz e montada a cada execucao, entao mudar a lista de views nao forca
    recompute-las. A primeira versao cacheava a matriz pronta, e acrescentar as
    dormentes teria obrigado a refazer tudo.
    """
    graph = text = None
    if cache.exists():
        saved = np.load(cache)
        graph = saved["graph"] if "graph" in saved else None
        text = saved["text"] if "text" in saved else None

    hand = [cohort.blocks[n] for n in HAND8 if n in VIEWS]
    quer_grafo = "Similarity graph" in VIEWS
    quer_texto = "Text" in VIEWS
    if (quer_grafo and graph is None) or (quer_texto and text is None):
        fit_idx, val_idx = train_test_split(train_idx, test_size=0.10,
                                            random_state=SEED,
                                            stratify=labels[train_idx])
        if quer_grafo and graph is None:
            # o espaco do kNN e SEMPRE o hand-8 completo, nao o subconjunto da
            # mascara (ressalva 3 do §5.9c): reconstruir o grafo por mascara
            # custaria 133 s x 2^n. O Shapley do grafo e portanto o do BLOCO
            # calculado do banco cheio, e isso tem de ir na legenda.
            source = np.hstack([cohort.blocks[n] for n in HAND8])
            space = StandardScaler().fit(source[fit_idx]).transform(source).astype(np.float32)
            log(f"  fold {fold}: grafo (kNN sobre {space.shape[1]}d)")
            oof, held = rich_similarity_features(
                space[fit_idx], labels[fit_idx],
                np.vstack([space[val_idx], space[test_idx]]),
                len(classes), seed=SEED, **GRAPH_PARAMS)
            oof = prior_correct_rich(oof, labels[fit_idx], len(classes))
            held = prior_correct_rich(held, labels[fit_idx], len(classes))
            graph = np.zeros((len(labels), oof.shape[1]), dtype=np.float32)
            graph[fit_idx] = oof
            graph[val_idx] = held[:len(val_idx)]
            graph[test_idx] = held[len(val_idx):]
        if quer_texto and text is None:
            log(f"  fold {fold}: texto (OOF aninhado)")
            text = nested_text_block(texts, labels_str, classes, index_of,
                                     stop_words, train_idx, test_idx, fold)
        cache.parent.mkdir(parents=True, exist_ok=True)
        guardar = {}
        if graph is not None:
            guardar["graph"] = graph
        if text is not None:
            guardar["text"] = text
        if guardar:
            np.savez(cache, **guardar)

    parts, spans, offset = [], [], 0
    blocos = hand + extras
    if quer_grafo:
        blocos.append(graph)
    if quer_texto:
        blocos.append(text)
    if len(blocos) != len(VIEWS):
        raise SystemExit(f"montei {len(blocos)} blocos para {len(VIEWS)} views -- abortando")
    for block in blocos:
        z = StandardScaler().fit(block[train_idx]).transform(block).astype(np.float32)
        parts.append(z)
        spans.append((offset, offset + z.shape[1]))
        offset += z.shape[1]
    return np.hstack(parts), np.array(spans)


def shapley(values: dict[int, float], n: int) -> np.ndarray:
    phi = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        for mask, v in values.items():
            if mask & bit:
                continue
            size = bin(mask).count("1")
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            phi[i] += weight * (values[mask | bit] - v)
    return phi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="multitask")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--views", nargs="+", default=None,
                    help="subconjunto de views a jogar. O default sao as 12 do "
                         "§5.9c e reproduz o publicado. `--views HAND8+grafo` e "
                         "o atalho para as 8 de producao mais o grafo: 2^9 = 512 "
                         "mascaras em vez de 4.096, e e o conjunto que responde "
                         "'o grafo pertence ao PRISM?' sem os dormentes e sem o "
                         "texto no meio.")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/shapley_twelve_views")
    ap.add_argument("--output", type=Path, default=ROOT / "results/multitask_shapley_twelve.csv")
    args = ap.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    global VIEWS
    if args.views == ["HAND8+grafo"]:
        VIEWS = list(HAND8) + ["Similarity graph"]
    elif args.views:
        fora = [v for v in args.views if v not in _TODAS]
        if fora:
            raise SystemExit(f"--views: {fora} nao esta entre {_TODAS}")
        VIEWS = [v for v in _TODAS if v in set(args.views)]   # ordem canonica

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels_str = np.asarray(cohort.labels, dtype=str)
    labels = LabelEncoder().fit_transform(labels_str)
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    stop_words = make_stop_words(classes)
    # legendas e dormentes so sao carregadas se a view estiver EM JOGO: o
    # arquivo de legendas e por corpus e os dormentes nao existem em todos,
    # entao carregar incondicionalmente aborta num corpus que nao joga Text
    texts = load_captions(CAPTIONS["estudante"], stems) if "Text" in VIEWS else None
    extras = []
    for name in [d for d in DORMANT if d in VIEWS]:
        values_path, stems_path = EXTERNAL[name]
        values = np.load(ROOT / values_path)
        order = pd.Index(np.load(ROOT / stems_path, allow_pickle=True)
                         .astype(str)).get_indexer(stems)
        if (order < 0).any():
            raise ValueError(f"{name}: {int((order < 0).sum())} stems ausentes")
        extras.append(values[order].astype(np.float32))
        log(f"  dormente {name}: {values.shape[1]}d")
    log(f"{args.dataset}: {len(stems):,} imagens, {len(classes)} classes, "
        f"{len(VIEWS)} views -> {2 ** len(VIEWS)} máscaras por fold")

    done = set()
    if args.output.exists():
        done = {(int(r.fold), int(r.mask)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} células")

    # O RAMO ANTIGO FAZIA StratifiedKFold INCONDICIONALMENTE -- nos quatro de
    # split publicado isso ignora a particao oficial (§7.5: no places_lt troca
    # 996x de desbalanceamento por 41x). Tamanhos IMPRESSOS, que e o antidoto.
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "experiments"))
    from evaluate_handcrafted_banks_lda import OFFICIAL as _OFF, TEST_COHORT as _TC
    if args.dataset in _TC:
        raise SystemExit(f"{args.dataset} tem coorte de teste SEPARADA ({_TC[args.dataset]}); "
                         f"este script ainda nao a empilha -- nao rodar sem isso")
    if args.dataset in _OFF:
        meta = pd.read_csv(ROOT / f"data/{args.dataset}_meta.csv", dtype={"stem": str})
        onde = meta.set_index("stem")["split"].reindex(stems).to_numpy(str)
        if (onde == "nan").any():
            raise SystemExit(f"{args.dataset}: {(onde == 'nan').sum()} stems fora do metadado")
        folds = [(np.flatnonzero(onde == "train"), np.flatnonzero(onde == "test"))]
        tipo = "split OFICIAL"
    else:
        folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                     .split(stems, labels))
        tipo = f"CV {N_FOLDS} folds"
    log(f"  [{tipo}] treino={len(folds[0][0]):,} teste={len(folds[0][1]):,} "
        f"classes={len(classes)}")
    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        masks = [m for m in range(2 ** len(VIEWS)) if (fold, m) not in done]
        if not masks:
            continue
        matrix, spans = build_fold(cohort, labels, labels_str, texts, classes,
                                   index_of, stop_words, train_idx, test_idx,
                                   fold, args.cache_dir / f"fold_{fold}.npz", extras)
        log(f"  fold {fold}: {len(masks):,} máscaras, {args.workers} workers")
        started = time.perf_counter()
        with Pool(args.workers, initializer=_init,
                  initargs=(matrix, labels, spans, train_idx, test_idx, len(classes))) as pool:
            rows = []
            for count, (mask, bacc, dims) in enumerate(
                    pool.imap_unordered(_evaluate, masks, chunksize=4), 1):
                rows.append(dict(fold=fold, mask=mask, n_views=bin(mask).count("1"),
                                 dimensions=dims, bacc=bacc))
                if count % 128 == 0 or count == len(masks):
                    frame = pd.DataFrame(rows)
                    frame.to_csv(args.output, mode="a", index=False,
                                 header=not args.output.exists(), float_format="%.4f")
                    rows = []
                    rate = count / (time.perf_counter() - started)
                    log(f"    {count:,}/{len(masks):,}  {rate:.1f} máscaras/s  "
                        f"restam {(len(masks) - count) / rate / 60:.0f} min")

    frame = pd.read_csv(args.output)
    n_part = len(folds)
    if len(frame) < n_part * 2 ** len(VIEWS):
        log("incompleto; rode de novo para retomar")
        return
    phis = []
    for fold in range(1, n_part + 1):
        v = dict(zip(frame[frame.fold == fold]["mask"], frame[frame.fold == fold]["bacc"]))
        phis.append(shapley(v, len(VIEWS)))
    phi = np.vstack(phis)
    full = frame[frame["mask"].eq(2 ** len(VIEWS) - 1)].groupby("fold").bacc.mean()
    print(f"\npipeline completo (LDA): {full.mean():.2f} BAcc")
    print(f"soma dos Shapley: {phi.mean(axis=0).sum():.2f}  (deve bater com "
          f"{full.mean() - 100 / len(classes):.2f} = completo - acaso)\n")
    print(f"{'view':22}{'Shapley':>9}{'±':>7}")
    order = np.argsort(-phi.mean(axis=0))
    for i in order:
        print(f"{display(VIEWS[i]):22}{phi[:, i].mean():+9.3f}{phi[:, i].std(ddof=1):7.3f}")


if __name__ == "__main__":
    main()
