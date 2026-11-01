"""Cache em disco para a feature de grafo de similaridade.

O grafo é determinístico dado (espaço de construção, índices de treino, índices
de consulta, k, alpha, max_iter, semente) -- mas custa 30-106s por fold, e já
foi recalculado idêntico em três scripts diferentes no mesmo dia.

**O que NÃO dá para cachear**: um grafo único para o dataset todo. Ele é
ajustado só no treino do fold (nested OOF), então subamostrar ou trocar o fold
muda o grafo de verdade. A chave inclui os índices exatos justamente para que
uma mudança de partição nunca reuse um grafo errado em silêncio.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "outputs/similarity_graph_cache"


def cache_key(modalities, train_index, test_index, n_classes,
              n_neighbors, alpha, max_iter, seed) -> str:
    digest = hashlib.sha1()
    digest.update("|".join(modalities).encode())
    digest.update(np.ascontiguousarray(np.sort(train_index), dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(np.sort(test_index), dtype=np.int64).tobytes())
    digest.update(f"{n_classes}|{n_neighbors}|{alpha}|{max_iter}|{seed}".encode())
    return digest.hexdigest()[:20]


def cached_similarity_features(compute, modalities, train_index, test_index,
                               n_classes, n_neighbors, alpha, max_iter, seed,
                               cache_dir: Path = CACHE_DIR, log=print):
    """Devolve (oof_train, target) do cache, ou calcula e grava.

    `compute` é uma função sem argumentos que devolve o par -- normalmente um
    `functools.partial` de `rich_similarity_features`, para que o custo de
    montar o espaço só aconteça em caso de falta no cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(modalities, train_index, test_index, n_classes,
                    n_neighbors, alpha, max_iter, seed)
    path = cache_dir / f"{key}.npz"
    if path.exists():
        with np.load(path) as cached:
            oof, target = cached["oof"], cached["target"]
        if len(oof) == len(train_index) and len(target) == len(test_index):
            log(f"    grafo do cache: {path.name} ({oof.shape[1]}d)")
            return oof, target
        log(f"    ! cache {path.name} com forma inesperada, recalculando")

    started = time.perf_counter()
    oof, target = compute()
    # Gravação atômica: um job morto no meio (rotina nesta máquina) deixaria
    # um .npz truncado que o próximo run carregaria como se fosse válido.
    temporary = path.with_suffix(".npz.tmp")
    # Gravar por handle, não por nome: `np.savez_compressed` ACRESCENTA ".npz"
    # a um caminho que não termine em ".npz", então passar "xxx.npz.tmp" cria
    # "xxx.npz.tmp.npz" e o rename seguinte falha com FileNotFoundError.
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, oof=oof, target=target)
    temporary.replace(path)
    log(f"    grafo calculado em {time.perf_counter() - started:.0f}s, "
        f"salvo em {path.name}")
    return oof, target
