"""Cache compacto dos RAGs ja exportados: so o que o Weisfeiler-Lehman precisa.

Os `.pt` em `outputs/<dataset>_<seg>_full/graphs/` tem 316 KB cada (32 GB por
substrato) e carregam `x` de 6 colunas `[L, a, b, cx, cy, log_area]`, `edge_index`
e `edge_attr`. O WL so usa **Lab** (para o codebook de cor) e **edge_index** (a
topologia), entao este cache guarda esses dois em float16/int16 -- ~10x menor, e
em disco LOCAL, porque o §1 mede NFS 12x mais lento e o estagio seguinte le o
cache uma vez POR FOLD (o codebook tem de ser ajustado so no treino, senao o
vocabulario fica transdutivo -- §6, o mecanismo que derrubou o SNF).

int16 nas arestas e seguro e conferido: o maior grafo da coorte tem ~2.400 nos,
bem abaixo do limite de 32.767. O script ABORTA se algum passar.

A ordem das linhas vem de `cohort.stems`, nunca de um glob -- o §5.18b custou
-2,92pp por reconstruir stems do metadado: o conjunto era o mesmo, a ORDEM nao.

Uso:
    python src/extractors/cache_rag_compact.py \
        --dataset multitask --graph-dir outputs/multitask_slic_full/graphs \
        --out-dir /tmp/prism_data/multitask_slic_rag --workers 32
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

SHARD = 2000
_WORKER: dict = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _init_worker(graph_dir: str) -> None:
    _WORKER["graph_dir"] = Path(graph_dir)


def _read_one(stem: str) -> tuple[str, np.ndarray | None, np.ndarray | None, str]:
    try:
        blob = torch.load(_WORKER["graph_dir"] / f"{stem}.pt", weights_only=False)
        lab = np.asarray(blob["x"][:, :3], dtype=np.float16)
        edges = np.asarray(blob["edge_index"], dtype=np.int64)
        if lab.shape[0] > 32000:
            return stem, None, None, f"{lab.shape[0]} nos excede int16"
        if edges.size and edges.max() >= lab.shape[0]:
            return stem, None, None, "edge_index aponta fora de x"
        return stem, lab, edges.astype(np.int16), ""
    except Exception as error:                            # noqa: BLE001
        return stem, None, None, f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from registry import load_blocks

    log(f"coorte de producao de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset)
    stems = [str(s) for s in cohort.stems]
    if args.limit:
        stems = stems[:args.limit]
    log(f"  {len(stems)} imagens")

    missing = [s for s in stems[:50] if not (args.graph_dir / f"{s}.pt").exists()]
    if missing:
        raise SystemExit(f"grafos ausentes em {args.graph_dir} (ex.: {missing[:3]})")

    shards = [(i, stems[i:i + SHARD]) for i in range(0, len(stems), SHARD)]
    log(f"{len(shards)} shards de ate {SHARD}; {args.workers} workers")

    started = time.time()
    for order, (offset, chunk) in enumerate(shards, 1):
        path = args.out_dir / f"shard_{offset:07d}.npz"
        if path.exists():
            continue
        labs, edges, node_off, edge_off, failures = [], [], [0], [0], []
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(str(args.graph_dir),)) as pool:
            for stem, lab, edge, err in pool.imap(_read_one, chunk, chunksize=8):
                if lab is None:
                    failures.append((stem, err))
                    continue
                labs.append(lab)
                edges.append(edge)
                node_off.append(node_off[-1] + lab.shape[0])
                edge_off.append(edge_off[-1] + edge.shape[1])
        if failures:
            raise SystemExit(f"{len(failures)} falhas no shard {offset} "
                             f"(ex.: {failures[:3]}); consertar e repetir")

        temporary = path.with_suffix(".npz.part")
        with open(temporary, "wb") as handle:        # §8: savez acrescenta .npz
            np.savez(handle,
                     lab=np.concatenate(labs, axis=0),
                     edges=np.concatenate(edges, axis=1),
                     node_offsets=np.asarray(node_off, dtype=np.int64),
                     edge_offsets=np.asarray(edge_off, dtype=np.int64),
                     stems=np.asarray(chunk))
        temporary.rename(path)
        done = offset + len(chunk)
        rate = done / max(1e-9, time.time() - started)
        log(f"  shard {order}/{len(shards)} ok  ({rate:.0f} img/s, "
            f"restam ~{(len(stems) - done) / max(1e-9, rate) / 60:.0f} min)")

    total = sum(f.stat().st_size for f in args.out_dir.glob("shard_*.npz"))
    log(f"pronto: {args.out_dir}  {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
