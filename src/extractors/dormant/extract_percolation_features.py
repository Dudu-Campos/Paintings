"""Hierarquia cromática do RAG: persistência de dimensão 0 sobre os grafos já exportados.

Todo o resto do pipeline mede **um** nível de escala -- SLIC com orçamento fixo
de 2.048 segmentos, Felzenszwalb com `scale=50`, Quickshift com `max_dist=10`.
Cada extrator escolhe uma escala e vive nela. O que falta é como a obra se
organiza *entre* escalas: quantas massas cromáticas sobrevivem conforme a
tolerância de cor cresce, e com que desigualdade.

O projeto já teve isso e perdeu sem medir. O `coarse2.cpp` (removido em 387e46d)
varria 12 limiares e gravava oito estatísticas de percolação por limiar, mas o
leitor padrão **descartava os steps** e devolvia só o grafo materializado em ε*
-- a curva servia apenas para escolher o joelho. A comparação que aposentou
aquele pipeline (percolação 21,67 contra SLIC-2048 25,17, mesmas 32 features)
foi um nível contra um nível; a hierarquia nunca entrou em nenhuma linha dela.

Aqui ela volta sem C++ e sem reler pixel, porque **os grafos exportados já a
contêm exatamente**. Cortar as arestas com `color_dist > ε` e contar componentes
é a filtração; e os componentes em qualquer ε são determinados pela sequência de
fusões de Kruskal, então uma única varredura ordenada dá a curva inteira em vez
de uma união-busca por limiar. As alturas de fusão são as arestas da árvore
geradora mínima, e o objeto é o dendrograma de ligação simples do RAG --
equivalentemente, a persistência de dimensão 0 da filtração por distância
cromática. Nome que existe na literatura, não análise caseira.

Duas decisões de desenho que definem se a medida vale alguma coisa:

**Ponderar por área, nunca contar nós.** O SLIC fixa o orçamento em 2.048
segmentos, então uma curva de contagem de nós mede o orçamento e não a obra.
Com peso de área, `giant` volta a dizer "que fração da tela virou uma massa só".
Vale ainda mais no Felzenszwalb, onde a contagem varia 699→3.670 e o HANDOFF
antigo já tinha registrado que a maior parte do ganho dele sobrevive à remoção
dos indicadores diretos de orçamento.

**Três famílias, decididas por medição e não por gosto.** `perc_*` é a herdeira
direta (grade fixa de ε, em unidades Lab absolutas, comparável entre imagens);
`reach_*` é a função inversa, livre de escala ("em que tolerância metade da tela
vira uma região?"); `merge_*` resume a distribuição de alturas de fusão. Os
prefixos existem para que o marginal possa testar as famílias separadamente sem
reextrair nada.

Uso:
    python experiments/extract_percolation_features.py \
        --index outputs/slic_gnn_full/index.csv \
        --graph-dir outputs/slic_gnn_full/graphs \
        --output results/percolation_slic_full_features.parquet
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Grade em unidades Lab, posicionada por medição e não herdada. A do
# `coarse2.cpp` ia de 2 a 40 porque percolava **pixels**; entre regiões
# adjacentes de um RAG as distâncias são uma ordem de grandeza menores, e com
# grau médio ~5,7 o componente gigante aparece cedo: em ε=2 o SLIC já está em
# 0,98 de área, ou seja a grade antiga cairia inteira depois do fim da ação.
# A transição medida mora em 0,5-4 (SLIC) e 0,5-6 (Felzenszwalb).
# `0.0` entra porque um quarto das arestas do SLIC tem distância exatamente
# nula: é a fração da tela cromaticamente chapada, e ela sozinha já forma um
# componente de 22% de área em média. Não há ponto entre 0 e 0,5 porque o Lab
# de 8 bits não produz distâncias ali -- a curva é literalmente plana.
EPSILONS = (0.0, 0.5, 0.65, 0.85, 1.1, 1.45, 1.9, 2.5, 3.3, 4.5, 6.5, 9.5)
# Frações de área do componente gigante e de sobrevivência de componentes cujos
# limiares de alcance viram feature (família `reach_*`).
GIANT_TARGETS = (0.10, 0.25, 0.50, 0.75, 0.90)
SURVIVAL_TARGETS = (0.50, 0.25, 0.10, 0.02)
MERGE_QUANTILES = (5, 10, 25, 50, 75, 90, 95)


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    total = values.sum()
    if values.size == 0 or total <= 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float((values.size + 1 - 2 * cumulative.sum() / cumulative[-1]) / values.size)


def component_stats(areas: np.ndarray, singleton_area: float,
                    n_nodes: int) -> tuple[float, ...]:
    """As seis estatísticas de um estado da filtração, todas ponderadas por área.

    `areas` são as frações de área dos componentes vivos e somam 1. A entropia
    é normalizada por `log(n_nodes)` (e não por `log(n_componentes)`) de
    propósito: assim ela cai junto com a contagem, medindo o colapso da
    estrutura, em vez de virar pura uniformidade da partição sobrevivente.
    """
    order = np.sort(areas)[::-1]
    positive = order[order > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    return (
        len(order) / n_nodes,                                   # comp_frac
        float(order[0]) if order.size else 0.0,                 # giant
        float(order[1]) if order.size > 1 else 0.0,             # second
        entropy / np.log(n_nodes) if n_nodes > 1 else 0.0,      # entropy
        gini(order),                                            # gini
        singleton_area,                                         # singleton_area
    )


def filtration(source: np.ndarray, target: np.ndarray, weight: np.ndarray,
               area: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kruskal em ordem crescente; devolve os estados da curva e as alturas.

    Uma passada resolve a filtração inteira porque os componentes em qualquer ε
    são exatamente os desta sequência truncada em ε. As arestas que de fato
    fundem são as da árvore geradora mínima, e seus pesos são as alturas do
    dendrograma.
    """
    n_nodes = len(area)
    parent = np.arange(n_nodes, dtype=np.int64)
    size = area.astype(np.float64, copy=True)
    count = np.ones(n_nodes, dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:                      # compressão de caminho
            parent[node], node = root, parent[node]
        return root

    order = np.argsort(weight, kind="stable")
    grid_index, heights = 0, []
    # Estados amostrados na grade: capturados **antes** de processar a primeira
    # aresta com peso acima do limiar, que é a definição do corte em ε.
    samples = np.zeros((len(EPSILONS), 6), dtype=np.float64)
    # Curva (altura, gigante, comp_frac) em cada fusão, para inverter depois.
    reach = np.zeros((len(order) + 1, 3), dtype=np.float64)
    reach[0] = (0.0, float(size.max()) if n_nodes else 0.0, 1.0)
    merges = 0

    def snapshot() -> tuple[float, ...]:
        roots = np.flatnonzero(parent == np.arange(n_nodes))
        singles = float(size[roots][count[roots] == 1].sum())
        return component_stats(size[roots], singles, n_nodes)

    for position in order:
        threshold = weight[position]
        while grid_index < len(EPSILONS) and EPSILONS[grid_index] < threshold:
            samples[grid_index] = snapshot()
            grid_index += 1
        left, right = find(int(source[position])), find(int(target[position]))
        if left == right:
            continue
        if count[left] < count[right]:                   # união por tamanho
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
        count[left] += count[right]
        merges += 1
        heights.append(float(threshold))
        reach[merges] = (float(threshold), float(size[left]),
                         (n_nodes - merges) / n_nodes)
    while grid_index < len(EPSILONS):
        samples[grid_index] = snapshot()
        grid_index += 1
    # `size[left]` é o componente que acabou de crescer, não necessariamente o
    # maior; o máximo acumulado corrige isso sem custo.
    reach = reach[: merges + 1]
    reach[:, 1] = np.maximum.accumulate(reach[:, 1])
    return samples, reach, np.asarray(heights, dtype=np.float64)


def first_crossing(level: np.ndarray, height: np.ndarray, target: float,
                   rising: bool, censor: float) -> float:
    """Altura em que a curva cruza `target` pela primeira vez.

    Sem cruzamento (RAG desconexo, ou tolerância insuficiente) o valor é
    censurado no maior peso observado -- é a informação honesta "não chegou até
    aqui", e mantém a coluna na mesma unidade das que chegaram.
    """
    hits = np.flatnonzero(level >= target) if rising else np.flatnonzero(level <= target)
    return float(height[hits[0]]) if hits.size else censor


def extract_one(task: tuple[str, str, str]) -> tuple[dict | None, str | None]:
    stem, style, path_value = task
    try:
        graph = torch.load(path_value, map_location="cpu", weights_only=False)
        nodes = np.asarray(graph["x"], dtype=np.float64)
        n_nodes, n_edges = len(nodes), int(graph["n_edges"])
        # O exportador guarda o log da fração de área (`extract_slic_rag_graphs.py:38`);
        # sem desfazer isso as "áreas" somariam qualquer coisa menos 1.
        area = np.exp(nodes[:, -1])
        area = area / area.sum() if area.sum() > 0 else np.full(n_nodes, 1 / n_nodes)
        # Arestas não direcionadas vêm uma vez e depois em cópia invertida.
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)[:, :n_edges]
        weight = np.asarray(graph["edge_attr"], dtype=np.float64)[:n_edges, 0]
        source, target = edge_index

        samples, reach, heights = filtration(source, target, weight, area)
        censor = float(weight.max()) if n_edges else 0.0
        row: dict[str, float | str] = {"stem": stem, "style": style}
        names = ("comp_frac", "giant", "second", "entropy", "gini", "singleton_area")
        for index, epsilon in enumerate(EPSILONS):
            tag = f"{epsilon:g}".replace(".", "p")
            for column, name in enumerate(names):
                row[f"perc_{name}_e{tag}"] = float(samples[index, column])
        for fraction in GIANT_TARGETS:
            row[f"reach_giant_{int(100 * fraction):02d}"] = first_crossing(
                reach[:, 1], reach[:, 0], fraction, True, censor)
        for fraction in SURVIVAL_TARGETS:
            row[f"reach_comp_{int(100 * fraction):02d}"] = first_crossing(
                reach[:, 2], reach[:, 0], fraction, False, censor)
        if heights.size:
            row.update({f"merge_q{q:02d}": float(np.percentile(heights, q))
                        for q in MERGE_QUANTILES})
            gaps = np.diff(np.sort(heights))
            widest = int(np.argmax(gaps)) if gaps.size else 0
            row.update({
                "merge_mean": float(heights.mean()),
                "merge_std": float(heights.std()),
                "merge_gini": gini(heights),
                "merge_zero_frac": float((heights <= 0).mean()),
                # Maior salto entre fusões consecutivas: a separação cromática
                # que mais "persiste" na obra, e a altura em que ela acontece.
                "merge_gap_max": float(gaps.max()) if gaps.size else 0.0,
                "merge_gap_at": float(np.sort(heights)[widest]) if gaps.size else 0.0,
                "merge_frac": heights.size / max(n_nodes - 1, 1)})
        else:
            row.update({f"merge_q{q:02d}": 0.0 for q in MERGE_QUANTILES})
            row.update({key: 0.0 for key in (
                "merge_mean", "merge_std", "merge_gini", "merge_zero_frac",
                "merge_gap_max", "merge_gap_at", "merge_frac")})
        return row, None
    except Exception as error:                       # uma imagem não paga o lote
        return None, f"{style}/{stem}: {type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    # Sem default: `outputs/slic_gnn_full` (WikiArt) foi apagado em 08/set --
    # intermediario consumido, e as imagens daquele corpus nao existem mais. Um
    # default morto faria o extrator achar zero grafos e gravar nada EM
    # SILENCIO, que e o modo de falha do §8.
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=root / "results/percolation_slic_full_features.parquet")
    parser.add_argument("--limit", type=int, default=0,
                        help="processa só as N primeiras (piloto)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    index = pd.read_csv(args.index)[["stem", "style"]].drop_duplicates("stem")
    if args.limit:
        index = index.head(args.limit)
    completed = pd.DataFrame()
    if args.output.exists() and not args.overwrite:
        completed = pd.read_parquet(args.output)
        print(f"retomando {len(completed):,} linhas de {args.output}", flush=True)
    done_stems = set(completed["stem"].astype(str)) if not completed.empty else set()
    tasks = [(str(row.stem), str(row.style), str(args.graph_dir / f"{row.stem}.pt"))
             for row in index.itertuples(index=False)
             if str(row.stem) not in done_stems]
    rows: list[dict] = completed.to_dict("records") if not completed.empty else []
    failures: list[str] = []
    started = time.perf_counter()
    print(f"{len(index):,} grafos no índice; {len(tasks):,} pendentes; "
          f"workers={args.workers}", flush=True)

    def save() -> None:
        frame = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(args.output)
        return frame

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            row, error = future.result()
            if row is None:
                failures.append(error or "falha desconhecida")
            else:
                rows.append(row)
            if done % 2000 == 0 or done == len(tasks):
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(f"{done:,}/{len(tasks):,}; total={len(rows):,}; "
                      f"falhas={len(failures):,}; {rate:.1f} grafos/s", flush=True)
            if done % args.checkpoint_every == 0:
                save()
    frame = save()
    if failures:
        path = args.output.with_suffix(".failures.txt")
        path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"! {len(failures):,} falhas em {path}", flush=True)
    print(f"salvo={args.output} shape={frame.shape} "
          f"tempo={time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
