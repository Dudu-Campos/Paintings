"""Persiste os RAGs do Felzenszwalb, no mesmo formato do exportador do SLIC.

Motivação: o bloco "Line graph" do pipeline é L(G) do RAG do **SLIC**, e é o
operador de maior valor medido (7,04pp de Shapley no MLP). A pergunta óbvia --
e a que um revisor faz -- é por que o mesmo operador nunca foi aplicado ao RAG
do Felzenszwalb, cujo grafo base isolado é melhor que o do SLIC (31,3% contra
29,7%). Este exportador produz os grafos que faltam para responder isso.

Os parâmetros são exatamente os de `outputs/wikiart_felzenszwalb_full_s50_m50`
(scale=50, min_size=50, sigma=0,8, max_side=1024), para que o line graph caia
sobre o mesmo RAG que gerou as features de Felzenszwalb já usadas no pipeline.

Saída idêntica à de `extract_slic_rag_graphs.py` -- um `.pt` e um `.label` por
imagem, mais `index.csv` -- para que `extract_line_graph_features.py` consuma
o diretório sem nenhuma mudança (`--graph-dir`).

Retomável: pula qualquer stem que já tenha `.pt` no disco.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from segmentation_multigraph_features import segment_region_graph
from slic_graph_features import resolve_wikiart_records, stratified_sample

NODE_FIELDS = ("L", "a", "b", "X", "Y", "area_frac")
METHOD = "felzenszwalb"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def extract_one(task: tuple[str, str, str, dict]) -> tuple[dict | None, str | None]:
    stem, style, path, parameters = task
    try:
        nodes, edges = segment_region_graph(path, METHOD, parameters)
        node_values = np.column_stack([nodes[field] for field in NODE_FIELDS]).astype(
            np.float32, copy=False
        )
        # Mesma transformação do exportador do SLIC: área é positiva e muito
        # assimétrica, o log preserva a informação e se comporta melhor.
        node_values[:, -1] = np.log(np.clip(node_values[:, -1], 1e-8, None))
        source = edges["src"].astype(np.int64, copy=False)
        target = edges["dst"].astype(np.int64, copy=False)
        edge_index = np.concatenate(
            [np.stack([source, target]), np.stack([target, source])], axis=1
        )
        edge_attr = np.column_stack(
            [edges["color_dist"], edges["spatial_dist"]]
        ).astype(np.float32, copy=False)
        edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)
        return {
            "stem": stem, "style": style,
            "x": node_values, "edge_index": edge_index, "edge_attr": edge_attr,
            "n_nodes": len(nodes), "n_edges": len(edges),
        }, None
    except Exception as exc:  # pragma: no cover - imagem corrompida
        return None, f"{style}/{stem}: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--image-dir", type=Path, default=root / "wikiart")
    parser.add_argument("--meta", type=Path, default=root / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/felzenszwalb_gnn_full/graphs")
    parser.add_argument("--sample-per-style", type=int, default=0,
                        help="0 = todas as imagens (padrão)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--felz-scale", type=float, default=50.0)
    parser.add_argument("--felz-min-size", type=int, default=50)
    parser.add_argument("--felz-sigma", type=float, default=0.8)
    parser.add_argument("--segments", type=int, default=2048)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    args = parser.parse_args()

    parameters = {
        "felz_scale": args.felz_scale,
        "felz_min_size": args.felz_min_size,
        "felz_sigma": args.felz_sigma,
        "segments": args.segments,
        "max_side": args.max_side,
    }

    records = resolve_wikiart_records(args.meta, args.image_dir).drop_duplicates("stem")
    records = (stratified_sample(records, args.sample_per_style, args.seed)
               .sort_values("stem").reset_index(drop=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in args.output_dir.glob("*.pt")}
    pending = records.loc[~records.stem.isin(existing)]
    tasks = [(row.stem, row.style, row.path, parameters)
             for row in pending.itertuples(index=False)]
    # Imagens grandes primeiro: melhora o balanceamento na cauda do pool.
    tasks.sort(key=lambda task: Path(task[2]).stat().st_size, reverse=True)
    log(f"Felzenszwalb RAG export: {len(records):,} pedidas, {len(existing):,} já salvas, "
        f"{len(tasks):,} pendentes; parâmetros={parameters}, workers={args.workers}")

    failures: list[str] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            graph, error = future.result()
            if graph is None:
                failures.append(error or "erro desconhecido")
            else:
                stem = str(graph.pop("stem"))
                style = str(graph.pop("style"))
                torch.save(graph, args.output_dir / f"{stem}.pt")
                (args.output_dir / f"{stem}.label").write_text(style + "\n", encoding="utf-8")
            if done % args.checkpoint_every == 0 or done == len(tasks):
                rate = done / max(time.perf_counter() - started, 1e-9)
                remaining = (len(tasks) - done) / max(rate, 1e-9)
                log(f"{done:,}/{len(tasks):,}; falhas={len(failures):,}; "
                    f"{rate:.2f} grafos/s; ETA {remaining / 60:.0f} min")

    # Não recarrega os .pt só para contar nós: na escala completa isso releria
    # dezenas de GB. O .label ao lado basta para montar o índice.
    rows = []
    for graph_path in sorted(args.output_dir.glob("*.pt")):
        label_path = graph_path.with_suffix(".label")
        if not label_path.exists():
            continue
        rows.append({"stem": graph_path.stem,
                     "style": label_path.read_text(encoding="utf-8").strip()})
    pd.DataFrame(rows).sort_values("stem").to_csv(
        args.output_dir.parent / "index.csv", index=False)
    if failures:
        (args.output_dir.parent / "failures.txt").write_text(
            "\n".join(failures) + "\n", encoding="utf-8")
    log(f"Salvos {len(rows):,} grafos em {args.output_dir}")


if __name__ == "__main__":
    main()
