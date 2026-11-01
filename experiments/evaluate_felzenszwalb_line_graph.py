"""Line graph do RAG do Felzenszwalb: soma ao pipeline, ou substitui o do SLIC?

O bloco "Line graph" do pipeline é L(G) do RAG do **SLIC** e é o operador de
maior valor medido (7,04pp de Shapley no MLP). A pergunta que um revisor faz --
e que nunca tinha sido testada -- é por que o mesmo operador nunca foi aplicado
ao RAG do Felzenszwalb, cujo grafo base isolado é melhor que o do SLIC (31,3%
contra 29,7%).

`src/pipeline/extract_felzenszwalb_rag_graphs.py` exportou os 79.902 RAGs e
`extract_line_graph_features.py` extraiu as mesmas 63 features sobre eles.
Aqui as duas perguntas distintas, pareadas nos mesmos folds:

    base          melhor config (1.091d): 7 modalidades + quadrantes de
                  Descriptors. Já contém L(G_slic).
    + L(G_felz)   base mais um branch de 63d  -> o operador soma?
    swap          base com L(G_slic) trocado por L(G_felz), mesmas dimensões
                  -> o operador rende mais sobre um substrato melhor?

A segunda é a pergunta científica: sem redundância em jogo, testa operador
sobre substrato. A primeira é a que decide se entra no pipeline.

Saída: outputs/felzenszwalb_line_graph/
    fold_results.csv   BAcc por condição, classificador e fold
    summary.csv        deltas pareados contra a base, com t-test
"""
from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS_PER_WORKER)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

torch.set_num_threads(int(THREADS_PER_WORKER))

from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_felz_line(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Mesmo esquema do line graph do SLIC, prefixo diferente."""
    raw = pd.read_parquet(path)
    columns = [column for column in raw if column not in {"stem", "style"}]
    if len(columns) != 63:
        raise ValueError(f"esperava 63 features, achei {len(columns)}")
    rename = {column: f"felzline_{column}" for column in columns}
    block = raw[["stem"] + columns].drop_duplicates("stem").rename(columns=rename)
    before = len(frame)
    frame = frame.merge(block, on="stem", how="inner", validate="one_to_one")
    if len(frame) != before:
        log(f"aviso: coorte caiu de {before:,} para {len(frame):,} no merge")
    return frame, list(rename.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path,
                        default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path,
                        default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-line-features", type=Path,
                        default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--felz-features", type=Path,
                        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path,
                        default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/felzenszwalb_line_graph")
    parser.add_argument("--classifiers", nargs="+", default=["mlp", "lda"],
                        choices=["mlp", "lda"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                    for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra = {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    felzline_block = frame[felzline_columns].to_numpy(dtype=np.float32)
    descriptor_quadrants, quadrant_names = family_quadrant_blocks("Descriptors", group_blocks)
    log(f"{len(frame):,} imagens, {n_classes} estilos, "
        f"L(G_felz) com {felzline_block.shape[1]} dimensões")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["classifier"]), str(r["condition"]), int(r["fold"])) for r in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra = {**fixed_extra, "Text": text_block}
        base_blocks, base_names = selected_blocks(BASELINE_MASK, group_blocks, extra)
        base_blocks = base_blocks + descriptor_quadrants
        base_names = base_names + quadrant_names

        # Substituição: troca o bloco de line graph do SLIC pelo do Felzenszwalb.
        swap_index = base_names.index("Line graph")
        swap_blocks = list(base_blocks)
        swap_blocks[swap_index] = felzline_block
        swap_names = list(base_names)
        swap_names[swap_index] = "Line graph (Felz)"

        conditions = {
            "base": (base_blocks, base_names),
            "+ L(G_felz)": (base_blocks + [felzline_block], base_names + ["Line graph (Felz)"]),
            "swap L(G_slic) -> L(G_felz)": (swap_blocks, swap_names),
        }
        for condition, (blocks, names) in conditions.items():
            dims = sum(block.shape[1] for block in blocks)
            for classifier in args.classifiers:
                if (classifier, condition, fold) in completed:
                    continue
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(blocks, labels, train, test)
                else:
                    metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                           args.epochs, args.patience)
                elapsed = time.perf_counter() - started
                rows.append({"fold": fold, "classifier": classifier, "condition": condition,
                             "n_branches": len(blocks), "dimensions": dims,
                             "elapsed_seconds": elapsed, **metrics})
                completed.add((classifier, condition, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(f"  fold {fold} | {classifier:<3} | {condition:<28} | {dims:5d}d "
                    f"| BAcc={metrics['bacc']:.4f} | {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    summary = []
    for classifier in detail["classifier"].unique():
        sub = detail[detail.classifier == classifier]
        base = sub[sub.condition == "base"].sort_values("fold")["bacc"].to_numpy()
        for condition in ("+ L(G_felz)", "swap L(G_slic) -> L(G_felz)"):
            other = sub[sub.condition == condition].sort_values("fold")["bacc"].to_numpy()
            if len(other) != len(base) or len(base) < 2:
                continue
            delta = 100 * (other - base)
            statistic, pvalue = ttest_rel(other, base)
            summary.append({"classifier": classifier, "condition": condition,
                            "folds": len(base),
                            "base_bacc_mean": 100 * base.mean(),
                            "condition_bacc_mean": 100 * other.mean(),
                            "delta_pp_mean": delta.mean(), "delta_pp_std": delta.std(),
                            "t_statistic": statistic, "p_paired": pvalue,
                            "positive_folds": int((delta > 0).sum())})
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log("\n=== Resumo ===")
    log(summary.round(4).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
