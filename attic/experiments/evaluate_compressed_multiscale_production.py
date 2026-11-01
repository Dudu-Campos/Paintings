"""Multiscale comprimido dentro da configuração de produção (com Text).

Dois resultados do projeto apontam para lados opostos e nunca se encontraram
no mesmo vetor:

  achado 11   no MLP, **descartar** os quadrantes de SLIC e Hilbert ganha
              +0,60pp -- só a família Descriptors paga o próprio branch.
              Config vencedora: 1.154d em 12 branches, 54,42%.
  grade       na sonda de escala, **comprimir** as três famílias a 91
              componentes empata com os 1.576d nativos (+0,05pp, p=0,92) e
              ganha nos dois maiores tamanhos de treino (+0,66 / +0,69).

A leitura possível é que o problema nunca foi *quais* famílias, e sim
*quantas dimensões* elas ocupavam: quatro branches de 130d custam capacidade,
um branch de 91d não. Se for isso, as famílias descartadas pelo achado 11
voltam a ser úteis assim que param de custar caro -- e o vetor encolhe.

Condições (todas com Text e L(G_felz), mesmos folds, mesmo MLP):

    base            quadrantes de Descriptors nativos, 4 branches   1.154d
    ms91            as três famílias, PCA-91, 1 branch                725d
    ms256           as três famílias, PCA-256, 1 branch               890d
    desc91          só Descriptors, PCA-91, 1 branch                  725d
    base+ms91       base mais as três comprimidas a 91               1.245d

`desc91` é o controle que separa as duas hipóteses: se ele empatar com `ms91`,
o ganho veio de comprimir; se `ms91` ganhar dele, veio de recuperar SLIC e
Hilbert. Sem esse braço os dois efeitos ficam confundidos.

PCA ajustado só no treino de cada fold. Retomável por (condição, fold).

Saída: outputs/compressed_multiscale_production/
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_felzenszwalb_line_graph import load_felz_line
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset

FAMILIES = ("SLIC", "Descriptors", "Hilbert")
# (nome, famílias comprimidas, largura, mantém quadrantes nativos de Descriptors)
CONDITIONS = (
    ("base",       (),                  0,   True),
    ("ms91",       FAMILIES,            91,  False),
    ("ms256",      FAMILIES,            256, False),
    ("desc91",     ("Descriptors",),    91,  False),
    ("base+ms91",  FAMILIES,            91,  True),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def compress(blocks, train, width):
    """Concatena os blocos e projeta em `width` componentes, fit só no treino."""
    values = np.hstack(blocks)
    imputer = SimpleImputer(strategy="median").fit(values[train])
    filled = imputer.transform(values)
    scaler = StandardScaler().fit(filled[train])
    scaled = scaler.transform(filled)
    reducer = PCA(n_components=width, svd_solver="randomized",
                  random_state=SEED).fit(scaled[train])
    return reducer.transform(scaled).astype(np.float32)


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
                        default=ROOT / "outputs/compressed_multiscale_production")
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
    quadrants = {family: family_quadrant_blocks(family, group_blocks)[0]
                 for family in FAMILIES}
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    log(f"{len(frame):,} imagens, {n_classes} estilos")

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(str(r["condition"]), str(r["classifier"]), int(r["fold"])) for r in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        core, _ = selected_blocks(BASELINE_MASK, group_blocks,
                                  {**fixed_extra, "Text": text_block})
        core = core + [felzline_block]

        for name, families, width, native_descriptors in CONDITIONS:
            if all((name, c, fold) in done for c in args.classifiers):
                continue
            blocks = list(core)
            if native_descriptors:
                blocks = blocks + quadrants["Descriptors"]
            if families:
                blocks = blocks + [compress(
                    [b for family in families for b in quadrants[family]],
                    train, width)]
            dimensions = sum(b.shape[1] for b in blocks)

            for classifier in args.classifiers:
                if (name, classifier, fold) in done:
                    continue
                started = time.perf_counter()
                if classifier == "mlp":
                    metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                           args.epochs, args.patience)
                else:
                    metrics = evaluate_lda(blocks, labels, train, test)
                rows.append({"condition": name, "classifier": classifier,
                             "fold": fold, "dimensions": dimensions,
                             "n_branches": len(blocks),
                             "elapsed_seconds": time.perf_counter() - started,
                             **{k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}})
                done.add((name, classifier, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(f"  fold {fold} | {name:<10} | {classifier} | {dimensions:5d}d "
                    f"{len(blocks):2d}br | BAcc={100 * metrics['bacc']:.2f}%")

    detail = pd.DataFrame(rows)
    for classifier in args.classifiers:
        block = detail[detail.classifier == classifier]
        if block.empty:
            continue
        table = block.pivot_table(index="fold", columns="condition", values="bacc") * 100
        if "base" not in table:
            continue
        log(f"\n=== {classifier.upper()} ===")
        log(f"{'condição':<12}{'dims':>6}{'br':>4}{'BAcc':>9}{'±':>7}"
            f"{'vs base':>9}{'p':>8}{'folds':>7}")
        for name, *_ in CONDITIONS:
            if name not in table:
                continue
            info = block[block.condition == name].iloc[0]
            delta = (table[name] - table["base"]).dropna()
            p = ttest_rel(table[name], table["base"])[1] if name != "base" else np.nan
            log(f"{name:<12}{int(info.dimensions):>6}{int(info.n_branches):>4}"
                f"{table[name].mean():>9.2f}{table[name].std():>7.2f}"
                f"{delta.mean():>+9.2f}{p:>8.4f}"
                f"{int((delta > 0).sum()):>4}/{len(delta)}")
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
