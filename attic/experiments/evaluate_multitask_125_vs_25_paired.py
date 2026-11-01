"""Multitask com 125 e com 25 classes, pareado em tudo o mais.

A comparação entre os dois recortes existia, mas nunca pareada, e as duas
medições disponíveis se contradiziam:

    125 classes, LDA, 7 mod (568d): ResNet 39,73 · hand 32,56 -> −7,17
     25 classes, LDA, 7 mod (568d): ResNet 49,50 · hand 38,32 -> −11,18
    125 classes, MLP, 7 mod (568d): ResNet 34,63 · hand 35,49 -> **+0,86**
     25 classes, MLP, 8 mod (639d): ResNet 50,14 · hand 49,49 -> −0,65

No MLP a diferença **troca de sinal** entre os recortes, mas as duas linhas não
são comparáveis: 125 rodou sem Felzenszwalb (não estava extraído em 21/ago) e
o hand-designed de 125 tem desvio de 2,92pp, 3 a 6x o de qualquer outra célula.

E nenhuma das quatro tem o grafo de similaridade. Ele só existe rodado em 25
classes, em duas versões, e o número principal do §4 (50,06, empate de −0,08
com a ResNet) usa a que inclui **Multiscale** -- o bloco que o §3 declara
aposentado. Na configuração declarada de produção o número é 49,70 e o gap é
−0,44.

Por que isso importa além da inconsistência: o subespaço discriminante do LDA é
limitado a `n_classes − 1`. Em 25 classes são 24 componentes, em 125 são 124.
Como a grade representação x dados do achado 5.1 foi medida com sonda LDA em 25
classes, toda ela lê 24 dimensões enquanto a entrada cresce de 71d para 681d. O
par LDA->LDA acima sugere que o teto **não é neutro entre os lados** -- afrouxá-lo
fecha 4pp do gap contra a ResNet, o que indica que ele penaliza o vetor
multi-bloco mais que o vetor denso da CNN.

Esta grade dá as 12 células que faltam, tudo o mais igual:

    recorte      x  classificador  x  condição
    {125, 25}       {LDA, MLP}        {hand-8, hand-8 + grafo, ResNet50}

MLP estreito (enc 128, cabeça 512-256-128) por decisão de 25/ago: é o
instrumento de produção do artigo, e mantê-lo é a defesa contra "afinaram a
favor do próprio método" (§6). Ressalva a registrar: em 125 classes a penúltima
camada tem 128 unidades para 125 classes -- não há gargalo estrutural de posto,
mas a margem é nula.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import load_block  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp  # noqa: E402
from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features)

SEED, N_FOLDS = 42, 5
VISUAL = ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
          "Hilbert", "Fourier", "Gabor"]
BENCHMARK = ROOT / "data/multitask_25class_benchmark.txt"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    """Normaliza rótulo para casar o arquivo do benchmark com o metadado.

    Três divergências, nenhuma delas com erro próprio:
      "Abstract Art"            vs "Abstract_Art"      (separador)
      "Analytical Realism"    vs espaço comum        (NBSP)
      "Naive Art (Primitivism)" vs "Naïve_Art_(...)"   (trema)
    Sem normalizar as três, casam 13 das 25; sem a terceira, 24 -- e 24 de 25 é
    exatamente o erro que passa batido.
    """
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


def clean(block: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(block)
    if bad.any():
        median = np.nanmedian(np.where(np.isfinite(block), block, np.nan), axis=0)
        block[bad] = np.take(np.nan_to_num(median), np.where(bad)[1])
    return block


def load_multitask(names: list[str]):
    frame = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})[["stem", "style"]]
    for name in names:
        block = load_block("multitask", name)
        if block is None:
            raise FileNotFoundError(f"bloco ausente: {name}")
        frame = frame.merge(block, on="stem", how="inner")
    frame = frame.reset_index(drop=True)
    blocks = {n: clean(frame[[c for c in frame.columns if c.startswith(f"{n}_")]]
                       .to_numpy(np.float32)) for n in names}
    return blocks, frame["style"].to_numpy(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", default=["125", "25"])
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"])
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--graph-dir", type=Path,
                        default=ROOT / "outputs/multitask_125_vs_25_graphs")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/multitask_125_vs_25_paired.csv")
    args = parser.parse_args()
    args.graph_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    names = VISUAL + ["ResNet50"]
    all_blocks, all_labels = load_multitask(names)
    log(f"coorte completa: {len(all_labels):,} imagens, "
        f"{len(np.unique(all_labels))} classes, {len(names)} blocos")

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(str(r.cut), str(r.classifier), str(r.condition), int(r.fold))
                for r in previous.itertuples()}
        log(f"retomando: {len(done)} células já feitas")

    for cut in args.cuts:
        if cut == "25":
            wanted = {key(l) for l in BENCHMARK.read_text().splitlines() if l.strip()}
            mask = np.array([key(v) in wanted for v in all_labels])
            got = len(np.unique(all_labels[mask]))
            if got != len(wanted):
                raise ValueError(f"casaram {got} de {len(wanted)} classes do benchmark")
        else:
            mask = np.ones(len(all_labels), dtype=bool)
        blocks = {n: b[mask] for n, b in all_blocks.items()}
        labels = LabelEncoder().fit_transform(all_labels[mask])
        n_classes = len(np.unique(labels))
        log(f"=== recorte {cut}: {len(labels):,} imagens, {n_classes} classes, "
            f"leitura do LDA = {n_classes - 1} componentes")

        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(labels)), labels))
        hand = [blocks[n] for n in VISUAL]
        resnet = [blocks["ResNet50"]]

        for fold, (train, test) in enumerate(splits, start=1):
            pending = [(c, cond) for c in args.classifiers
                       for cond in ("hand-8", "hand-8 + grafo", "ResNet50")
                       if (cut, c, cond, fold) not in done]
            if not pending:
                log(f"  recorte {cut} fold {fold}: completo, pulando")
                continue

            graph_block = None
            if any(cond == "hand-8 + grafo" for _, cond in pending):
                cache = args.graph_dir / f"graph_{cut}_fold{fold}.npy"
                if cache.exists():
                    graph_block = np.load(cache)
                else:
                    started = time.perf_counter()
                    space = np.hstack([
                        StandardScaler().fit(b[train]).transform(b).astype(np.float32)
                        for b in hand])
                    oof, target = rich_similarity_features(
                        space[train], labels[train], space[test], n_classes,
                        args.n_neighbors, args.alpha, args.max_iter, SEED + fold)
                    graph_block = np.zeros((len(labels), oof.shape[1]), dtype=np.float32)
                    graph_block[train] = prior_correct_rich(oof, labels[train], n_classes)
                    graph_block[test] = prior_correct_rich(target, labels[train], n_classes)
                    del space, oof, target
                    np.save(cache, graph_block)
                    log(f"  recorte {cut} fold {fold}: grafo {graph_block.shape[1]}d "
                        f"em {(time.perf_counter() - started) / 60:.1f} min")

            for classifier, condition in pending:
                use = {"hand-8": hand,
                       "hand-8 + grafo": hand + [graph_block],
                       "ResNet50": resnet}[condition]
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(use, labels, train, test)
                else:
                    metrics = evaluate_mlp(use, labels, train, test, fold,
                                           args.epochs, args.patience)
                pd.DataFrame([{
                    "cut": cut, "classifier": classifier, "condition": condition,
                    "fold": fold, "n_images": len(labels), "n_classes": n_classes,
                    "dimensions": sum(b.shape[1] for b in use),
                    "seconds": time.perf_counter() - started,
                    **{k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}}]).to_csv(
                    args.output, mode="a", index=False,
                    header=not args.output.exists(), float_format="%.6f")
                log(f"  {cut:>3}cl {classifier:<3} {condition:<15} f{fold} "
                    f"BAcc={100 * metrics['bacc']:.2f} "
                    f"({sum(b.shape[1] for b in use)}d, {time.perf_counter() - started:.0f}s)")

    summarize(args.output)


def summarize(path: Path) -> None:
    if not path.exists():
        return
    detail = pd.read_csv(path, dtype={"cut": str})
    print()
    for cut in ("125", "25"):
        block = detail[detail.cut.eq(cut)]
        if block.empty:
            continue
        print(f"=== {cut} classes "
              f"({int(block.n_images.iloc[0]):,} imagens, "
              f"leitura do LDA = {int(block.n_classes.iloc[0]) - 1} comp.) ===")
        for classifier in ("lda", "mlp"):
            piece = block[block.classifier.eq(classifier)]
            pivot = piece.pivot_table(index="fold", columns="condition", values="bacc")
            if pivot.empty:
                continue
            cells = {}
            for condition in ("hand-8", "hand-8 + grafo", "ResNet50"):
                if condition not in pivot:
                    cells[condition] = None
                    continue
                values = 100 * pivot[condition].to_numpy()
                cells[condition] = values
            line = f"  {classifier.upper():<4}"
            for condition in ("hand-8", "hand-8 + grafo", "ResNet50"):
                v = cells[condition]
                line += (f"  {condition}: " + ("—" if v is None
                         else f"{v.mean():.2f}±{v.std(ddof=1):.2f}"))
            if cells["hand-8 + grafo"] is not None and cells["ResNet50"] is not None:
                gap = cells["hand-8 + grafo"] - cells["ResNet50"]
                line += f"   gap={gap.mean():+.2f}"
            if cells["hand-8 + grafo"] is not None and cells["hand-8"] is not None:
                delta = cells["hand-8 + grafo"] - cells["hand-8"]
                _, p = ttest_rel(cells["hand-8 + grafo"], cells["hand-8"])
                line += f"   Δgrafo={delta.mean():+.2f} (p={p:.3f})"
            print(line)
        print()


if __name__ == "__main__":
    main()
