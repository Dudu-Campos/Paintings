"""O ganho de theta do Fourier sobrevive à concatenação com as outras 7 views?

Terceiro teste de transferência da série (Hilbert §5.17, Fourier §5.17b), e o
**mais forte dos três**, por uma razão registrada antes de rodar: o §5.10 mede o
Fourier com o **maior LOO do banco** (+0,65pp, o bloco menos substituível pelos
outros sete). Hilbert tem LOO de −0,03 e Fourier de +0,33 -- neles a não-
transferência era quase esperada, porque o extra já estava coberto. Aqui não
está. Se algum theta vai transferir, é este.

**E o confundidor de capacidade não existe neste caso.** A melhor configuração
isolada é `(512,12)`, que tem **exatamente as mesmas 152 dimensões** de theta*
`(256,12)` -- ao contrário do Hilbert (57d -> 201d) e do Fourier (32d -> 64d), onde
foi preciso um braço de subamostra aleatória para separar informação de
capacidade. Aqui a comparação já nasce pareada em dimensionalidade.

**Previsão registrada:** transfere pouco, mas mais que os dois anteriores (0% e
5%), porque é o único com folga de insubstituibilidade. Se der zero também, a
afirmação do §5.17 fica robusta até no bloco menos coberto pelos outros.

Braços:
  `(512,12)`  melhor isolada, +1,12pp, MESMAS dims. O teste limpo.
  `(256,24)`  +162 colunas mortas (27,7% do vetor) e −0,76pp isolada. Mede se o
              peso morto que a granularidade compra atrapalha o BranchMLP, ou se
              o encoder por bloco simplesmente o ignora.
  `(64,12)`   pior resolução, −2,57pp isolada. Ancora o outro extremo: se nem
              perder 2,6pp isolado move o pipeline, a planura não é sobre o topo
              da grade.
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
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from evaluate_full_factorial_ablation import evaluate_mlp  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402

STAR = (256, 12)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_fourier")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--thetas", nargs="+",
                        default=["256x12", "512x12", "256x24", "64x12"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--isolated", action="store_true",
                          help="mede a view SOZINHA, mesmo laco de treino")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/theta_fourier_pipeline.csv")
    args = parser.parse_args()

    blocks, raw_labels, _ = load_corpus("multitask", VISUAL + ["Gabor"])
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    mask = np.array([key(v) in wanted for v in raw_labels])
    if len(np.unique(raw_labels[mask])) != len(wanted):
        raise ValueError("casamento de classes do benchmark incompleto")
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels[mask])
    n_classes = len(np.unique(labels))
    # VISUAL ja inclui "Fourier"; somar de novo duplicaria o bloco (§10.8).
    order = [n for n in VISUAL + ["Gabor"] if n in blocks]
    if "Fourier" not in order:
        raise SystemExit("o bloco Fourier não está na coorte -- nada a trocar")
    log(f"hand-8: {len(labels):,} imagens, {n_classes} classes, blocos={order}")

    # A coorte do registro define a ordem das linhas; os stems de cada extração
    # de theta são realinhados a ela, e não o contrário.
    cohort = pd.read_csv(args.meta, dtype={"stem": str})
    cohort = cohort[[key(str(v)) in wanted for v in cohort["style"]]]
    reference = cohort.stem.to_numpy()
    if len(reference) != len(labels):
        raise ValueError(f"coorte do registro ({len(labels)}) != metadado ({len(reference)})")

    def fourier_block(size: int, bins: int) -> np.ndarray:
        frame = pd.read_parquet(args.work_dir / f"s{size}_b{bins}.parquet")
        columns = [c for c in frame.columns if c not in ("stem", "style")]
        stems = frame.stem.astype(str).to_numpy()
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        if index.isna().any():
            raise ValueError(f"({size},{bins}): {int(index.isna().sum())} stems ausentes")
        return clean(frame[columns].to_numpy(np.float64)[index.to_numpy(dtype=int)])

    star_name = f"fourier {STAR[0]}x{STAR[1]}"
    conditions = {}
    for spec in args.thetas:
        size, bins = (int(v) for v in spec.split("x"))
        conditions[f"fourier {size}x{bins}"] = fourier_block(size, bins)
    if star_name not in conditions:
        raise SystemExit(f"theta* ({star_name}) tem de estar na lista: é a referência")
    # Sem braço de subamostra: (512,12) já tem as mesmas 152 dims de theta*.

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    rows = []
    for name, block in conditions.items():
        # `--isolated`: a MESMA view sozinha, pelo MESMO carregador e o MESMO
        # `evaluate_mlp`. O §5.17 media o lado isolado com sonda LDA e o lado do
        # banco com BranchMLP, e em 10/set o Hilbert mostrou que 1/3 do
        # encolhimento aparente da dispersao era so a compressao LDA->MLP
        # (0,66x, batendo a razao do §5.9d). Instrumento pareado por
        # construcao, nao por conferencia.
        chosen = [block] if args.isolated else [blocks[n] if n != "Fourier" else block for n in order]
        for fold, (train, test) in enumerate(splits, start=1):
            started = time.perf_counter()
            metrics = evaluate_mlp(chosen, labels, train, test, fold, epochs=150, patience=15)
            rows.append({"condition": name, "fold": fold,
                         "fourier_dims": block.shape[1],
                         "dimensions": sum(b.shape[1] for b in chosen),
                         "seconds": time.perf_counter() - started,
                         "bacc": 100 * metrics["bacc"]})
            log(f"  {name:<22} f{fold}  {rows[-1]['dimensions']:>4}d  "
                f"BAcc {rows[-1]['bacc']:5.2f}")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    frame = pd.DataFrame(rows)
    summary = frame.groupby("condition").agg(
        fourier_dims=("fourier_dims", "first"), bacc=("bacc", "mean"), sd=("bacc", "std"))
    log("\n" + summary.round(2).to_string())
    base = frame[frame.condition.eq(star_name)].set_index("fold")["bacc"]
    log("\ndelta contra theta* NO PIPELINE (pareado por fold):")
    for name in summary.index:
        if name == star_name:
            continue
        other = frame[frame.condition.eq(name)].set_index("fold")["bacc"]
        shared = base.index.intersection(other.index)
        delta = other[shared] - base[shared]
        _, p = stats.ttest_rel(other[shared], base[shared])
        log(f"  {name:<22} {delta.mean():+6.2f}pp  p={p:.4f}  "
            f"{int((delta > 0).sum())}/{len(delta)}")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
