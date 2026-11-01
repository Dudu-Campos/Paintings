"""O ganho de theta do Gabor sobrevive à concatenação com as outras 7 views?

Réplica do teste que fechou o §5.17 no Hilbert, agora na segunda representação
varrida -- e a escolhida justamente por ser o caso OPOSTO em duas propriedades.

**Por que o Gabor testa a afirmação e o Hilbert sozinho não testava.** Lá o
argumento tinha uma saída fácil: o CKA contra theta* nunca cai abaixo de 0,958,
então dava para dizer que a vizinhança é lisa porque *a transformação é
invariante*, e que a fusão não tem mérito nenhum. No Gabor o CKA cai a **0,676**
e decai monotonicamente nos dois sentidos a partir de 128 px -- as frequências
estão em ciclos por pixel, então cada configuração mede uma escala física
diferente. Se a vizinhança de theta* continuar lisa NO PIPELINE mesmo aqui, a
explicação por invariância morre e sobra a explicação por fusão.

**Previsão registrada antes de rodar** (sweep_theta_gabor.py, 06/set):
isolado, `(512,8)` bate theta* por **+2,86pp** sob LDA, e a subida é monotônica
nos dois eixos. A previsão é que **nada disso transfira** -- delta dentro do
ruído ou negativo --, replicando o `(256,32)` do Hilbert, que ganhava +1,40pp
isolado e perdia 0,67pp concatenado.

Três braços de theta e um controle:
  `(512,8)`  o melhor isolado, e o mais caro (16,5x theta*).
  `(256,8)`  ponto intermediário, para ver se o transporte é gradual ou nulo.
  `(64,8)`   a ÚNICA configuração que domina theta* (melhor E mais barata):
             +0,34pp isolado a 97% do custo. É o candidato real a mudar a
             produção, porque não cobra nada.
  `(512,8) -> 32d`  controle de dimensionalidade. O melhor isolado tem 64d
             contra 32d de theta*, então metade do "ganho" pode ser capacidade.
             Subamostra ALEATÓRIA, não as melhores colunas: selecionar por
             rótulo reintroduziria o viés circular do §5.15.
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

STAR = (128, 4)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_gabor")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--thetas", nargs="+",
                        default=["128x4", "512x8", "256x8", "64x8"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--isolated", action="store_true",
                          help="mede a view SOZINHA, mesmo laco de treino")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/theta_gabor_pipeline.csv")
    args = parser.parse_args()

    blocks, raw_labels, _ = load_corpus("multitask", VISUAL + ["Gabor"])
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    mask = np.array([key(v) in wanted for v in raw_labels])
    if len(np.unique(raw_labels[mask])) != len(wanted):
        raise ValueError("casamento de classes do benchmark incompleto")
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels[mask])
    n_classes = len(np.unique(labels))
    order = [n for n in VISUAL + ["Gabor"] if n in blocks]
    if "Gabor" not in order:
        raise SystemExit("o bloco Gabor não está na coorte -- nada a trocar")
    log(f"hand-8: {len(labels):,} imagens, {n_classes} classes, blocos={order}")

    # A coorte do registro define a ordem das linhas; os stems de cada extração
    # de theta são realinhados a ela, e não o contrário.
    cohort = pd.read_csv(args.meta, dtype={"stem": str})
    cohort = cohort[[key(str(v)) in wanted for v in cohort["style"]]]
    reference = cohort.stem.to_numpy()
    if len(reference) != len(labels):
        raise ValueError(f"coorte do registro ({len(labels)}) != metadado ({len(reference)})")

    def gabor_block(target: int, n_thetas: int, columns: int | None) -> np.ndarray:
        directory = args.work_dir / f"t{target}_n{n_thetas}"
        values = np.load(directory / "gabor_features.npy")
        stems = np.load(directory / "gabor_stems.npy", allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        if index.isna().any():
            raise ValueError(f"({target},{n_thetas}): {int(index.isna().sum())} stems ausentes")
        matrix = clean(values[index.to_numpy(dtype=int)])
        if columns is not None and columns < matrix.shape[1]:
            picked = np.random.default_rng(SEED).choice(matrix.shape[1], columns, replace=False)
            matrix = matrix[:, np.sort(picked)]
        return matrix

    star_name = f"gabor {STAR[0]}x{STAR[1]}"
    conditions = {}
    for spec in args.thetas:
        target, n_thetas = (int(v) for v in spec.split("x"))
        conditions[f"gabor {target}x{n_thetas}"] = gabor_block(target, n_thetas, None)
    if star_name not in conditions:
        raise SystemExit(f"theta* ({star_name}) tem de estar na lista: é a referência")
    star_dims = conditions[star_name].shape[1]
    conditions[f"gabor 512x8 -> {star_dims}d"] = gabor_block(512, 8, star_dims)

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
        chosen = [block] if args.isolated else [blocks[n] if n != "Gabor" else block for n in order]
        for fold, (train, test) in enumerate(splits, start=1):
            started = time.perf_counter()
            metrics = evaluate_mlp(chosen, labels, train, test, fold, epochs=150, patience=15)
            rows.append({"condition": name, "fold": fold,
                         "gabor_dims": block.shape[1],
                         "dimensions": sum(b.shape[1] for b in chosen),
                         "seconds": time.perf_counter() - started,
                         "bacc": 100 * metrics["bacc"]})
            log(f"  {name:<22} f{fold}  {rows[-1]['dimensions']:>4}d  "
                f"BAcc {rows[-1]['bacc']:5.2f}")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    frame = pd.DataFrame(rows)
    summary = frame.groupby("condition").agg(
        gabor_dims=("gabor_dims", "first"), bacc=("bacc", "mean"), sd=("bacc", "std"))
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
