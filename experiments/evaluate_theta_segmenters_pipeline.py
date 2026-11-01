"""O theta dos segmentadores transfere para o hand-8?

Quarto e quinto testes de transferencia da serie (Hilbert §5.17, Gabor §5.17b,
Fourier §5.17c). Aqui a pergunta ganha um alvo especifico que os tres anteriores
nao tinham, porque a varredura mediu um CKA de **0,111** no SLIC a 256 px com a
contagem de segmentos preservada: a representacao vira quase outra coisa e a
BAcc isolada nao muda (26,29 contra 26,02 de theta*).

Se isso tambem nao mover o pipeline, o §5.17 ganha o caso extremo -- ate uma
representacao que mede algo 89% diferente e intercambiavel dentro da fusao.

Condicoes, tres por segmentador (o desenho da varredura foi em CRUZ, entao nao
ha grade cheia para varrer):

  SLIC   (1024,2048) theta*   (1024,4096) melhor isolada   (256,2048) CKA 0,111
  Felz   (1024,  50) theta*   (1024, 100) vizinha          (256,  50) pior isolada

O bloco do sweep tem as MESMAS 71 colunas do bloco de producao (conferido); a
troca reordena para a ordem do registry antes de concatenar, senao o MLP receberia
as features permutadas em silencio.
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
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import VISUAL, clean  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_mlp  # noqa: E402
from evaluate_external_modalities import load_block  # noqa: E402
from evaluate_fusion_delta_across_corpora import load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from protocol import SEED  # noqa: E402

PLANS = {
    # Grade CHEIA (as 6 extracoes que existem em disco por view), nao a cruz de
    # 3 do §5.17e. A robustez de theta e afirmacao sobre DISPERSAO, e um desvio
    # sobre 3 pontos nao e distribuicao -- ver a nota de 10/set sobre interpolar.
    "slic": ("SLIC", (1024, 2048),
             [(1024, 512), (1024, 1024), (1024, 2048), (1024, 4096),
              (256, 2048), (512, 2048)]),
    "felzenszwalb": ("Felzenszwalb", (1024, 50),
                     [(1024, 25), (1024, 50), (1024, 100), (1024, 200),
                      (256, 50), (512, 50)]),
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=list(PLANS), required=True)
    ap.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    ap.add_argument("--classes-file", type=Path,
                    default=ROOT / "data/multitask_25class_benchmark.txt")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_segmenters")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--isolated", action="store_true",
                      help="mede a view SOZINHA, mesmo laco de treino")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    block_name, star, grid = PLANS[args.method]
    if args.output is None:
        args.output = ROOT / f"results/theta_{args.method}_pipeline.csv"

    blocks, raw_labels, _ = load_corpus("multitask", VISUAL + ["Gabor"])
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    mask = np.array([key(v) in wanted for v in raw_labels])
    if len(np.unique(raw_labels[mask])) != len(wanted):
        raise ValueError("casamento de classes incompleto")
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels[mask])
    order = [n for n in VISUAL + ["Gabor"] if n in blocks]
    log(f"hand-8: {len(labels):,} imagens, {len(np.unique(labels))} classes, trocando {block_name}")

    # A ORDEM das colunas tem de ser a do registry, senao a troca permuta as
    # features em silencio e o MLP recebe outro bloco sem nenhum erro.
    reference_cols = [c[len(block_name) + 1:] for c in
                      load_block("multitask", block_name).columns if c != "stem"]
    cohort = pd.read_csv(args.meta, dtype={"stem": str})
    cohort = cohort[[key(str(v)) in wanted for v in cohort["style"]]]
    stems = cohort.stem.to_numpy()
    if len(stems) != len(labels):
        raise ValueError(f"coorte {len(labels)} != metadado {len(stems)}")

    def swap(max_side: int, gran: int) -> np.ndarray:
        d = args.work_dir / f"{args.method}_s{max_side}_g{gran}"
        f = pd.read_parquet(d / "combined_features.parquet")
        f["stem"] = f["stem"].astype(str)
        f = f.drop_duplicates("stem").set_index("stem").reindex(stems)
        if f[reference_cols].isna().all(axis=1).any():
            raise ValueError(f"({max_side},{gran}): stems ausentes")
        return clean(f[reference_cols].to_numpy(np.float64))

    conditions = {f"{args.method} {a}x{b}": swap(a, b) for a, b in grid}
    star_name = f"{args.method} {star[0]}x{star[1]}"
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
        chosen = [block] if args.isolated else [blocks[n] if n != block_name else block for n in order]
        for fold, (train, test) in enumerate(splits, start=1):
            m = evaluate_mlp(chosen, labels, train, test, fold, epochs=150, patience=15)
            rows.append({"condition": name, "fold": fold,
                         "dimensions": sum(b.shape[1] for b in chosen),
                         "bacc": 100 * m["bacc"]})
            log(f"  {name:<24} f{fold}  {rows[-1]['dimensions']:>4}d  BAcc {rows[-1]['bacc']:5.2f}")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    frame = pd.DataFrame(rows)
    summary = frame.groupby("condition").bacc.agg(["mean", "std"])
    log("\n" + summary.round(2).to_string())
    base = frame[frame.condition.eq(star_name)].set_index("fold")["bacc"]
    log("\ndelta contra theta* NO PIPELINE (pareado por fold):")
    for name in summary.index:
        if name == star_name:
            continue
        o = frame[frame.condition.eq(name)].set_index("fold")["bacc"]
        d = o - base
        _, p = stats.ttest_rel(o, base)
        log(f"  {name:<24} {d.mean():+6.2f}pp  p={p:.4f}  {int((d > 0).sum())}/{len(d)}")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
