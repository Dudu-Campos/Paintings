"""WND-CHARM contra o banco hand-8, e a pergunta que importa: os dois fundem?

Multitask 125 classes, coorte de 99.816, `BranchMLP` estreito de producao,
5 folds, `StratifiedKFold(5, shuffle=True, random_state=SEED)`,
`evaluate_mlp(..., epochs=150, patience=15)` -- o mesmo caminho do
`evaluate_theta_gabor_pipeline.py`, que o §5.17b registra reproduzindo o hand-8
ao centesimo. Nada de laco de treino reimplementado (§5.6: custou 0,47pp).

**A pergunta NAO e quem ganha.** O §5.14 mede que o que domina e a diversidade
de familias e nao o critério de escolha dentro delas -- o braco ALEATORIO do
banco ja da +6,5pp sobre Descriptors com dimensionalidade pareada, e o critério
compra so +1,24pp em cima. O WND-CHARM tem 15 familias de algoritmo contra as
nossas 8 views, sobre uma hierarquia de 12 cadeias de transformada. Entao o
§5.14 **preve que ele seja competitivo ou melhor** no eixo hand-crafted isolado.
Previsao registrada antes de rodar.

O que decide algo e a FUSAO. Dois bancos hand-crafted projetados de forma
independente, sobre as mesmas imagens, e o teste mais limpo possivel da
alegacao de cobertura de familias -- e remove a objecao de que escolhemos as
nossas oito para serem complementares.

    fundem folgado  -> a tese multi-view ganha evidencia de um desenho de terceiros
    sao substitutos -> o espaco de familias hand-crafted esta SATURADO, que e
                       limite medido e vale publicacao igual

DOIS CONFUNDIDORES QUE O DESENHO TEM DE CONTROLAR, senao nada e atribuivel:

1. **COR.** O WND-CHARM e monocromatico -- zero features de cor nas 2.895. Do
   nosso lado, 99 das 130 colunas do `Descriptors` sao cor (`lab_*`: 90 de
   histograma + 9 de estatisticas, §10.2). O braco `hand-8 sem cor` remove essas
   99. **O controle e PARCIAL e isto tem de ser declarado**: as quatro views de
   grafo pesam aresta por `color_dist` (§9.3) e isso nao e removivel por coluna.
   Sem o braco, um ganho nosso se le como "mais familias" quando pode ser "tem
   cor".

2. **ARQUITETURA.** O `BranchMLP` manda TODO bloco para 128 dims, entao o
   hand-8 recebe 8 x 128 = 1024 de largura e o WND-CHARM como bloco unico
   recebe 128 -- compressao de 22,6x contra 4,8x, e handicap estrutural, nao
   resultado. O §5.13/§5.16 medem que a primeira camada bloco-diagonal vale
   +0,93pp sobre um denso de mesma largura, logo a alocacao importa. Os dois
   bracos (1 bloco e 15 blocos por algoritmo) vao na tabela.

Guarda de caminho: o `hand-8` tem de cair em 38,5-38,8. O §9.7 registra que ele
reproduz em **38,53 / 38,71 / 38,74** entre execucoes (nao-determinismo de GPU),
e os deltas aqui sao dessa ordem -- entao a banda e a do proprio projeto.

Uso:
    python experiments/evaluate_wndcharm_vs_hand8.py \
        --features results/wndcharm_multitask_features.parquet
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import HAND8, clean                     # noqa: E402
from protocol import SEED, evaluate_mlp              # noqa: E402
from registry import load_blocks                     # noqa: E402

BANDA_HAND8 = (38.3, 39.0)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def algorithm_of(raw: str) -> str:
    """`Haralick Textures (Fourier (Wavelet ())) [7]` -> `Haralick Textures`."""
    match = re.match(r"^(.*?)\s*\(", raw)
    return match.group(1) if match else raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--features", type=Path,
                        default=ROOT / "results" / "wndcharm_multitask_features.parquet")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "wndcharm_vs_hand8.csv")
    args = parser.parse_args()

    log(f"coorte de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset, list(HAND8))
    # `cohort.labels` vem como STRING (nome do estilo) e o `train_one` faz
    # `np.bincount` para o `class_weight`. A convencao do projeto e codificar
    # (`evaluate_logit_add_multitask.py:246`, `evaluate_theta_gabor_pipeline.py:88`).
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    stems = pd.Index([str(s) for s in cohort.stems])
    log(f"  {len(stems)} imagens, {cohort.dimensions}d, "
        f"{len(np.unique(labels))} classes, blocos {list(cohort.blocks)}")

    # `dtype={"stem": str}` e obrigatorio: o pandas infere stem numerico como
    # int64 e o merge contra str quebra (§8).
    log(f"lendo {args.features}")
    frame = pd.read_parquet(args.features)
    frame["stem"] = frame["stem"].astype(str)
    index = pd.Series(np.arange(len(frame)), index=frame["stem"]).reindex(stems)
    if index.isna().any():
        raise SystemExit(
            f"{int(index.isna().sum())} stems da coorte ausentes no parquet -- "
            f"reindexar com buraco daria bloco desalinhado em silencio (§5.18b)")
    columns = [c for c in frame.columns if c != "stem"]
    charm = clean(frame[columns].to_numpy(dtype=np.float32)[index.to_numpy(dtype=int)])
    log(f"  {charm.shape[1]} features, reindexadas para a ordem de cohort.stems")

    # Coluna constante na coorte nao e so peso morto: o §5.17c mede que ela
    # machuca 231% MAIS no pipeline que sozinha, porque o ramo gasta capacidade
    # de encoder em entrada que nao varia. Remover aqui e o analogo do
    # DEAD_COLUMNS do registry, e a contagem e informativa por si.
    alive = charm.std(axis=0) > 0
    log(f"  colunas constantes na coorte: {int((~alive).sum())} de {charm.shape[1]}")
    raw_names = json.loads(args.features.with_suffix(".names.json").read_text())["raw_names"]
    raw_alive = [n for n, keep in zip(raw_names, alive) if keep]
    charm = charm[:, alive]

    groups: dict[str, list[int]] = {}
    for position, raw in enumerate(raw_alive):
        groups.setdefault(algorithm_of(raw), []).append(position)
    log(f"  {len(groups)} familias de algoritmo: "
        + ", ".join(f"{k} ({len(v)})" for k, v in
                    sorted(groups.items(), key=lambda kv: -len(kv[1]))))

    hand8 = [cohort.blocks[name] for name in HAND8]
    charm15 = [charm[:, positions] for positions in groups.values()]

    # O RECORTE DE COR E POSICIONAL, e tem de ser: o `Descriptors` vem de `.npy`,
    # entao o registry sinteriza nomes genericos (`Descriptors_0`..`_129`) e nao
    # ha o que filtrar por nome. O layout esta no §10.2 e no
    # `extract_descriptors_full.compact_feature_names()`:
    #     0-89   histograma Lab (30 bins x 3 canais)   COR
    #     90-98  estatisticas Lab (mean/std/skew x 3)  COR
    #     99-116 LBP uniforme P=16 R=2                 nao-cor
    #     117-129 Haralick via GLCM                    nao-cor
    #
    # **Guarda que confere a posicao por dentro, em vez de confiar no comentario.**
    # O LBP e `np.histogram(..., bins=18, range=(0,18), density=True)`: largura de
    # bin exatamente 1, logo as 18 colunas somam 1,0 em TODA imagem. Nenhuma outra
    # fatia do bloco tem essa propriedade, entao ela localiza o recorte. O §10.2
    # inferiu estes indices do codigo e confirmou por outro caminho (a fatia
    # interna de Haralick bate o bloco extraido a parte em 0,03pp); aqui a
    # conferencia e local e barata.
    descriptors = cohort.blocks["Descriptors"]
    COR = slice(0, 99)
    LBP = slice(99, 117)
    soma_lbp = descriptors[:, LBP].sum(axis=1)
    if not np.allclose(soma_lbp, 1.0, atol=1e-3):
        raise SystemExit(
            f"as colunas 99-116 do Descriptors nao somam 1,0 "
            f"(min {soma_lbp.min():.4f}, max {soma_lbp.max():.4f}) -- o layout do "
            f"§10.2 nao vale neste parquet e o recorte de cor estaria errado")
    log(f"  layout do Descriptors conferido: colunas 99-116 somam 1,0 "
        f"(LBP density), logo cor = 0-98")

    sem_cor = np.hstack([descriptors[:, LBP], descriptors[:, 117:]])
    cor = descriptors[:, COR]
    log(f"  Descriptors: {cor.shape[1]} colunas de cor, {sem_cor.shape[1]} sem cor")
    hand8_sem_cor = [sem_cor if name == "Descriptors" else cohort.blocks[name]
                     for name in HAND8]

    conditions = {
        "hand-8": hand8,
        "hand-8 sem cor": hand8_sem_cor,
        "WND-CHARM (1 bloco)": [charm],
        "WND-CHARM (15 blocos)": charm15,
        "WND-CHARM (15) + cor do Descriptors": charm15 + [cor],
        "hand-8 + WND-CHARM (15)": hand8 + charm15,
    }

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    rows: list[dict] = []
    for name, blocks in conditions.items():
        for fold, (train, test) in enumerate(splits, start=1):
            started = time.perf_counter()
            metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                   epochs=150, patience=15)
            rows.append({
                "condition": name, "fold": fold,
                "n_blocks": len(blocks),
                "dimensions": sum(b.shape[1] for b in blocks),
                "seconds": time.perf_counter() - started,
                "bacc": 100 * metrics["bacc"],
                "accuracy": 100 * metrics["accuracy"],
                "macro_f1": 100 * metrics["macro_f1"],
            })
            log(f"  {name:<36} f{fold}  {rows[-1]['n_blocks']:>2}b "
                f"{rows[-1]['dimensions']:>5}d  BAcc {rows[-1]['bacc']:5.2f}")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows)
    summary = table.groupby("condition", sort=False).agg(
        blocos=("n_blocks", "first"), dims=("dimensions", "first"),
        bacc=("bacc", "mean"), sd=("bacc", "std"),
        acc=("accuracy", "mean"), f1=("macro_f1", "mean"),
        s=("seconds", "mean"))
    log("\n" + summary.round(2).to_string())

    base = table[table.condition.eq("hand-8")].set_index("fold")["bacc"]
    if not BANDA_HAND8[0] <= base.mean() <= BANDA_HAND8[1]:
        log(f"\n!! hand-8 em {base.mean():.2f}, fora da banda {BANDA_HAND8} do §9.7 -- "
            f"o caminho NAO e o de producao; nao reportar os deltas")
    else:
        log(f"\nhand-8 em {base.mean():.2f}: dentro da banda do §9.7, caminho validado")

    log("\ndelta contra hand-8, pareado por fold:")
    for name in conditions:
        if name == "hand-8":
            continue
        other = table[table.condition.eq(name)].set_index("fold")["bacc"]
        delta = other - base
        p = stats.ttest_rel(other, base).pvalue if len(delta) > 1 else np.nan
        log(f"  {name:<36} {delta.mean():+6.2f} +- {delta.std():4.2f}  "
            f"p={p:.4f}  {int((delta > 0).sum())}/{len(delta)}")

    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
