"""O que uma CNN treinada em rótulo já aprende das features hand-designed?

Pergunta invertida em relação ao pré-treino. Lá, a rede era **obrigada** a
prever as features e mediu-se o quanto conseguia. Aqui a CNN nunca viu as
features: foi treinada só com os 27 rótulos de estilo. Regredindo linearmente
do embedding dela para cada bloco, o R² diz quanto daquele bloco **emerge
sozinho** da supervisão por rótulo.

As duas medições respondem coisas diferentes e podem discordar:

    R² do pré-treino     quanto a arquitetura CONSEGUE representar quando o
                         bloco está na loss
    R² daqui             quanto ela desenvolve ESPONTANEAMENTE sem ser pedido

Um bloco com R² alto aqui é **redundante como alvo de pré-texto** -- pedir por
ele não acrescenta nada ao que a rede aprenderia de qualquer jeito. Um bloco
com R² baixo é o oposto: é informação que a supervisão por rótulo não produz, e
portanto o alvo mais valioso. É também, pela mesma conta, a parte do pipeline
hand-designed que mais justifica existir ao lado de uma CNN.

Dois cuidados de construção:

  - **Um checkpoint só.** Os cinco folds da CNN são redes treinadas
    independentemente e seus 2.048d vivem em espaços arbitrariamente
    rotacionados, sem correspondência entre si. Misturá-los (como o embedding
    OOF faz) destrói a regressão. Aqui usa-se o checkpoint do fold 1.
  - **Só imagens fora do treino da CNN.** O ajuste e o teste da ridge saem
    ambos do conjunto de TESTE do fold 1 da CNN, dividido ao meio. Assim
    nenhuma linha vem de uma imagem cujo rótulo a CNN memorizou -- o que
    importa menos aqui (os alvos são label-free) mas custa nada garantir.

Comparabilidade com `per_block_fold1.csv`: aqueles R² vêm de uma cabeça de
regressão treinada de ponta a ponta; estes vêm de uma sonda **linear** sobre
features congeladas, que é um limite inferior do que a representação contém.
Comparar as duas colunas em valor absoluto é injusto com esta; o que vale é o
**ordenamento entre blocos** dentro de cada coluna.

Saída: results/cnn_learns_handdesigned.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from final_tfidf_visual_mlp import SEED  # noqa: E402
from pretrain_cnn_on_handdesigned import load_targets  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path,
                        default=ROOT / "outputs/cnn_embedding_branch/embed_fold1.npy")
    parser.add_argument("--embedding-stems", type=Path,
                        help="stems do embedding, quando ele não estiver na ordem "
                             "da coorte (ex.: os braços extraídos por "
                             "extract_cnn_full.py, que usam ordem própria)")
    parser.add_argument("--label", type=str, default="",
                        help="nome da fonte, só para o log e o nome do arquivo")
    parser.add_argument("--cnn-dir", type=Path, default=ROOT / "outputs/cnn_scratch")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--alpha", type=float, nargs="+",
                        default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cnn_learns_handdesigned.csv")
    args = parser.parse_args()

    stems, targets, spans = load_targets()
    embedding = np.load(args.embedding)
    if args.embedding_stems:
        source_stems = np.load(args.embedding_stems, allow_pickle=True).astype(str)
        position = {stem: index for index, stem in enumerate(source_stems)}
        index = np.array([position.get(stem, -1) for stem in stems])
        if (index < 0).any():
            raise SystemExit(f"{int((index < 0).sum()):,} stems da coorte ausentes "
                             f"em {args.embedding_stems.name}")
        embedding = np.ascontiguousarray(embedding[index])
    if len(embedding) != len(stems):
        raise SystemExit(f"embedding {embedding.shape} != coorte {len(stems)}")
    log(f"coorte {len(stems):,} | alvo {targets.shape[1]}d em {len(spans)} blocos")

    predictions = pd.read_csv(args.cnn_dir / f"fold_{args.fold:02d}_predictions.csv")
    held_out = set(predictions["stem"].astype(str))
    mask = np.array([s in held_out for s in stems])
    log(f"imagens fora do treino da CNN (teste do fold {args.fold}): "
        f"{int(mask.sum()):,}")

    values, blocks = embedding[mask], targets[mask]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(values))
    cut = len(order) // 2
    fit, test = order[:cut], order[cut:]
    log(f"ridge: {len(fit):,} ajuste / {len(test):,} teste")

    x_scaler = StandardScaler().fit(values[fit])
    train_x, test_x = x_scaler.transform(values[fit]), x_scaler.transform(values[test])
    y_scaler = StandardScaler().fit(blocks[fit])
    train_y, test_y = y_scaler.transform(blocks[fit]), y_scaler.transform(blocks[test])

    # Um alpha por execução, escolhido pelo R² global no teste -- é otimista,
    # mas idêntico para todos os blocos, então não distorce o ordenamento.
    best = None
    for alpha in args.alpha:
        model = Ridge(alpha=alpha).fit(train_x, train_y)
        predicted = model.predict(test_x)
        score = 1 - ((test_y - predicted) ** 2).mean() / test_y.var()
        log(f"  alpha={alpha:>7.1f}  R² global={score:.4f}")
        if best is None or score > best[0]:
            best = (score, alpha, predicted)
    _, alpha, predicted = best
    log(f"alpha escolhido: {alpha}")

    rows = []
    for name, (start, stop) in spans.items():
        truth = test_y[:, start:stop]
        residual = ((truth - predicted[:, start:stop]) ** 2).mean()
        rows.append({"block": name, "dims": stop - start,
                     "mse": float(residual),
                     "r2": float(1 - residual / truth.var())})
    table = pd.DataFrame(rows).sort_values("r2", ascending=False)
    if args.label:
        table.insert(0, "source", args.label)
        args.output = args.output.with_name(
            f"{args.output.stem}_{args.label.replace(' ', '_')}.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.6f")

    log("\n=== quanto do bloco a CNN treinada em RÓTULO já contém "
        "(sonda linear, fold de teste) ===")
    log(table.round(4).to_string(index=False))

    pretext = ROOT / "outputs/cnn_pretrained_handdesigned/per_block_fold1.csv"
    if pretext.exists():
        other = pd.read_csv(pretext)[["block", "r2"]].rename(
            columns={"r2": "r2_pretexto"})
        merged = table[["block", "dims", "r2"]].merge(other, on="block", how="left")
        merged = merged.rename(columns={"r2": "r2_rotulo"})
        merged["diferenca"] = merged["r2_pretexto"] - merged["r2_rotulo"]
        log("\n=== contra o pré-texto (que foi TREINADO nesses alvos) ===")
        log("diferenca alta = o pré-texto ensinou algo que o rótulo não dá")
        log(merged.round(4).to_string(index=False))
        merged.to_csv(args.output.with_name("cnn_learns_vs_pretext.csv"),
                      index=False, float_format="%.6f")
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
