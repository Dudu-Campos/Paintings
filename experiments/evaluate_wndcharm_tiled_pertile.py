"""WND-CHARM tiled, protocolo FIEL: classifica CADA tile e so DEPOIS agrega.

Corrige `evaluate_wndcharm_tiled_vs_hand8.py`, cujo "pooled" faz media das
FEATURES dos 16 tiles ANTES de classificar -- isso reaproxima o metodo de um
descritor global, exatamente o erro que a extracao tiled existia pra corrigir
(ver o docstring de `extract_wndcharm_tiled_features.py`, citando a taxonomia
Global/Local do §7.11b). Achado por revisao externa em 29/set.

O artigo (Shamir et al. 2010): "features are computed on each of the 16 tiles
... averaging the 16 similarity vectors of the tiles, providing a single
similarity vector". A estrutura certa e CLASSIFICAR por tile e agregar a
DECISAO, nao agregar a FEATURE.

**Ainda nao e o WND5 exato** (mesma razao do script anterior: a formula de
agregacao por classe do WND5 nao esta determinada com certeza suficiente pra
reivindicar reproducao, §7.11b). Usa o MESMO MLP de producao, mas agora
respeitando a ESTRUTURA do protocolo:

  1. treina em linhas POR TILE (16x mais linhas que por imagem -- cada tile
     de uma imagem de treino e um exemplo de treino independente, rotulo da
     imagem-mae, exatamente como o artigo descreve: "providing 16 feature
     vectors" por imagem, tratados como exemplos);
  2. no teste, calcula a probabilidade por CLASSE em cada um dos 16 tiles da
     imagem (`train_one(..., return_probs=True)`, extensao aditiva de
     `models.py` -- nao muda nenhum chamador existente, ver o docstring la);
  3. MEDIA as 16 distribuicoes de probabilidade por imagem -- e a operacao
     literal de "averaging the ... similarity vectors, providing a single
     similarity vector" -- e so ENTAO faz argmax.

Fit/validacao seguem exatamente a mesma particao por IMAGEM do
`evaluate_mlp` (10% do treino externo, estratificado, semente SEED+fold) --
os 16 tiles de uma imagem NUNCA atravessam a fronteira fit/validacao/teste,
senao um tile de validacao vazaria a imagem-irma de treino.

Uso:
    python experiments/evaluate_wndcharm_tiled_pertile.py [--folds 5]
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import HAND8, clean                     # noqa: E402
from models import train_one                          # noqa: E402
from protocol import SEED, prediction_metrics          # noqa: E402
from registry import load_blocks                       # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def expand_tiles(tiles: np.ndarray, labels: np.ndarray, image_idx: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(k imagens, 16, D) -> (k*16, D) linhas; rotulo repetido 16x cada.

    `reshape` em ordem C alinha [img0_tile0..15, img1_tile0..15, ...], que
    casa exatamente com `np.repeat(labels, 16)`.
    """
    block = tiles[image_idx]
    flat = block.reshape(-1, block.shape[-1])
    expanded_labels = np.repeat(labels[image_idx], 16)
    return flat, expanded_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--tiled-dir", type=Path,
                        default=ROOT / "outputs" / "multitask_wndcharm_tiled")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "wndcharm_tiled_pertile.csv")
    args = parser.parse_args()

    log(f"coorte de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset, list(HAND8))
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    stems = [str(s) for s in cohort.stems]
    log(f"  {len(stems)} imagens, {n_classes} classes")

    tiled_stems = [str(s) for s in np.load(args.tiled_dir / "stems.npy")]
    if tiled_stems != stems:
        raise SystemExit("stems.npy nao reproduz cohort.stems -- abortando (§5.18b)")

    log("carregando features.npy (n, 16, 2895) EM MEMORIA (sem pooling desta vez)")
    started = time.time()
    tiles = np.load(args.tiled_dir / "features.npy")
    tiles = clean(tiles.reshape(-1, tiles.shape[-1])).reshape(tiles.shape)
    log(f"  carregado + limpo em {time.time()-started:.0f}s, shape {tiles.shape}")

    flat_all = tiles.reshape(-1, tiles.shape[-1])
    alive = flat_all.std(axis=0) > 0
    del flat_all
    log(f"  colunas constantes (sobre TODAS as linhas de tile): {int((~alive).sum())} de {tiles.shape[-1]}")
    tiles = tiles[:, :, alive]
    dims = tiles.shape[-1]

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]

    rows: list[dict] = []
    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        fit_img, val_img = train_test_split(
            outer_train, test_size=0.1, stratify=labels[outer_train],
            random_state=SEED + fold)

        train_x, train_y = expand_tiles(tiles, labels, fit_img)
        val_x, val_y = expand_tiles(tiles, labels, val_img)
        test_x_raw, test_y_tile = expand_tiles(tiles, labels, outer_test)

        scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x).astype(np.float32, copy=False)
        val_x = scaler.transform(val_x).astype(np.float32, copy=False)
        test_x = scaler.transform(test_x_raw).astype(np.float32, copy=False)
        del test_x_raw

        log(f"  fold {fold}: treino {train_x.shape[0]} linhas de tile "
            f"({len(fit_img)} img), val {val_x.shape[0]}, teste {test_x.shape[0]} "
            f"({len(outer_test)} img)")

        started = time.perf_counter()
        metrics, _ = train_one(
            train_x, train_y, val_x, val_y, test_x, test_y_tile,
            [dims], n_classes, SEED + fold, args.epochs, args.patience,
            return_probs=True,
        )
        tile_probs = metrics["test_probs"]                    # (n_test*16, C)
        image_probs = tile_probs.reshape(len(outer_test), 16, n_classes).mean(axis=1)
        image_pred = image_probs.argmax(axis=1)
        final = prediction_metrics(labels[outer_test], image_pred)

        elapsed = time.perf_counter() - started
        rows.append({
            "fold": fold, "dimensions": dims,
            "n_train_tiles": train_x.shape[0], "n_test_images": len(outer_test),
            "best_epoch": metrics["best_epoch"], "validation_bacc": metrics["validation_bacc"],
            "seconds": elapsed,
            "bacc": 100 * final["bacc"], "accuracy": 100 * final["accuracy"],
            "macro_f1": 100 * final["macro_f1"],
            # tile-level bacc (SEM agregacao) como diagnostico: mostra quanto
            # a media de 16 decisoes recupera sobre decidir tile a tile.
            "tile_level_bacc": 100 * metrics["bacc"],
        })
        log(f"  fold {fold}  BAcc por-imagem {rows[-1]['bacc']:5.2f}  "
            f"(tile a tile isolado: {rows[-1]['tile_level_bacc']:5.2f})  "
            f"epoca {metrics['best_epoch']}  ({elapsed:.0f}s)")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows)
    log("\n" + table[["fold", "bacc", "accuracy", "macro_f1", "tile_level_bacc",
                      "best_epoch", "seconds"]].round(2).to_string(index=False))
    log(f"\nBAcc media (agregado por imagem): {table['bacc'].mean():.2f} +- {table['bacc'].std():.2f}")
    log(f"BAcc media (tile isolado, sem agregar): {table['tile_level_bacc'].mean():.2f} +- "
        f"{table['tile_level_bacc'].std():.2f}")
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
