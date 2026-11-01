"""Metadado do Food-101 no formato que os extratores esperam.

O tar já vem no layout certo (`images/<classe>/<id>.jpg`), então aqui só se
constrói o CSV. Duas decisões que importam:

* **`stem` com prefixo de classe.** Os ids do Food-101 são numéricos e únicos
  no dataset, mas prefixar mantém o padrão dos outros corpora e imuniza contra
  o modo de falha que custou uma reextração no OilPainting -- extratores que
  indexam por nome de arquivo perdem imagens em silêncio quando há colisão.
* **A coluna `split` preserva a partição oficial** (750 treino / 250 teste por
  classe, dos arquivos `meta/train.txt` e `meta/test.txt`). É o protocolo que a
  literatura usa e o único caminho para número comparável; a validação cruzada
  de 5 folds do projeto fica como análise paralela, não como substituta.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "data/food101/food-101")
    parser.add_argument("--output", type=Path, default=ROOT / "data/food101_meta.csv")
    args = parser.parse_args()

    split_of = {}
    for split in ("train", "test"):
        path = args.base / "meta" / f"{split}.txt"
        for line in path.read_text().splitlines():
            if line.strip():
                split_of[line.strip()] = split
    log(f"partição oficial: {sum(v == 'train' for v in split_of.values()):,} treino, "
        f"{sum(v == 'test' for v in split_of.values()):,} teste")

    rows = []
    for image in sorted((args.base / "images").glob("*/*.jpg")):
        style = image.parent.name
        key = f"{style}/{image.stem}"
        rows.append({"stem": f"{style}__{image.stem}", "filename": image.name,
                     "style": style, "split": split_of.get(key, "unknown")})
    frame = pd.DataFrame(rows)
    if frame["stem"].duplicated().any():
        raise ValueError(f"{int(frame['stem'].duplicated().sum())} stems duplicados")
    unknown = int(frame["split"].eq("unknown").sum())
    if unknown:
        log(f"AVISO: {unknown:,} imagens fora da partição oficial")
    frame.to_csv(args.output, index=False)
    counts = frame["style"].value_counts()
    log(f"salvo: {args.output} | {len(frame):,} imagens, {frame['style'].nunique()} classes")
    log(f"por classe: min {counts.min()}, max {counts.max()}")


if __name__ == "__main__":
    main()
