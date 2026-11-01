"""Metadado do NCT-CRC-HE-100K (histologia colorretal) no formato do pipeline.

Sexto corpus, e o primeiro fora de foto natural ou arte. Serve para testar
**fora da amostra** a explicação do §4.2 -- "a vantagem da CNN é pré-treino, e
pré-treino é ligado a domínio" -- que hoje está ajustada a cinco corpora todos
do mesmo tipo de imagem.

Kather et al., 100.000 tiles de 224x224 a 0,5 um/px, com normalização de
coloração (Macenko). **Não usar a variante NONORM**: o nosso banco é fortemente
baseado em cor e mediria variação de lâmina em vez de tipo de tecido.

Nove classes: ADI (tecido adiposo), BACK (fundo), DEB (detrito), LYM
(linfócitos), MUC (muco), MUS (músculo liso), NORM (mucosa normal), STR (estroma
associado a câncer), TUM (epitélio de adenocarcinoma).

Duas decisões, pelas regras do §8:

* **`stem` com prefixo de classe.** Os nomes já são únicos (`ADI-AAAMHQMK`),
  mas prefixar mantém o padrão e imuniza contra os extratores que indexam por
  nome de arquivo.
* **`split` reservado para o teste externo.** O CRC-VAL-HE-7K é de **pacientes
  diferentes** -- é o protocolo publicado e o único que separa "aprendeu tecido"
  de "aprendeu o escâner". Se ele não estiver presente, a coluna sai toda como
  `train` e a avaliação cai em 5-fold, que NÃO controla paciente.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"ADI": "adipose", "BACK": "background", "DEB": "debris",
           "LYM": "lymphocytes", "MUC": "mucus", "MUS": "smooth muscle",
           "NORM": "normal mucosa", "STR": "cancer-associated stroma",
           "TUM": "adenocarcinoma epithelium"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def collect(base: Path, split: str) -> list[dict]:
    rows = []
    for image in sorted(base.glob("*/*.tif")):
        style = image.parent.name
        rows.append({"stem": f"{style}__{image.stem}", "filename": image.name,
                     "style": style, "split": split})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path,
                        default=Path("/tmp/prism_data/nct_crc/NCT-CRC-HE-100K"))
    parser.add_argument("--test-dir", type=Path,
                        default=Path("/tmp/prism_data/nct_crc/CRC-VAL-HE-7K"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/nct_crc_meta.csv")
    args = parser.parse_args()

    rows = collect(args.train_dir, "train")
    log(f"treino: {len(rows):,} tiles")
    if args.test_dir.exists():
        external = collect(args.test_dir, "test")
        rows += external
        log(f"teste externo (pacientes diferentes): {len(external):,} tiles")
    else:
        log("AVISO: CRC-VAL-HE-7K ausente -- sem teste externo, e 5-fold NÃO "
            "controla paciente")

    frame = pd.DataFrame(rows)
    if frame["stem"].duplicated().any():
        raise ValueError(f"{int(frame['stem'].duplicated().sum())} stems duplicados")
    frame.to_csv(args.output, index=False)
    counts = frame[frame.split.eq("train")]["style"].value_counts()
    log(f"salvo: {args.output} | {len(frame):,} linhas, {frame['style'].nunique()} classes")
    for name, n in counts.items():
        log(f"  {name:<5} {CLASSES.get(name, '?'):<28} {n:>7,}")
    log(f"desbalanceamento no treino: {counts.max() / counts.min():.2f}x "
        f"(acurácia trivial {100 * counts.max() / counts.sum():.2f}%)")


if __name__ == "__main__":
    main()
