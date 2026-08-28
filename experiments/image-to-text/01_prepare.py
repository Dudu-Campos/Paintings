"""Etapa 01 — preparação básica dos dados.

Varre o COCO (train/val 2017), resolve o caminho de cada imagem e monta a
tabela mestre consultada por todas as etapas seguintes. Nenhum modelo é
carregado aqui — só I/O de arquivos e organização de splits.

Entrada:
    data/coco/                     imagens COCO (cada servidor baixa a sua)

Saída:
    artifacts/image_index.csv      1 linha por imagem, colunas:
        image_id, file_name, path, split (train/val/test), width, height

Uso:
    python 01_prepare.py --coco-root data/coco --limit 30000 --seed 42
"""

from __future__ import annotations

import argparse

# from config import IMAGE_INDEX_CSV, COCO_ROOT, OUT_DIR, SEED, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--coco-root", type=Path, default=COCO_ROOT)
    # parser.add_argument("--out", type=Path, default=IMAGE_INDEX_CSV)
    # parser.add_argument("--limit", type=int, default=0, help="0 = todas")
    # parser.add_argument("--seed", type=int, default=SEED)
    # parser.add_argument("--val-frac", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()  # noqa: F841
    # TODO: localizar train2017/ e val2017/ (e anotações, se necessárias)
    # TODO: (opcional) amostrar --limit imagens de forma determinística
    # TODO: derivar split train/val/test reprodutível (seed)
    # TODO: ler width/height (PIL) para logs de sanidade
    # TODO: escrever image_index.csv em OUT_DIR (mkdir se preciso)
    # TODO: asserts — nenhum path faltando, sem image_id duplicado
    raise NotImplementedError("scaffold — etapa 01 ainda não implementada")


if __name__ == "__main__":
    main()
