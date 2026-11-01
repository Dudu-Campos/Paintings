#!/usr/bin/env bash
# Legendas que faltam: coorte de TESTE do nct_crc, places_lt (view reconstruida
# em 08/set) e minc2500 (a fila daquele dia pegou um cache parcial de smoke).
#
# EM SERIE por sequencia no script, sem esperar por processo. O §8 registra oito
# casos de `pgrep -f` casando com o proprio wrapper; o nono foi aqui, e o filtro
# por `python3` NAO salvou -- o corpo do `bash -c` continha a linha inteira,
# interpretador incluso, entao os dois encadeamentos esperaram por si mesmos.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3
C=experiments/evaluate_student_captions_multitask.py

echo "=== [$(date '+%T')] nct_crc_val (coorte de TESTE externo)"
$PY $C --dataset nct_crc_val --meta data/nct_crc_val_meta.csv \
       --image-dir /tmp/prism_data/nct_crc || echo "FALHOU: nct_crc_val"

echo "=== [$(date '+%T')] places_lt"
$PY $C --dataset places_lt --meta data/places_lt_meta.csv \
       --image-dir data/places_lt/images || echo "FALHOU: places_lt"

echo "=== [$(date '+%T')] minc2500"
$PY $C --dataset minc2500 --meta data/minc2500_meta.csv \
       --image-dir /tmp/prism_data/minc/minc-2500/images || echo "FALHOU: minc2500"

echo "=== [$(date '+%T')] TUDO PRONTO"
