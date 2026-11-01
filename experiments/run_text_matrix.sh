#!/usr/bin/env bash
# 1) canal de texto do estudante para o OilPainting (nunca gerado)
# 2) hand-8 + logit-add + texto no Multitask, que fecha a cadeia de producao
#    (§9.7 registra que os dois nunca foram medidos juntos)
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3

echo "=== [$(date '+%F %T')] legendas do estudante: oilpainting"
$PY experiments/evaluate_student_captions_multitask.py \
    --dataset oilpainting \
    --meta data/oilpainting_meta.csv \
    --image-dir data/oilpainting/image || { echo "FALHOU: legendas"; exit 1; }

echo "=== [$(date '+%F %T')] FIM etapa 1"
