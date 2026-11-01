#!/usr/bin/env bash
# Enfileira food101 e oilpainting ATRÁS do driver que já roda nct_crc+minc2500.
#
# A absinto tem 1 GPU visível; dois drivers simultâneos saturam a utilização e
# atrasam os dois (medido em 25/ago: 4 processos levaram o fine-tuning de 2,5
# para 2,83 min/época). Esperar é mais rápido que competir.
#
# A espera é por **PID explícito via /proc**, não por pgrep: `pgrep -f` casa com
# o próprio shell que o contém, e isso já custou tempo sete vezes neste projeto
# (§8 do CLAUDE.md).
set -u
cd "$(dirname "$0")/.."
DRIVER_PID=293409
while [ -d /proc/$DRIVER_PID ]; do sleep 300; done
echo "=== [$(date +'%d/%b %H:%M:%S')] driver $DRIVER_PID terminou; iniciando food101 + oilpainting"
bash experiments/run_finetune_grid_resilient.sh --gpus 0 \
  --datasets "food101 oilpainting" \
  --models "resnet50 vit_b_16" \
  --seeds "42 43 44 45 46"
echo "=== [$(date +'%d/%b %H:%M:%S')] FILA_ABSINTO_PRONTA"
