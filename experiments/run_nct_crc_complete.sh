#!/usr/bin/env bash
# Completa a tabela do NCT-CRC depois que a corrida atual (logica antiga de
# `[pulado]`) terminar. O script e retomavel pela chave (head, condition, fold,
# seed), entao so as 3 condicoes que faltavam rodam de novo.
set -u
cd "$(dirname "$0")/.."
until grep -qE '^=== \[.*\] FIM' logs/nct_crc_chained.log 2>/dev/null; do sleep 60; done
echo "=== [$(date '+%F %T')] completando nct_crc"
.ABSvenv/bin/python3 experiments/evaluate_preprocess_effect_oilpainting.py \
    --dataset nct_crc --heads narrow --seeds 42 43 44 45 46 \
    --output results/nct_crc_final_table.csv || echo "FALHOU"
echo "=== [$(date '+%F %T')] FIM"
