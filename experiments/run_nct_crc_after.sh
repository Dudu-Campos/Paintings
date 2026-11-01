#!/usr/bin/env bash
# Encadeia o NCT-CRC depois que a corrida de tabelas externas em curso terminar.
#
# Espera pelo marcador de FIM no log, nao por PID: `$!` depois de `setsid` e o
# PID do proprio setsid, que morre em milissegundos, e `pgrep -f` casa o shell
# que o contem (§8, oito ocorrencias). O log e o unico sinal confiavel.
#
# O NCT-CRC e o unico corpus com COORTE DE TESTE SEPARADA (nct_crc_val, 7.180
# tiles de pacientes diferentes). Duas condicoes serao PULADAS com aviso -- ViT
# congelado e Haralick/HOG nunca foram extraidos para essa coorte, e as imagens
# do corpus nao estao mais em disco.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3
LOG=logs/external_final_tables.log

echo "=== [$(date '+%F %T')] aguardando a corrida atual"
until grep -qE '^=== \[.*\] FIM' "$LOG" 2>/dev/null; do sleep 60; done
echo "=== [$(date '+%F %T')] corrida atual terminou; comecando nct_crc"

$PY experiments/evaluate_preprocess_effect_oilpainting.py --dataset nct_crc \
    --heads narrow --seeds 42 43 44 45 46 \
    --output results/nct_crc_final_table.csv || echo "FALHOU: nct_crc"
echo "=== [$(date '+%F %T')] FIM"
