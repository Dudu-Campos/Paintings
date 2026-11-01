#!/usr/bin/env bash
# Tabelas finais dos QUATRO corpora externos, no split PUBLICADO de cada um.
#
# Refeito em 05/set depois de um bug: `OFFICIAL` estava definido e nunca usado,
# entao a versao anterior rodava 5-fold CV sobre a coorte inteira e ignorava o
# split. Para o Places-LT isso e o erro que o §7.5 proibe (996x -> 41x de
# desbalanceamento); para o NCT-CRC apaga a separacao por paciente.
#
# Em corpus de split oficial a SEMENTE e a unidade de replicacao -- nao ha
# folds. Sao 5 ajustes por condicao, nao 25.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3
for ds in minc2500 nct_crc food101 places_lt; do
  echo "=== [$(date '+%F %T')] $ds"
  $PY experiments/evaluate_preprocess_effect_oilpainting.py --dataset "$ds" \
      --heads narrow --seeds 42 43 44 45 46 \
      --output "results/${ds}_final_table.csv" || echo "FALHOU: $ds"
done
echo "=== [$(date '+%F %T')] FIM"
