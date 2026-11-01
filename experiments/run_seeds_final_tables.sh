#!/usr/bin/env bash
# Cinco sementes nas condicoes de FEATURES dos tres corpora de arte.
#
# POR QUE: ate 03/set essas linhas rodavam com semente unica e o desvio
# reportado era ENTRE FOLDS, enquanto os backbones fine-tuned ja tinham 5
# sementes -- duas unidades de replicacao na mesma tabela. E o §5.6 mediu que o
# `hand-8` reproduz em 38,53 / 38,71 / 38,74 entre execucoes com a MESMA semente
# (nao-determinismo de GPU), o que torna indistinguivel de ruido qualquer delta
# da ordem de 0,5pp -- que e a ordem do marginal do grafo e do logit-add.
#
# A semente 42 ja esta em disco e o script retoma, entao so as 4 novas rodam.
# Estimativa a partir da coluna `seconds`: ~9,1 h no total.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3
wait_for() { while pgrep -af "$1" | grep python3 | grep -qv "bash -c"; do sleep 60; done; }

wait_for build_places_lt_view
echo "=== [$(date '+%F %T')] disco livre"

for ds in oilpainting rijksdraw multitask; do
  echo "=== [$(date '+%F %T')] $ds"
  $PY experiments/evaluate_preprocess_effect_oilpainting.py --dataset "$ds" \
      --heads narrow --seeds 42 43 44 45 46 \
      --output "results/${ds}_final_table.csv" || echo "FALHOU: $ds"
done
echo "=== [$(date '+%F %T')] FIM"
