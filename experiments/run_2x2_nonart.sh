#!/usr/bin/env bash
# 2x2 (grafo aditivo x texto) nos corpora nao-artisticos com split OFICIAL.
# food101 ja rodou. nct_crc fica de FORA: a coorte dele no registry e so treino
# e o teste e a coorte separada `nct_crc_val` (§4.1d), que o script atual nao
# atravessa -- precisa de TEST_COHORT, como o evaluate_preprocess_effect tem.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3
for d in minc2500 places_lt; do
  echo "=== [$(date '+%T')] inicio $d"
  $PY experiments/evaluate_logit_add_text_multitask.py --dataset "$d" \
    || echo "FALHOU: $d"
  echo "=== [$(date '+%T')] fim $d"
done
echo "=== TUDO PRONTO"
