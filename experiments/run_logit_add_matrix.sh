#!/usr/bin/env bash
# Roda o logit-add nos dois corpora de arte que ja tem fine-tuning, emitindo
# BAcc, Acc e F1 (a versao anterior so tinha BAcc, §5.6).
#
# ENCADEADO, nao paralelo: a GPU desta maquina e compartilhada (havia 28,7 GB de
# outro usuario em 02/set) e o §8 registra que rodar medicoes concorrentes ja
# invalidou resultado aqui. Tambem espera o job de fusao de decisao terminar.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3

wait_for() {  # espera um script sair, sem casar com o proprio shell (§8)
  local pattern="$1"
  while true; do
    local pid
    pid=$(pgrep -af "$pattern" | grep python3 | grep -v "bash -c" | awk '{print $1}' | head -1)
    [ -z "$pid" ] && break
    sleep 60
  done
}

wait_for evaluate_decision_fusion_cv
echo "=== [$(date '+%F %T')] fusao de decisao liberou a maquina"

for ds in multitask oilpainting; do
  echo "=== [$(date '+%F %T')] logit-add $ds"
  $PY experiments/evaluate_logit_add_multitask.py \
      --dataset "$ds" --output "results/${ds}_logit_add_metrics.csv" || echo "FALHOU: $ds"
done
echo "=== [$(date '+%F %T')] FIM"
