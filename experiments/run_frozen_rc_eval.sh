#!/usr/bin/env bash
# Avalia as congeladas resize+crop de rijksdraw e multitask (§7.17), depois que
# a cadeia de legendas liberar a maquina.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3
wait_for() { while pgrep -af "$1" | grep python3 | grep -qv "bash -c"; do sleep 60; done; }

wait_for "evaluate_student_captions|evaluate_logit_add_text"
echo "=== [$(date '+%F %T')] maquina livre"
for ds in rijksdraw multitask; do
  echo "=== [$(date '+%F %T')] preprocess effect: $ds"
  $PY experiments/evaluate_preprocess_effect_oilpainting.py --dataset "$ds" \
      --heads narrow --output "results/${ds}_final_table.csv" || echo "FALHOU: $ds"
done
echo "=== [$(date '+%F %T')] FIM"
