#!/usr/bin/env bash
# Wrapper de auto-restart para o grafo de similaridade no Multitask.
# Mesma razão do fatorial: a absinto mata jobs longos por SIGKILL externo.
# O script Python é retomável (pula (classificador, condição, fold) já no CSV).
set -u
cd /exp/duducampos/Paintings

OUT=outputs/multitask_similarity_graph/fold_results.csv
LOG=logs/multitask_similarity_graph.log
EXPECTED=20            # 2 classificadores x 2 condições x 5 folds
MAX_TRIES=40

for attempt in $(seq 1 "$MAX_TRIES"); do
  rows=0
  [ -f "$OUT" ] && rows=$(( $(wc -l < "$OUT") - 1 ))
  if [ "$rows" -ge "$EXPECTED" ]; then
    echo "=== [$(date +%H:%M:%S)] completo: $rows/$EXPECTED" | tee -a "$LOG"
    exit 0
  fi
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt, $rows/$EXPECTED linhas" | tee -a "$LOG"
  .ABSvenv/bin/python3 experiments/evaluate_multitask_similarity_graph.py \
      --dataset multitask \
      --classes-file data/multitask_25class_benchmark.txt >> "$LOG" 2>&1
  status=$?
  echo "=== [$(date +%H:%M:%S)] saiu com código $status" | tee -a "$LOG"
  [ "$status" -eq 0 ] && break
  sleep 10
done

rows=0
[ -f "$OUT" ] && rows=$(( $(wc -l < "$OUT") - 1 ))
echo "=== [$(date +%H:%M:%S)] fim: $rows/$EXPECTED linhas" | tee -a "$LOG"
