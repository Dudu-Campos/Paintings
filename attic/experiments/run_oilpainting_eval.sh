#!/usr/bin/env bash
set -u
cd /exp/duducampos/Paintings
for clf in lda mlp; do
  for attempt in $(seq 1 20); do
    echo "=== [$(date +%H:%M:%S)] $clf tentativa $attempt" | tee -a logs/oilpainting_eval.log
    .ABSvenv/bin/python3 experiments/evaluate_external_modalities.py \
        --datasets oilpainting --classifier "$clf" --min-per-class 0 \
        --output "results/external_modalities_oilpainting_${clf}.csv" \
        >> logs/oilpainting_eval.log 2>&1
    status=$?
    echo "=== [$(date +%H:%M:%S)] $clf saiu com código $status" | tee -a logs/oilpainting_eval.log
    [ "$status" -eq 0 ] && break
    sleep 10
  done
done
