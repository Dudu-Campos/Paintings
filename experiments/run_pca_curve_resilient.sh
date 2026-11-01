#!/usr/bin/env bash
# Auto-restart: a absinto mata jobs longos por SIGKILL externo.
set -u
cd /exp/duducampos/Paintings
OUT=results/pca_curve_by_subset_multitask.csv
LOG=logs/multitask_pca_curve.log
for attempt in $(seq 1 40); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt" | tee -a "$LOG"
  .ABSvenv/bin/python3 experiments/evaluate_pca_curve_by_subset_multitask.py \
      --dataset multitask --classes-file data/multitask_25class_benchmark.txt \
      --output "$OUT" >> "$LOG" 2>&1
  status=$?
  echo "=== [$(date +%H:%M:%S)] saiu com código $status" | tee -a "$LOG"
  [ "$status" -eq 0 ] && break
  sleep 10
done
