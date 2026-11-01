#!/usr/bin/env bash
set -u
cd /exp/duducampos/Paintings
LOG=logs/multitask_nested_mlp.log
for attempt in $(seq 1 40); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt" | tee -a "$LOG"
  .ABSvenv/bin/python3 experiments/evaluate_nested_mlp_multitask.py \
      --dataset multitask --classes-file data/multitask_25class_benchmark.txt >> "$LOG" 2>&1
  status=$?
  echo "=== [$(date +%H:%M:%S)] saiu com código $status" | tee -a "$LOG"
  [ "$status" -eq 0 ] && break
  sleep 10
done
