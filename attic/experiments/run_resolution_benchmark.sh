#!/usr/bin/env bash
set -u
cd /exp/duducampos/Paintings
for attempt in $(seq 1 20); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt" | tee -a logs/resolution_benchmark.log
  .ABSvenv/bin/python3 experiments/benchmark_modality_cost_by_resolution.py \
      --sides 128 256 512 1024 2048 --repeats 3 >> logs/resolution_benchmark.log 2>&1
  status=$?
  echo "=== [$(date +%H:%M:%S)] saiu com código $status" | tee -a logs/resolution_benchmark.log
  [ "$status" -eq 0 ] && break
  sleep 10
done
