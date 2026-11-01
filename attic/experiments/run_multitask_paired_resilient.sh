#!/usr/bin/env bash
# Auto-restart: a absinto mata jobs longos por SIGKILL externo, sem traceback.
# A saída é incremental e os grafos ficam em cache por (recorte, fold), então
# reentrar é barato.
set -u
cd "$(dirname "$0")/.."
for attempt in $(seq 1 60); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt"
  .ABSvenv/bin/python3 experiments/evaluate_multitask_125_vs_25_paired.py "$@" && break
  status=$?; echo "=== [$(date +%H:%M:%S)] caiu (codigo $status), retomando em 30s"; sleep 30
done
