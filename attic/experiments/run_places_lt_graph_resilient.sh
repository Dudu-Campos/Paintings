#!/usr/bin/env bash
# Auto-restart para evaluate_places_lt_similarity_graph.py.
# A absinto mata jobs longos por SIGKILL externo, sem traceback; a saída do
# script é incremental e idempotente, então basta reentrar.
set -u
cd "$(dirname "$0")/.."
for attempt in $(seq 1 40); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt"
  .ABSvenv/bin/python3 experiments/evaluate_places_lt_similarity_graph.py "$@" && break
  status=$?; echo "=== [$(date +%H:%M:%S)] caiu (codigo $status), retomando em 20s"
  sleep 20
done
