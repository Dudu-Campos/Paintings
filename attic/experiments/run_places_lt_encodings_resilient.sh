#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
for attempt in $(seq 1 20); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt"
  .ABSvenv/bin/python3 experiments/evaluate_places_lt_graph_encodings.py "$@" && break
  status=$?; echo "=== [$(date +%H:%M:%S)] caiu (codigo $status), retomando em 20s"; sleep 20
done
