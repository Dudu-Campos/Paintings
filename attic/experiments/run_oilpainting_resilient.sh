#!/usr/bin/env bash
set -u
cd /exp/duducampos/Paintings
for attempt in $(seq 1 40); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt" | tee -a logs/oilpainting_extraction.log
  bash experiments/run_oilpainting_extraction.sh >> logs/oilpainting_driver.log 2>&1
  status=$?
  echo "=== [$(date +%H:%M:%S)] driver saiu com código $status" | tee -a logs/oilpainting_extraction.log
  if [ -f results/line_graph_felz_oilpainting_features.parquet ] && \
     [ -f outputs/oilpainting_quadrant_features/bottom_right/slic/combined_features.parquet ]; then
    echo "=== [$(date +%H:%M:%S)] completo" | tee -a logs/oilpainting_extraction.log
    break
  fi
  sleep 10
done
