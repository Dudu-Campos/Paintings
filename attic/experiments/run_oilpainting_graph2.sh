#!/usr/bin/env bash
set -u
cd /exp/duducampos/Paintings
run () {
  local tag=$1; shift
  for attempt in $(seq 1 20); do
    echo "=== [$(date +%H:%M:%S)] $tag tentativa $attempt" | tee -a logs/oilpainting_graph.log
    .ABSvenv/bin/python3 experiments/evaluate_multitask_similarity_graph.py \
        --dataset oilpainting --min-per-class 0 --exclude ResNet50 Multiscale "$@" \
        >> logs/oilpainting_graph.log 2>&1
    status=$?
    echo "=== [$(date +%H:%M:%S)] $tag saiu com código $status" | tee -a logs/oilpainting_graph.log
    [ "$status" -eq 0 ] && return 0
    sleep 10
  done
}
# base de producao (sem Multiscale), com e sem log-lift
run "sem MS, log-lift"     --output-dir outputs/oilpainting_graph_no_multiscale
run "sem MS, sem log-lift" --no-prior-correct \
    --output-dir outputs/oilpainting_graph_no_ms_no_loglift
