#!/usr/bin/env bash
set -u
cd /exp/duducampos/Paintings
# Duas condicoes: com e sem Multiscale na base. O grafo nunca inclui
# Multiscale no espaco de construcao -- com 1.032 de 1.671 dims ele domina a
# distancia euclidiana do kNN (viés registrado no CLAUDE.md de 21/ago).
run () {
  local tag=$1; shift
  for attempt in $(seq 1 20); do
    echo "=== [$(date +%H:%M:%S)] $tag tentativa $attempt" | tee -a logs/oilpainting_graph.log
    .ABSvenv/bin/python3 experiments/evaluate_multitask_similarity_graph.py \
        --dataset oilpainting --min-per-class 0 "$@" >> logs/oilpainting_graph.log 2>&1
    status=$?
    echo "=== [$(date +%H:%M:%S)] $tag saiu com código $status" | tee -a logs/oilpainting_graph.log
    [ "$status" -eq 0 ] && return 0
    sleep 10
  done
}
run "com Multiscale" --exclude ResNet50 --graph-exclude Multiscale \
    --output-dir outputs/oilpainting_graph_with_multiscale
run "sem Multiscale" --exclude ResNet50 Multiscale \
    --output-dir outputs/oilpainting_graph_no_multiscale
