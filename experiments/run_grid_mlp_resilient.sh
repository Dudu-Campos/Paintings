#!/usr/bin/env bash
# Grade representação x dados com o BranchMLP de produção, em vez da sonda LDA.
# A saída é incremental e idempotente (retoma por célula), então reentrar é barato.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.ABSvenv/bin/python3}
for tentativa in $(seq 1 40); do
  echo "=== [$(date +'%d/%b %H:%M:%S')] tentativa $tentativa"
  $PY experiments/evaluate_representation_data_grid_multitask.py \
    --classifier mlp --classes-file data/multitask_25class_benchmark.txt \
    --output results/representation_data_grid_multitask_mlp.csv && break
  status=$?; echo "=== caiu (codigo $status), retomando em 30s"; sleep 30
done
echo "=== [$(date +'%d/%b %H:%M:%S')] GRADE_MLP_PRONTA"
