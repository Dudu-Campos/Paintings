#!/usr/bin/env bash
# Quatro bancos sob MLP SEM BRANCH, 7 corpora, 5 sementes (§5.19b).
# LARGURA A LARGURA e SEMENTE A SEMENTE, nao corpus a corpus: se morrer, todos
# os corpora ficam na MESMA profundidade em vez de 3 prontos e 4 intocados.
set -u
cd "$(dirname "$0")/.."
PY=.venv-rtx3090-py312/bin/python3
LOCK=/tmp/prism_banks_dense.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "lock existe, saindo"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }

for seed in 42 43 44 45 46; do
  for ds in rijksdraw oilpainting multitask nct_crc minc2500 food101 places_lt; do
    say "s$seed $ds"
    "$PY" experiments/evaluate_banks_dense_mlp.py --datasets "$ds" \
      --seeds "$seed" --arquiteturas denso-128 \
      >> "logs/banks_dense_${ds}.log" 2>&1
    say "s$seed $ds rc=$?"
  done
  say "SEMENTE $seed COMPLETA"
done
say FIM
