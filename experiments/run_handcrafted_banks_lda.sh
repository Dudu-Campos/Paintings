#!/usr/bin/env bash
# Haralick x HOG x WND-CHARM x hand-8 sob sonda linear, nos sete corpora.
# CPU puro. Ordem do mais barato ao mais caro. Retomavel por linha.
set -u
cd "$(dirname "$0")/.."
LOCK=/tmp/prism_banks_lda.lock
PY=.venv-rtx3090-py312/bin/python3
if ! mkdir "$LOCK" 2>/dev/null; then echo "lock existe, saindo"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }
for ds in rijksdraw nct_crc minc2500 food101 places_lt multitask; do
  say "$ds"
  "$PY" experiments/evaluate_handcrafted_banks_lda.py --dataset "$ds" --seeds 42 \
    > "logs/banks_lda_${ds}.log" 2>&1
  say "$ds terminou (rc=$?)"
  tail -8 "logs/banks_lda_${ds}.log" | grep -E "^\[.*(BAcc|ordem|ORDEM)" | tail -3
done
say "FIM"
