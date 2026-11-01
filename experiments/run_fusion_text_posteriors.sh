#!/usr/bin/env bash
# Posteriors de `hand-8 + grafo + texto` nos SETE corpora, para o segundo
# scatter da figura de regimes de peso (§4.1f).
#
# Retomavel em tres niveis, que e o que o §1 exige de job longo:
#   - bloco de texto aninhado    cache por (corpus, fold), independente de semente
#   - posteriors por corpus      `outputs/{corpus}_our_*_text.npz`, relidos
#   - corpora                    cada um e um comando separado
set -u
PY="${PY:-.ABSvenv/bin/python3}"
CV="multitask oilpainting rijksdraw"
OFICIAL="nct_crc minc2500 places_lt food101"

for c in $CV; do
    echo "=== [CV] $c ==="
    "$PY" experiments/evaluate_decision_fusion_cv.py --dataset $c --with-text \
        $([ "$c" = multitask ] && echo "--grid-name multitask125") || exit 1
done
for c in $OFICIAL; do
    echo "=== [oficial] $c ==="
    "$PY" experiments/evaluate_decision_fusion_official.py --dataset $c --with-text || exit 1
done
echo "=== TODOS OS SETE CONCLUIDOS ==="
