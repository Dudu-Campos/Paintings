#!/usr/bin/env bash
# Ablacao por view em LDA nos sete corpora -- a regra do §7.19 (sonda linear
# DENTRO da familia hand-designed). Sem grafo de similaridade e sem texto, por
# decisao de 06/out: os dois sao blocos tipo-posterior e o §7.19 mede que o LDA
# os favorece (razao LDA/MLP 1,005 no grafo contra mediana ~0,75).
#
# CPU puro -- nao depende da GPU, que caiu pela terceira vez em cinco dias.
# Ordem do mais barato ao mais caro. Retomavel: o script pula (classifier,
# condition, fold) que ja estejam no CSV.
set -u
cd "$(dirname "$0")/.."
LOCK=/tmp/prism_view_ablation.lock
PY=.venv-rtx3090-py312/bin/python3
if ! mkdir "$LOCK" 2>/dev/null; then echo "lock existe, saindo"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }

for ds in oilpainting rijksdraw nct_crc food101 minc2500 multitask places_lt; do
  say "$ds"
  "$PY" experiments/evaluate_minc2500_view_ablation.py \
    --dataset "$ds" --classifiers lda --seeds 42 43 44 45 46 > "logs/view_ablation_${ds}.log" 2>&1
  rc=$?
  say "$ds terminou (rc=$rc)"
  [ $rc -ne 0 ] && tail -3 "logs/view_ablation_${ds}.log"
done
say "FIM"
