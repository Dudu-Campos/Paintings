#!/usr/bin/env bash
# WL subtree no rijksdraw: o corpus onde a pergunta da LOCALIDADE mora (§5.19).
# Encadeamento em ARQUIVO, nao inline -- §8, oito vezes.
set -u
cd "$(dirname "$0")/.."
PY=.venv-rtx3090-py312/bin/python3
CACHE=/tmp/prism_data/rijksdraw_slic_rag
LOCK=/tmp/prism_wl_rijksdraw.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "lock existe, saindo"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }

if ! grep -q "pronto:" logs/cache_rag_slic_rijksdraw.log 2>/dev/null; then
  say "cache do RAG (SLIC), ~13 min esperados"
  "$PY" src/extractors/cache_rag_compact.py --dataset rijksdraw \
    --graph-dir outputs/rijksdraw_slic_full/graphs \
    --out-dir "$CACHE" --workers 16 > logs/cache_rag_slic_rijksdraw.log 2>&1
  say "cache rc=$?  shards=$(ls $CACHE/shard_*.npz 2>/dev/null | wc -l)"
else
  say "cache ja existe, pulando"
fi
grep -q "pronto:" logs/cache_rag_slic_rijksdraw.log || { say "cache FALHOU"; tail -5 logs/cache_rag_slic_rijksdraw.log; exit 1; }

say "pipeline WL: 3 bracos x 5 folds"
"$PY" experiments/evaluate_wl_pipeline_multitask.py --dataset rijksdraw \
  --cache-dir "$CACHE" --k 64 --folds 5 \
  --output results/rijksdraw_wl_pipeline.csv > logs/wl_pipeline_rijksdraw.log 2>&1
say "pipeline rc=$?"
say FIM
