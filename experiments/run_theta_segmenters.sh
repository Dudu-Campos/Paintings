#!/usr/bin/env bash
# Varredura de theta dos dois segmentadores, EM SERIE: cada um usa 28 workers e
# rodar em paralelo so os faria disputar os mesmos nucleos.  ~5,5 h no total
# (Felz 3,2 h + SLIC 2,3 h, estimado dos ms/img do §5.4b).
set -u
PY=${PY:-.ABSvenv/bin/python3}
for method in slic felzenszwalb; do
  echo "=== [$(date '+%d/%b %H:%M:%S')] inicio $method"
  $PY experiments/sweep_theta_segmenters.py --method "$method" --workers 28 --folds 3 \
    >> "logs/theta_${method}.log" 2>&1
  echo "=== [$(date '+%d/%b %H:%M:%S')] fim $method (exit $?)"
done
echo "=== TUDO PRONTO"
