#!/usr/bin/env bash
# UM WORKER POR CORPUS, em paralelo. A GPU estava em 8% com um processo e o
# perfil diz por que: 85% do tempo e transferencia CPU->GPU de 10 MB em lotes
# pequenos; forward+backward e 6%. Nao ha o que acelerar DENTRO da unidade sem
# tocar no `train_one` de producao (§5.6: nao reimplementar), entao a
# paralelizacao e por UNIDADE.
#
# Um worker por corpus, e nao por semente, por tres razoes:
#   - cada worker carrega as features UMA vez (multitask sao ~2 GB por banco)
#   - o cache do WND-CHARM e por corpus: dois workers no mesmo corpus colidiriam
#   - CSV por corpus e a convencao que o `*_handcrafted_banks_lda.csv` ja usa
set -u
cd "$(dirname "$0")/.."
PY=.venv-rtx3090-py312/bin/python3
LOCK=/tmp/prism_banks_dense_par.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "lock existe, saindo"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }

PIDS=()
for ds in multitask places_lt food101 nct_crc minc2500 rijksdraw oilpainting; do
  "$PY" experiments/evaluate_banks_dense_mlp.py --datasets "$ds" \
    --seeds 42 43 44 45 46 --arquiteturas denso-128 \
    --output "results/${ds}_banks_dense_mlp.csv" \
    > "logs/banks_dense_${ds}.log" 2>&1 &
  PIDS+=($!)
  say "lancado $ds pid=${PIDS[-1]}"
  sleep 8   # escalona o pico de leitura das features, que e I/O de NFS
done
say "esperando ${#PIDS[@]} workers"
FALHOU=0
for pid in "${PIDS[@]}"; do wait "$pid" || FALHOU=$((FALHOU+1)); done
say "FIM -- $FALHOU workers com rc != 0"
