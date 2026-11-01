#!/usr/bin/env bash
# Quatro bancos sob MLP sem branch, com AUTO-RESTART (§1: todo job longo nasce
# com wrapper e saida incremental idempotente -- eu nao tinha posto, e a queda
# de GPU de 08/out 12:33 matou 4 workers de uma vez e perdeu a fila).
#
# TRES workers concorrentes, nao sete. A queda NAO foi medida como efeito do
# paralelismo (os 4 que morreram estavam com a GPU em MENOS carga que o pico de
# 7), mas com menos workers cada queda custa menos e o restart progride mais.
#
# Retomada: cada unidade grava a linha no CSV do proprio corpus, e o script
# refaz so o que falta. Fila ordenada pelo que FALTA, nao pelo nome.
set -u
cd "$(dirname "$0")/.."
PY=.venv-rtx3090-py312/bin/python3
LOCK=/tmp/prism_banks_dense_res.lock
PARALELO=3
MAX_TENTATIVAS=20
if ! mkdir "$LOCK" 2>/dev/null; then echo "lock existe, saindo"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }

falta() {   # quantas unidades faltam no corpus $1
  "$PY" - "$1" <<'PYEOF'
import os, sys
import pandas as pd
ALVO = {'rijksdraw':20,'oilpainting':20,'multitask':20,
        'nct_crc':4,'minc2500':4,'food101':4,'places_lt':3}
ds = sys.argv[1]; alvo = ALVO[ds] * 5
f = f'results/{ds}_banks_dense_mlp.csv'
if not os.path.exists(f):
    print(alvo); raise SystemExit
d = pd.read_csv(f)
print(max(0, alvo - int((d.arquitetura == 'denso-128').sum())))
PYEOF
}

um_corpus() {   # roda $1 ate completar ou estourar as tentativas
  local ds=$1 t=0
  while [ "$t" -lt "$MAX_TENTATIVAS" ]; do
    local n; n=$(falta "$ds")
    if [ "$n" -eq 0 ]; then say "$ds COMPLETO"; return 0; fi
    t=$((t+1))
    say "$ds tentativa $t, faltam $n"
    "$PY" experiments/evaluate_banks_dense_mlp.py --datasets "$ds" \
      --seeds 42 43 44 45 46 --arquiteturas denso-128 \
      --output "results/${ds}_banks_dense_mlp.csv" \
      >> "logs/banks_dense_${ds}.log" 2>&1
    local rc=$?
    [ "$rc" -ne 0 ] && { say "$ds rc=$rc -- esperando a GPU 60s"; sleep 60; }
  done
  say "$ds DESISTIU apos $MAX_TENTATIVAS tentativas"; return 1
}

# fila do que mais falta para o que menos falta
FILA=$(for ds in multitask rijksdraw food101 nct_crc minc2500 places_lt oilpainting; do
         echo "$(falta "$ds") $ds"; done | sort -rn | awk '$1 > 0 {print $2}')
say "fila: $(echo $FILA)"
[ -z "$FILA" ] && { say "nada a fazer"; exit 0; }

ATIVOS=0
for ds in $FILA; do
  um_corpus "$ds" & ATIVOS=$((ATIVOS+1))
  sleep 10
  if [ "$ATIVOS" -ge "$PARALELO" ]; then wait -n 2>/dev/null || true; ATIVOS=$((ATIVOS-1)); fi
done
wait
say FIM
