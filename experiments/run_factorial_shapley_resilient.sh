#!/usr/bin/env bash
# Reinicia o fatorial 2^9 enquanto ele não terminar.
#
# Motivo: a máquina é compartilhada e o job já foi morto duas vezes por
# SIGKILL externo (sem traceback, sem estourar limite nosso -- medido: um
# ajuste na máscara maior custa 3,2 GB de pico, contra ~106 GB livres).
# O script Python é idempotente e retomável (pula (máscara, fold) já no CSV),
# então reiniciar é seguro e não reprocessa nada.
#
# Não usar `pgrep -f`/`pkill -f` para checar estado: o padrão casa com o
# próprio shell que o contém (gotcha registrado no CLAUDE.md, custou um
# exit 144 e três horas de monitor observando a si mesmo).
set -u
cd /exp/duducampos/Paintings

PY=.ABSvenv/bin/python3
OUT=results/factorial_shapley_multitask_25class.csv
LOG=logs/multitask_factorial_shapley.log
EXPECTED=2555          # 511 máscaras x 5 folds
MAX_TRIES=40

for attempt in $(seq 1 "$MAX_TRIES"); do
  rows=$(( $(wc -l < "$OUT") - 1 ))
  if [ "$rows" -ge "$EXPECTED" ]; then
    echo "=== [$(date +%H:%M:%S)] completo: $rows/$EXPECTED linhas" | tee -a "$LOG"
    exit 0
  fi
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt, $rows/$EXPECTED linhas" | tee -a "$LOG"
  $PY experiments/evaluate_factorial_shapley_multitask.py \
      --dataset multitask \
      --classes-file data/multitask_25class_benchmark.txt \
      --output "$OUT" >> "$LOG" 2>&1
  status=$?
  echo "=== [$(date +%H:%M:%S)] saiu com código $status" | tee -a "$LOG"
  [ "$status" -eq 0 ] && break
  sleep 10
done

rows=$(( $(wc -l < "$OUT") - 1 ))
echo "=== [$(date +%H:%M:%S)] fim: $rows/$EXPECTED linhas" | tee -a "$LOG"
