#!/bin/bash
# Cadeia completa image-to-text no COCO train2017+val2017 (~123k imagens).
# Roda 01 prepare -> 02 teacher (vLLM) -> 03 tokenizer -> 04 student -> 05 eval.
# Cada etapa loga em artifacts/full_run/<etapa>.log e escreve um marcador .done.
set -eo pipefail

REPO=/home/debian/william/Paintings
STAGE="$REPO/experiments/image-to-text"
WA=/home/debian/.cache/kagglehub/datasets/steubk/wikiart/versions/1
RUN="$STAGE/artifacts/full_run"
mkdir -p "$RUN"
cd "$STAGE"
source "$REPO/.venv-leak/bin/activate"

MASTER="$RUN/master.log"
step() { echo "[$(date +%H:%M:%S)] === $1 ===" | tee -a "$MASTER"; }

# ------------------------------------------------------------------
# 01 — prepare: indexa train2017 + val2017 num índice único (~123k)
# ------------------------------------------------------------------
if [ ! -f "$RUN/01.done" ]; then
  step "01 prepare (train2017+val2017)"
  stdbuf -oL python 01_prepare.py --split-source train2017 val2017 --seed 42 \
      > "$RUN/01_prepare.log" 2>&1
  touch "$RUN/01.done"
fi

# ------------------------------------------------------------------
# 02 — teacher: Qwen2-VL via vLLM sobre TODAS as imagens indexadas
# ------------------------------------------------------------------
if [ ! -f "$RUN/02.done" ]; then
  step "02 teacher vLLM (COCO completo)"
  # --batch 20000: fatia p/ não segurar 123k imagens PIL em RAM de uma vez;
  # o continuous batching do vLLM ainda maximiza o throughput dentro do chunk.
  stdbuf -oL python 02_teacher_text_llm.py --batch 20000 --max-tokens 64 \
      > "$RUN/02_teacher.log" 2>&1
  touch "$RUN/02.done"
fi

# ------------------------------------------------------------------
# 03 — tokenizer: re-treina BPE (vocab 12k) no corpus grande
# ------------------------------------------------------------------
if [ ! -f "$RUN/03.done" ]; then
  step "03 tokenizer (vocab 12k)"
  stdbuf -oL python 03_tokenizer.py --vocab-size 12000 --min-frequency 2 \
      > "$RUN/03_tokenizer.log" 2>&1
  touch "$RUN/03.done"
fi

# ------------------------------------------------------------------
# 04 — student: treina no COCO completo, early-stopping implícito
#      (o melhor checkpoint é por ROUGE-L de validação)
# ------------------------------------------------------------------
if [ ! -f "$RUN/04.done" ]; then
  step "04 student (treino completo)"
  stdbuf -oL python 04_student.py --epochs 40 --batch-size 256 --bf16 \
      --eval-every 1 > "$RUN/04_student.log" 2>&1
  touch "$RUN/04.done"
fi

# ------------------------------------------------------------------
# 05 — evaluate: fidelidade COCO test + WikiArt (vazamento + TF-IDF)
# ------------------------------------------------------------------
if [ ! -f "$RUN/05.done" ]; then
  step "05 evaluate (COCO test + WikiArt vs CATR)"
  stdbuf -oL python 05_evaluate.py --wikiart-root "$WA" --subset test \
      > "$RUN/05_evaluate.log" 2>&1
  touch "$RUN/05.done"
fi

step "CADEIA COMPLETA — FIM"
echo "EXIT_CODE=0" | tee -a "$MASTER"
