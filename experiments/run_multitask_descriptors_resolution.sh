#!/usr/bin/env bash
# Descriptors em resolucao alta: o teste que decide se o ganho e "mais pixels".
#
# O §9.5 mediu que o pipeline precisa de PELO MENOS UMA familia de segmentacao em
# resolucao alta (as duas a 224 px custam 3,30pp). A objecao que sobra e mais
# grave: e se todo o ganho for so ver mais pixels? Os descritores globais rodam a
# 128 px (`TARGET` no extrator, sem justificativa registrada -- §7.1).
#
# Se Descriptors@1024 >> Descriptors@128, resolucao e alavanca geral e a nossa
# historia vira "olhamos mais pixels que a CNN".
# Se Descriptors@1024 ~ Descriptors@128, a resolucao ajuda ESPECIFICAMENTE as
# views de grafo, o que e afirmacao sobre as representacoes e nao sobre pixels.
#
# CONFUNDIDOR a declarar: LBP (R=1 e R=3) e a distancia do GLCM estao em unidades
# de pixel, entao mudar o alvo muda O QUE e medido, nao so o custo. Mesma
# pegadinha do `felz_scale` (§9.3). Nao ha como separar sem reparametrizar.
set -u
PY=.ABSvenv/bin/python3
IMG=data/multitask_painting100k/original/images
META=data/multitask_meta.csv
W=${1:-10}
LOG=logs/multitask_descriptors_resolution.log
mkdir -p logs

for T in 512 1024; do
  OUT=outputs/multitask_descriptors_r${T}
  if [ -f "$OUT/desc_full_features.npy" ]; then
    echo "=== [$(date +%H:%M:%S)] target=$T JA FEITO" | tee -a "$LOG"; continue
  fi
  echo "=== [$(date +%H:%M:%S)] descriptors target=$T (workers=$W)" | tee -a "$LOG"
  $PY src/extractors/extract_descriptors_full.py \
      --image_dir "$IMG" --meta "$META" --output "$OUT" \
      --target "$T" --workers "$W" >> logs/multitask_desc_r${T}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] target=$T saiu com $?" | tee -a "$LOG"
done
echo "=== [$(date +%H:%M:%S)] COMPLETO" | tee -a "$LOG"
