#!/usr/bin/env bash
# Legendas do estudante para os QUATRO corpora nao-artisticos.
#
# O §5.18 registrou a previsao ANTES de rodar: "o canal deve funcionar onde
# CONTEUDO identifica o rotulo (arte) e falhar onde nao identifica (MINC e
# material, NCT-CRC e tecido)". Os tres de arte ja estao medidos e ordenam como
# previsto -- oilpainting +4,25 > multitask +0,84 > rijksdraw +0,01 (§5.18b).
# Estes quatro sao o outro lado da previsao, e sem eles um revisor pergunta por
# que o canal so foi testado onde funciona.
#
# Em serie: uma GPU, e rodar em paralelo so faria disputar memoria.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3

run() {  # nome, meta, image-dir
  echo "=== [$(date '+%d/%b %H:%M:%S')] inicio $1"
  $PY experiments/evaluate_student_captions_multitask.py \
      --dataset "$1" --meta "$2" --image-dir "$3" \
      || { echo "FALHOU: $1"; return 1; }
  echo "=== [$(date '+%d/%b %H:%M:%S')] fim $1"
}

run minc2500  data/minc2500_meta.csv  /tmp/prism_data/minc/minc-2500/images
run food101   data/food101_meta.csv   /tmp/prism_data/food101/food-101/images
run nct_crc   data/nct_crc_meta.csv   /tmp/prism_data/nct_crc/NCT-CRC-HE-100K
run places_lt data/places_lt_meta.csv data/places_lt/images
echo "=== TUDO PRONTO"
