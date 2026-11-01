#!/usr/bin/env bash
# Reextrai as congeladas com Resize(256)+CenterCrop(224) para rijksdraw e
# multitask, que so tinham Letterbox (§7.17).
#
# POR QUE: a extracao congelada usa Letterbox com barras cinzas, mas os
# fine-tuned avaliam com resize+crop, e o §8 mediu o par valendo 4,91pp nos
# MESMOS pesos. Usar Letterbox so no congelado HANDICAPA o adversario -- corre a
# nosso favor, o que e exatamente o motivo de consertar. No oilpainting o
# conserto valeu +0,49pp (ResNet) e +0,95pp (ViT).
#
# Le os ORIGINAIS do multitask, nao a copia local: ela esta com o lado menor
# reduzido a 256 (§9.0). Para resize_crop a fonte quase nao importa (Resize(256)
# normaliza o lado menor), mas a assimetria fica declarada em vez de implicita.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3

wait_for() { while pgrep -af "$1" | grep python3 | grep -qv "bash -c"; do sleep 30; done; }
wait_for "evaluate_logit_add|evaluate_student_captions"
echo "=== [$(date '+%F %T')] maquina livre"

run() {  # dataset image_dir meta model
  local out="outputs/$1_$4_frozen_rc"
  if [ -f "$out/cnn_full_features.npy" ]; then echo "  ja existe: $out"; return; fi
  echo "=== [$(date '+%F %T')] $1 / $4 / resize_crop"
  $PY src/baselines/extract_cnn_full.py \
      --image_dir "$2" --meta "$3" --output "$out" \
      --model "$4" --init imagenet --preprocess resize_crop \
      || echo "FALHOU: $1/$4"
}

for m in resnet50 vit_b_16; do
  run rijksdraw  data/rijksmuseum/drawing               data/rijksdraw_meta.csv  "$m"
  run multitask  data/multitask_painting100k/original/images data/multitask_meta.csv  "$m"
done
echo "=== [$(date '+%F %T')] FIM"
