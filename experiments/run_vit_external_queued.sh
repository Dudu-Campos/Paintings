#!/usr/bin/env bash
# Extrai ViT-B/16 congelado para Food-101 e Places-LT, mas SÓ depois que o
# fine-tuning liberar a GPU. A absinto é compartilhada e em 25/ago quatro
# processos simultâneos saturaram a utilização em 100% -- o fine-tuning caiu de
# 2,5 para 2,83 min/época. Esperar é mais rápido que competir.
set -u
cd "$(dirname "$0")/.."
WAIT_FOR=${WAIT_FOR:-train_resnet_wikiart}
while ps -eo args | grep -q "[^]]$WAIT_FOR"; do sleep 120; done
echo "=== [$(date +%H:%M:%S)] GPU liberada, começando"
for ds in food101 places_lt; do
  case $ds in
    food101)   IMG=/tmp/prism_data/food-101/images ;;
    places_lt) IMG=/tmp/prism_data/places_lt/images ;;
  esac
  OUT=outputs/${ds}_vit_frozen
  [ -f "$OUT/cnn_full_features.npy" ] && { echo "=== $ds já extraído"; continue; }
  echo "=== [$(date +%H:%M:%S)] $ds ($IMG)"
  for attempt in 1 2 3; do
    .ABSvenv/bin/python3 src/baselines/extract_cnn_full.py \
      --model vit_b_16 --init imagenet \
      --image_dir "$IMG" --meta data/${ds}_meta.csv --output "$OUT" && break
    echo "=== tentativa $attempt falhou, repetindo em 20s"; sleep 20
  done
  echo "=== [$(date +%H:%M:%S)] $ds pronto"
done
echo TUDO_PRONTO
