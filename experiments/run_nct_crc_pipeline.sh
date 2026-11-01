#!/usr/bin/env bash
# NCT-CRC-HE-100K: extração completa, em três etapas na ordem do §7.9.
#
# A ordem importa: a ResNet congelada e o probe do ImageNet saem ANTES do banco
# completo, para que a previsão ("é aqui que o pré-treino não transfere, logo é
# aqui que devemos ir relativamente melhor") fique registrada antes do resultado.
#
# Ressalva de projeto: as imagens são 224x224 por construção (tiles de lâmina a
# 0,5 um/px). Com --segments 2048 isso dá ~24 px por segmento, que é o mesmo
# critério pelo qual EuroSAT e CIFAR foram rejeitados. Rodamos a configuração
# padrão para manter comparabilidade com os outros cinco corpora, e a variante
# com menos segmentos fica como ablação separada.
set -u
cd "$(dirname "$0")/.."
IMG=/tmp/prism_data/nct_crc/NCT-CRC-HE-100K
PY=.ABSvenv/bin/python3

echo "=== [$(date +%H:%M:%S)] etapa 1: ResNet50 congelada"
if [ ! -f outputs/nct_crc_cnn_frozen/cnn_full_features.npy ]; then
  for t in 1 2 3; do
    $PY src/baselines/extract_cnn_full.py --model resnet50 --init imagenet \
      --image_dir "$IMG" --meta data/nct_crc_meta.csv \
      --output outputs/nct_crc_cnn_frozen && break
    echo "=== tentativa $t falhou, repetindo em 30s"; sleep 30
  done
fi

echo "=== [$(date +%H:%M:%S)] etapa 2: probe do ImageNet (previsão registrada)"
$PY experiments/probe_imagenet_domain.py --dataset nct_crc || echo "probe falhou"

echo "=== [$(date +%H:%M:%S)] etapa 3: banco de 8 modalidades"
SKIP_MULTISCALE=1 bash experiments/run_dataset_extraction.sh nct_crc "$IMG" 28

echo "=== [$(date +%H:%M:%S)] PIPELINE_NCT_PRONTO"
