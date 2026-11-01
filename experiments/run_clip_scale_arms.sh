#!/usr/bin/env bash
# Duas escalas de CLIP para a tabela de ablacao de captioner, da MESMA fonte.
#
# O cache de producao do B/32 (`outputs/multitask_clip_vitb32.npy`) foi feito da
# copia local reduzida, que /tmp levou. Extrair o L/14 dos originais e comparar
# com aquele cache confundiria ESCALA DO MODELO com fonte de imagem -- o §8 mede
# esse tipo de par valendo 4,91pp. Entao os dois saem dos originais, e o B/32
# vira CONTROLE: se ele reproduzir o 12,74 publicado, a fonte nao importa.
set -u
cd /exp/duducampos/Paintings
PY=.ABSvenv/bin/python3
IMGS=/tmp/prism_data/multitask_painting100k/images

extrair () {   # apelido  modelo
  local apelido="$1" modelo="$2" saida="outputs/multitask_clip_${1}_orig.npy"
  if [ -f "$saida" ]; then echo "== ${apelido}: cache ja existe"; return 0; fi
  echo "== extraindo ${apelido} (${modelo}) dos originais"
  $PY experiments/extract_clip_cache_model.py --dataset multitask \
      --clip-model "$modelo" --image-dir "$IMGS" --output "$saida" || return 1
}

bracos () {    # apelido  modelo
  local apelido="$1" modelo="$2"
  local saida="results/multitask_clip_text_arms_${apelido}_orig.json"
  if [ -f "$saida" ]; then echo "== ${apelido}: arms ja existe"; return 0; fi
  echo "== zero-shot ${apelido}"
  $PY experiments/evaluate_clip_text_arms.py --dataset multitask \
      --clip-model "$modelo" --clip-cache "outputs/multitask_clip_${apelido}_orig.npy" \
      --skip-probe --output "$saida" || return 1
}

extrair b32 openai/clip-vit-base-patch32  || { echo FALHOU_b32_extrai; exit 1; }
bracos  b32 openai/clip-vit-base-patch32  || { echo FALHOU_b32_arms;   exit 1; }
extrair l14 openai/clip-vit-large-patch14 || { echo FALHOU_l14_extrai; exit 1; }
bracos  l14 openai/clip-vit-large-patch14 || { echo FALHOU_l14_arms;   exit 1; }
echo "== fim"
