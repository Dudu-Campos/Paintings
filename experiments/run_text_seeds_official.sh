#!/usr/bin/env bash
# Barra de erro nos QUATRO corpora de split oficial, por SEMENTE.
#
# Eles rodaram numa particao so (o split publicado), entao nao tem +-, nao tem p
# e nao admitem teste pareado -- e e neles que moram os tres maiores marginais da
# serie (+20,18 food101, +8,83 places_lt, +8,59 minc2500). Reportar +20,18 ao
# lado de +0,84 +- 0,95 mistura dois tipos de numero.
#
# A particao NAO muda (e o split publicado); o que varia e a semente de TREINO.
# O §5.6 mediu que este experimento e limitado por ruido de treino e nao por
# numero de folds, e o controle de 09/set mostra o nao-determinismo de GPU
# movendo ate 0,42pp numa unica particao.
#
# Ordem SEMENTE-MAIOR de proposito: depois de ~11 min ha duas sementes nos quatro
# corpora (primeira leitura possivel), em vez de um corpus pronto e tres vazios.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.ABSvenv/bin/python3}
SEEDS=${SEEDS:-"43 44 45 46"}
DATASETS=${DATASETS:-"minc2500 nct_crc places_lt food101"}

for seed in $SEEDS; do
  for ds in $DATASETS; do
    out="results/${ds}_logit_add_text_seed${seed}.csv"
    # Retomavel: o script pula (condicao, fold) ja gravados. Um arquivo COMPLETO
    # (4 condicoes) e pulado inteiro em vez de re-treinado.
    if [ -f "$out" ] && [ "$(( $(wc -l < "$out") - 1 ))" -ge 4 ]; then
      echo "[skip] $ds seed $seed ja completo"; continue
    fi
    echo "=== $ds seed $seed  ($(date +%H:%M:%S)) ==="
    $PY experiments/evaluate_logit_add_text_multitask.py \
        --dataset "$ds" --seed "$seed" \
      || echo "FALHOU: $ds seed $seed" | tee -a logs/text_seeds_failures.txt
  done
done
echo "=== fim ($(date +%H:%M:%S)) ==="
