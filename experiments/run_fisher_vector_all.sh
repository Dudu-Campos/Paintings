#!/usr/bin/env bash
# Fisher vector nos SETE corpora (oito extracoes: o nct_crc tem coorte de teste
# separada), na configuracao que se aproxima do protocolo publicado.
#
# POR QUE A CONFIGURACAO NAO E A PUBLICADA. A linhagem que o artigo do Food-101
# delega (ele diz "the same parameters as in [19]", Juneja et al. CVPR 2013, e
# cita a piramide por [32], Sanchez et al. IJCV 2013) e K=256, PCA-80 e piramide
# de 8 celulas -- 2*256*80*8 = 327.680 dimensoes, ou 708 GB nos sete corpora,
# que nenhuma sonda deste projeto aguenta (o LDA levou 2,4 h em 32.768d).
#
# O que esta receita faz:
#   piramide 1x1+3x1   ACRESCENTA o nivel espacial que faltava (o layout de
#                      Sanchez: imagem inteira + tres faixas horizontais)
#   PCA-80             casa com a linhagem (era 64)
#   GMM K=64           e o UNICO parametro do FV que o artigo do Food-101 de
#                      fato especifica ("a gmm with 64 modes"), e e o que faz a
#                      piramide caber: 2*64*80*4 = 40.960d
# Desvio declarado: omitimos o nivel 2x2 (seriam 8 celulas, 81.920d) e usamos
# SIFT onde a pipeline de superpixel deles usa SURF.
#
# O custo NAO muda com a piramide: medido em 11/out, o SIFT e 99,0% do custo por
# imagem e nao se repete -- os posteriores saem uma vez e cada nivel so
# reparticiona quem ja esta calculado. A piramide sai por +0,3%.
#
# Retomavel em dois niveis: o driver pula corpus com `features.npy` pronto, e o
# extrator pula shard pronto. Pode ser morto e relancado a qualquer momento.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv-rtx3090-py312/bin/python3}
WORKERS=${WORKERS:-112}
LOGDIR=logs/fisher_vector
mkdir -p "$LOGDIR"

# ordem por VALOR, nao por custo: o oilpainting e o mais barato e serve de teste
# de ponta a ponta do codigo novo em escala real; o food101 vem depois porque e
# o unico com numero publicado (IFV 38,88) e portanto a validacao externa da
# implementacao; o multitask e o corpus principal de arte.
CORPORA=(
  "oilpainting|data/oilpainting/image"
  "food101|data/food101/food-101/images"
  "multitask|data/multitask_painting100k/original/images"
  "nct_crc|data/nct_crc/images/NCT-CRC-HE-100K"
  "nct_crc_val|data/nct_crc/images/CRC-VAL-HE-7K"
  "places_lt|data/places_lt/images"
  "minc2500|data/minc2500/minc-2500/images"
  "rijksdraw|data/rijksmuseum/drawing"
)

for entrada in "${CORPORA[@]}"; do
  ds="${entrada%%|*}"
  img="${entrada##*|}"
  out="outputs/${ds}_fisher_vector"
  if [ -f "$out/features.npy" ]; then
    echo "[$(date +%H:%M:%S)] $ds: features.npy existe -- pulando"
    continue
  fi
  if [ ! -d "$img" ]; then
    echo "[$(date +%H:%M:%S)] !! $ds: $img nao existe -- PULANDO" | tee -a "$LOGDIR/driver.log"
    continue
  fi
  # a coorte de teste do nct_crc TEM de usar o codebook do treino: PCA e GMM
  # ajustadas separadamente dariam espacos de features diferentes nos dois lados
  # do split, e nada na saida denunciaria.
  if [ "$ds" = "nct_crc_val" ]; then
    if [ ! -f outputs/nct_crc_fisher_vector/codebook.npz ]; then
      echo "[$(date +%H:%M:%S)] !! nct_crc_val sem o codebook do treino -- PULANDO" \
        | tee -a "$LOGDIR/driver.log"
      continue
    fi
    mkdir -p "$out"
    cp outputs/nct_crc_fisher_vector/codebook.npz "$out/codebook.npz"
    echo "[$(date +%H:%M:%S)] nct_crc_val: codebook do TREINO copiado" | tee -a "$LOGDIR/driver.log"
  fi
  echo "[$(date +%H:%M:%S)] $ds <- $img" | tee -a "$LOGDIR/driver.log"
  t0=$(date +%s)
  "$PY" -u src/extractors/extract_fisher_vector_features.py \
    --dataset "$ds" --image-dir "$img" --workers "$WORKERS" \
    --max-side 512 --step 8 --scales 16,24,32,48 \
    --pca-dim 80 --gmm-k 64 --piramide 1x1+3x1 \
    --max-desc 10000 --fit-images 1500 --fit-threads 32 \
    > "$LOGDIR/${ds}.log" 2>&1
  rc=$?
  dt=$(( ($(date +%s) - t0) / 60 ))
  echo "[$(date +%H:%M:%S)] $ds terminou rc=$rc em ${dt} min" | tee -a "$LOGDIR/driver.log"
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] !! $ds FALHOU -- ver $LOGDIR/${ds}.log" | tee -a "$LOGDIR/driver.log"
  fi
done
echo "[$(date +%H:%M:%S)] driver concluido" | tee -a "$LOGDIR/driver.log"
