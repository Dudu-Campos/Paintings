#!/usr/bin/env bash
# Fisher vector na OUTRA maquina. Divisao combinada em 12/out: a margarita NAO
# toca em nada que este script produz, e este script NAO toca no food101 nem no
# rijksdraw transdutivo, que ja estao em disco. O §1 e explicito -- nunca o
# mesmo --out-dir em duas maquinas, porque os shards incrementais sao o MESMO
# arquivo no NFS.
#
# PRE-REQUISITO, e e o unico: `kornia` so existe no venv da margarita.
#   <venv>/bin/pip install kornia==0.8.2
#   <venv>/bin/python3 -c "from kornia.feature import DenseSIFTDescriptor; print('ok')"
#
# Lancar com PY apontando para o interpretador LOCAL (§1: os venvs no NFS nao
# sao portaveis -- cada bin/python3 aponta para um interpretador da maquina):
#   PY=.ABSvenv/bin/python3 setsid nohup bash experiments/run_fisher_outra_maquina.sh \
#     > logs/fisher_outra.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:?defina PY com o interpretador LOCAL desta maquina}
L=logs/fisher_outra_maquina.log; mkdir -p logs/fisher_vector; : > "$L"
diz() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

CFG="--max-side 512 --upscale 2 --stride 3 --bin-sizes 4,6,8,11,16,23,32 \
     --pca-dim 80 --gmm-k 256 --augment-xy --intra-norm --max-desc 0 \
     --fit-images 1500 --workers 16"

# ---------------------------------------------------------------------------
# FASE 1 -- os quatro de SPLIT OFICIAL. Ja sao indutivos por construcao: o
# metadado tem coluna `split` e o `--fit-split train` (default) ajusta PCA/GMM
# so no treino publicado. UMA extracao cada. ~2 h no total.
# ---------------------------------------------------------------------------
for e in "minc2500|data/minc2500/minc-2500/images" \
         "places_lt|data/places_lt/images" \
         "nct_crc|data/nct_crc/images/NCT-CRC-HE-100K"; do
  ds="${e%%|*}"; img="${e##*|}"; out="outputs/${ds}_fisher_vector_pub"
  [ -f "$out/features.npy" ] && { diz "$ds: pronto"; continue; }
  diz "FASE 1: $ds"
  $PY -u src/extractors/extract_fisher_vector_gpu.py --dataset "$ds" \
      --image-dir "$img" --out-dir "$out" $CFG \
      > "logs/fisher_vector/${ds}_pub.log" 2>&1
  diz "FASE 1: $ds rc=$?"
done

# A coorte de teste do nct_crc TEM de reusar o codebook do TREINO: ajustar PCA e
# GMM separadamente poria os dois lados do split em espacos de features
# diferentes, e NADA na saida denunciaria.
if [ ! -f outputs/nct_crc_val_fisher_vector_pub/features.npy ]; then
  if [ -f outputs/nct_crc_fisher_vector_pub/codebook.npz ]; then
    mkdir -p outputs/nct_crc_val_fisher_vector_pub
    cp outputs/nct_crc_fisher_vector_pub/codebook.npz \
       outputs/nct_crc_val_fisher_vector_pub/codebook.npz
    diz "FASE 1: nct_crc_val (codebook do TREINO copiado)"
    $PY -u src/extractors/extract_fisher_vector_gpu.py --dataset nct_crc_val \
        --image-dir data/nct_crc/images/CRC-VAL-HE-7K \
        --out-dir outputs/nct_crc_val_fisher_vector_pub $CFG \
        > logs/fisher_vector/nct_crc_val_pub.log 2>&1
    diz "FASE 1: nct_crc_val rc=$?"
  else
    diz "!! nct_crc_val sem o codebook do treino -- PULANDO"
  fi
fi

# ---------------------------------------------------------------------------
# FASE 2 -- os de CV, versao INDUTIVA. O metadado deles nao tem `split`, entao o
# default ajustaria o codebook na coorte INTEIRA: transdutivo. Nao e vazamento
# de rotulo (PCA e GMM nao veem classe), mas medimos que vale **+0,87pp sob
# sonda linear** no rijksdraw (p=0,0076, 5/5).
#
# Estritamente indutivo exige UM CODEBOOK POR FOLD e recodificar TODAS as
# imagens sob ele -- o treino tambem muda de espaco. Sao 5 extracoes por corpus;
# cachear descritores para evitar isso e impossivel (817k x 128 dims x 4 bytes =
# 418 MB POR IMAGEM).
#
# A particao e a do `evaluate_banks_dense_mlp.py`: StratifiedKFold com SEED
# CONSTANTE, nao a semente do laco. Por isso cada codebook serve as 5 sementes.
# ---------------------------------------------------------------------------
for e in "oilpainting|data/oilpainting/image" \
         "multitask|data/multitask_painting100k/original/images"; do
  ds="${e%%|*}"; img="${e##*|}"
  for k in 1 2 3 4 5; do
    out="outputs/${ds}_fisher_vector_ind_f${k}"
    [ -f "$out/features.npy" ] && { diz "$ds fold $k: pronto"; continue; }
    diz "FASE 2: $ds fold $k"
    $PY -u src/extractors/extract_fisher_vector_gpu.py --dataset "$ds" \
        --image-dir "$img" --out-dir "$out" $CFG \
        --fit-stems "data/${ds}_fold${k}_train_stems.txt" \
        > "logs/fisher_vector/${ds}_ind_f${k}.log" 2>&1
    diz "FASE 2: $ds fold $k rc=$?"
    # A sonda roda SO no fold k: o codebook so e honesto nele, e rodar os
    # outros quatro com ele seria transdutivo de novo, em silencio.
    $PY -u experiments/evaluate_banks_dense_mlp.py --datasets "$ds" \
        --bancos "Fisher vector" --arquiteturas denso-128 linear --folds "$k" \
        --fv-sufixo "_ind_f${k}" \
        --output "results/${ds}_banks_dense_mlp.csv" \
        >> "logs/banks_fvind_${ds}.log" 2>&1
    diz "FASE 2: $ds fold $k sonda rc=$?"
  done
done
diz "concluido"
