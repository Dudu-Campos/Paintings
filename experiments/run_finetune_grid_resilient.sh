#!/usr/bin/env bash
# Fine-tuning grid for the final CNN baselines.
#
# Canonical design:
#   4 CV datasets x 2 models x 5 folds x 5 seeds = 200 runs
#   4 official-split datasets x 2 models x 1 split x 5 seeds = 40 runs
#   total = 240 model trainings
#   models = ResNet50 ImageNet-1k and ViT-B/16 ImageNet-1k, both fine-tuned
#   no VGG, no Pandora, no Text.
#
# Each fold/split is a separate resumable run. A finished run is identified by
# fold_01_posteriors.npz ... fold_05_posteriors.npz for CV datasets, or only
# fold_01_posteriors.npz for datasets with official train/val/test splits.
#
# Examples:
#   bash experiments/run_finetune_grid_resilient.sh --dry-run
#   setsid nohup bash experiments/run_finetune_grid_resilient.sh --gpus 0,1 \
#       > logs/finetune_grid_driver.log 2>&1 < /dev/null &

set -u
cd "$(dirname "$0")/.."

PY=${PY:-.ABSvenv/bin/python3}
ROOT_OUT=${ROOT_OUT:-outputs/final_finetune_grid}
LOG_DIR=${LOG_DIR:-logs/final_finetune_grid}
DRY_RUN=0
GPUS="0"
DATASETS="multitask25 multitask125 oilpainting rijksdraw food101 places_lt nct_crc minc2500"
MODELS="resnet50 vit_b_16"
SEEDS="42 43 44 45 46"
ATTEMPTS=${ATTEMPTS:-3}

usage() {
  sed -n '1,22p' "$0"
  echo
  echo "Options:"
  echo "  --dry-run"
  echo "  --gpus 0,1"
  echo "  --datasets \"multitask25 minc2500\""
  echo "  --models \"resnet50 vit_b_16\""
  echo "  --seeds \"42 43\""
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --gpus) GPUS=$2; shift 2 ;;
    --datasets) DATASETS=$2; shift 2 ;;
    --models) MODELS=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$ROOT_OUT" "$LOG_DIR"

say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*" | tee -a "$LOG_DIR/driver.log"; }


dataset_args() {
  case "$1" in
    multitask25)
      echo "--image-dir /tmp/prism_data/multitask_painting100k/images --meta data/multitask_meta.csv --task style --min-per-class 50 --restrict-stems data/multitask_25class_cohort_stems.csv --folds 5"
      ;;
    multitask125)
      # min-per-class 5 preserva as 125 classes (a menor tem 10 imagens) e as
      # 99.816 do §4; o 50 antigo cortava 36 classes e virava tarefa de 89.
      echo "--image-dir /tmp/prism_data/multitask_painting100k/images --meta data/multitask_meta.csv --task style --min-per-class 5 --folds 5"
      ;;
    oilpainting)
      echo "--image-dir data/oilpainting/image --meta data/oilpainting_meta.csv --task style --min-per-class 50 --folds 5"
      ;;
    rijksdraw)
      # data/rijksdraw_labels.csv tem 1 linha (RP-T-1940-571) cujo .jpg no disco
      # é na verdade uma página HTML de erro salva pelo download original --
      # confirmado idêntico no NFS e na cópia local (mesmo md5), não é corrupção
      # da cópia. data/rijksdraw_labels_clean.csv é o original menos essa linha
      # (48.295, ainda 11 classes); o original não foi tocado.
      echo "--image-dir /tmp/prism_data/rijksdraw/drawing --meta data/rijksdraw_labels_clean.csv --task period --min-per-class 50 --folds 5"
      ;;
    food101)
      echo "--image-dir /tmp/prism_data/food-101/images --meta data/food101_meta.csv --task style --min-per-class 5 --official-split"
      ;;
    places_lt)
      echo "--image-dir /tmp/prism_data/places_lt/images --meta data/places_lt_meta.csv --task style --min-per-class 5 --official-split"
      ;;
    nct_crc)
      echo "--image-dir /tmp/prism_data/nct_crc --meta data/nct_crc_meta.csv --task style --min-per-class 5 --official-split"
      ;;
    minc2500)
      echo "--image-dir /tmp/prism_data/minc/minc-2500/images --meta data/minc2500_meta.csv --task style --min-per-class 5 --official-split"
      ;;
    *) echo "unknown dataset: $1" >&2; return 2 ;;
  esac
}

# PACIENCIA == EPOCAS, desde 02/set (§7.16). O `CosineAnnealingLR` planeja a
# descida sobre o orcamento inteiro, mas a paciencia cortava antes: o MINC morria
# na epoca 16 com o LR ainda em ~89% do inicial, e o multitask125 parou em 17,8
# de 75 (ResNet) e 10,1 de 40 (ViT). O platô intermediario e caracteristica do
# agendamento, nao sinal de convergencia -- o Food-101 provou isso ao ganhar
# +2,07pp DEPOIS do plato, e sobreviveu por acaso (uma melhora de 0,03pp zerou o
# contador).
#
# Isto NAO desliga a selecao de epoca: o trainer continua guardando o melhor
# estado por BAcc de validacao. So deixa o cosine terminar.
#
# ASSIMETRIA A DECLARAR NO ARTIGO: food101, minc2500, nct_crc, oilpainting e
# multitask125 rodaram com paciencia 12/10 e NAO foram refeitos. A mudanca torna
# o backbone mais forte, entao ela e conservadora (corre contra nos), mas os
# numeros de backbone deixam de ser comparaveis entre datasets ate que os cinco
# sejam refeitos.
model_args() {
  case "$1" in
    resnet50)
      echo "--model resnet50 --init imagenet --augment strong --optimizer sgd --lr 0.01 --weight-decay 0.0001 --batch-size 128 --epochs 75 --patience 75 --workers 12"
      ;;
    vit_b_16)
      echo "--model vit_b_16 --init imagenet --augment strong --optimizer adamw --lr 0.0001 --weight-decay 0.05 --label-smoothing 0.1 --batch-size 64 --epochs 40 --patience 40 --workers 12"
      ;;
    *) echo "unknown model: $1" >&2; return 2 ;;
  esac
}

is_official() {
  case "$1" in
    food101|places_lt|nct_crc|minc2500) return 0 ;;
    *) return 1 ;;
  esac
}

folds_for() {
  if is_official "$1"; then echo "1"; else echo "1 2 3 4 5"; fi
}

gpu_list() {
  echo "$GPUS" | tr ',' ' '
}

run_one() {
  local dataset=$1 model=$2 seed=$3 fold=$4 gpu=$5
  local out="$ROOT_OUT/${dataset}/${model}/seed_${seed}/fold_${fold}"
  local tag
  tag=$(printf "%s_%s_seed%s_fold%02d" "$dataset" "$model" "$seed" "$fold")
  local log="$LOG_DIR/${tag}.log"
  local marker
  marker=$(printf "%s/fold_%02d_posteriors.npz" "$out" "$fold")

  if [ -f "$marker" ]; then
    say "skip $tag"
    return 0
  fi

  mkdir -p "$out"
  local args ds_args mdl_args
  ds_args=$(dataset_args "$dataset") || return $?
  mdl_args=$(model_args "$model") || return $?
  args="$ds_args $mdl_args --seed $seed --only-fold $fold --output $out"

  if [ "$DRY_RUN" = "1" ]; then
    say "dry-run $tag"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/baselines/train_resnet_wikiart.py $args --dry-run > "$log" 2>&1
    return $?
  fi

  for attempt in $(seq 1 "$ATTEMPTS"); do
    say "start $tag on gpu=$gpu attempt=$attempt"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/baselines/train_resnet_wikiart.py $args >> "$log" 2>&1
    if [ -f "$marker" ]; then
      say "ok $tag"
      return 0
    fi
    say "failed $tag attempt=$attempt"
    sleep 30
  done
  say "GAVE UP $tag"
  return 1
}

wait_slot() {
  local max_jobs=$1
  while [ "$(jobs -pr | wc -l)" -ge "$max_jobs" ]; do
    wait -n
  done
}

main() {
  local gpus
  gpus=($(gpu_list))
  local max_jobs=${#gpus[@]}
  local next_gpu=0
  : > "$LOG_DIR/failures.tsv"
  say "grid datasets=[$DATASETS] models=[$MODELS] seeds=[$SEEDS] gpus=[$GPUS] dry_run=$DRY_RUN"

  for dataset in $DATASETS; do
    for model in $MODELS; do
      for seed in $SEEDS; do
        for fold in $(folds_for "$dataset"); do
          wait_slot "$max_jobs"
          gpu=${gpus[$next_gpu]}
          next_gpu=$(( (next_gpu + 1) % max_jobs ))
          {
            run_one "$dataset" "$model" "$seed" "$fold" "$gpu" || \
              echo -e "${dataset}\t${model}\t${seed}\t${fold}" >> "$LOG_DIR/failures.tsv"
          } &
        done
      done
    done
  done
  wait
  if [ -s "$LOG_DIR/failures.tsv" ]; then
    say "FALHAS:"
    cat "$LOG_DIR/failures.tsv" | tee -a "$LOG_DIR/driver.log"
    return 1
  fi
  say "FIM"
}

main
