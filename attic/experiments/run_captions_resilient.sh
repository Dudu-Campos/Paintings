#!/usr/bin/env bash
# CATR v3 / MS COCO sobre os datasets novos. Retomável: o script pula o que
# já está no .jsonl de saída, então reiniciar após um SIGKILL não reprocessa.
set -u
cd /exp/duducampos/Paintings
run () {
  local name=$1 sample=$2 out=$3
  for attempt in $(seq 1 40); do
    echo "=== [$(date +%H:%M:%S)] $name tentativa $attempt" | tee -a logs/captions.log
    .ABSvenv/bin/python3 src/affect/generate_catr_coco_wikiart.py \
        --sample "$sample" --output "$out" --batch-size 32 >> logs/captions.log 2>&1
    status=$?
    echo "=== [$(date +%H:%M:%S)] $name saiu com código $status" | tee -a logs/captions.log
    [ "$status" -eq 0 ] && return 0
    sleep 10
  done
}
run oilpainting data/oilpainting_caption_sample.csv outputs/affect_captions/catr_coco_oilpainting.jsonl
run multitask   data/multitask_caption_sample.csv   outputs/affect_captions/catr_coco_multitask.jsonl
echo "=== [$(date +%H:%M:%S)] legendas concluidas" | tee -a logs/captions.log
