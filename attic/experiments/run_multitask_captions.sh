#!/usr/bin/env bash
set -u
cd /exp/duducampos/Paintings
for attempt in $(seq 1 40); do
  echo "=== [$(date +%H:%M:%S)] multitask tentativa $attempt" | tee -a logs/captions.log
  .ABSvenv/bin/python3 src/affect/generate_catr_coco_wikiart.py \
      --sample data/multitask_caption_sample.csv \
      --output outputs/affect_captions/catr_coco_multitask.jsonl \
      --batch-size 32 >> logs/captions.log 2>&1
  status=$?
  echo "=== [$(date +%H:%M:%S)] saiu com código $status" | tee -a logs/captions.log
  [ "$status" -eq 0 ] && break
  sleep 10
done
