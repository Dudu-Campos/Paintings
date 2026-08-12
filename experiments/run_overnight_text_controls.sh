#!/usr/bin/env bash
set -euo pipefail

cd /exp/duducampos/Paintings

.venv-rtx3090-py312/bin/python src/slm/caption_inference.py \
  --image_dir wikiart \
  --model_dir outputs/caption_model \
  --output outputs/affect_captions/coco_wikiart_captions.jsonl \
  --batch_size 64 \
  --max_length 64 \
  --num_beams 1 \
  --fp16

.venv-rtx3090-py312/bin/python \
  src/affect/generate_prompt_ablation_wikiart_fast.py \
  --output outputs/affect_captions/qwen_prompt_ablation_500_per_style.jsonl \
  --sample_csv results/qwen_prompt_ablation_sample.csv \
  --max_per_style 500 \
  --batch_size 4 \
  --max_new_tokens 64
