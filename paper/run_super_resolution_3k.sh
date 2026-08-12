#!/usr/bin/env bash
set -euo pipefail

cd /exp/duducampos/Paintings
PYTHON=.venv-rtx3090-py312/bin/python
ROOT=outputs/super_resolution_3k
SAMPLE=paper/super_resolution_3k_sample.csv

"$PYTHON" paper/prepare_super_resolution_pilot.py \
  --sample "$SAMPLE" \
  --output-root "$ROOT"

pids=()
for condition in normalized_1024 bicubic_x2 realesrgan_x2; do
  "$PYTHON" src/pipeline/slic_multigraph_features.py \
    --image-dir "$ROOT/$condition" \
    --manifest "$ROOT/${condition}_manifest.csv" \
    --output-dir "$ROOT/features/$condition/slic" \
    --workers 16 \
    --checkpoint-every 250 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
for condition in normalized_1024 bicubic_x2 realesrgan_x2; do
  "$PYTHON" src/baselines/extract_descriptors_full.py \
    --image_dir "$ROOT/$condition" \
    --manifest "$ROOT/${condition}_manifest.csv" \
    --output "$ROOT/features/$condition" \
    --workers 8 &
  pids+=("$!")
  "$PYTHON" src/baselines/extract_hilbert_features.py \
    --image_dir "$ROOT/$condition" \
    --manifest "$ROOT/${condition}_manifest.csv" \
    --output "$ROOT/features/$condition" \
    --workers 8 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" paper/evaluate_super_resolution_pilot.py \
  --sample "$ROOT/sample.csv" \
  --feature-root "$ROOT/features" \
  --output paper/super_resolution_3k_results.csv

"$PYTHON" paper/evaluate_super_resolution_branches.py \
  --root "$ROOT" \
  --output paper/super_resolution_3k_branch_results.csv

echo "super-resolution 3k pipeline complete"
