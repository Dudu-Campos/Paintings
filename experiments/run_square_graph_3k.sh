#!/usr/bin/env bash
set -euo pipefail

cd /exp/duducampos/Paintings
PYTHON=.venv-rtx3090-py312/bin/python
SAMPLE=results/super_resolution_3k_sample.csv
FEATURES=results/square_graph_3k_features.parquet

"$PYTHON" experiments/extract_square_graph_features.py \
  --sample "$SAMPLE" --output "$FEATURES" --workers 32 --checkpoint-every 1000
"$PYTHON" experiments/analyze_line_graph_redundancy.py \
  --sample "$SAMPLE" --line-features "$FEATURES" \
  --output results/square_graph_3k_redundancy.csv
"$PYTHON" experiments/evaluate_square_graph_3k.py \
  --sample "$SAMPLE" --square-features "$FEATURES" \
  --line-features results/line_graph_3k_features.parquet \
  --output results/square_graph_3k_results.csv
"$PYTHON" experiments/rank_line_graph_3k_features.py \
  --sample "$SAMPLE" --line-features "$FEATURES" \
  --output results/square_graph_3k_feature_ranking.csv

echo "square-graph 3k pipeline complete"
