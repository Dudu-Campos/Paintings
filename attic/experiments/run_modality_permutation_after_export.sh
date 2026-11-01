#!/usr/bin/env bash
# The embedding exporter writes one complete series encoder/cache per outer fold.
set -euo pipefail

for fold in 1 2 3 4 5; do
    while [[ ! -f "outputs/modality_embedding_cache/series_cnn/embeddings/fold_${fold}_series_embedding.npz" ]]; do
        sleep 30
    done
done

exec .venv/bin/python experiments/permutation_cached_modality_embeddings.py --permutations 100
