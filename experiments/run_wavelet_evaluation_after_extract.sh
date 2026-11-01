#!/usr/bin/env bash
# Wait for a complete restart-safe wavelet extraction before starting CV.
set -euo pipefail

while [[ ! -f outputs/hilbert_wavelet_pyramid_full/features.npy ]]; do
    sleep 30
done

exec .venv/bin/python experiments/evaluate_multiscale_graph_fourier_wavelet.py
