#!/usr/bin/env bash
# Grade CHEIA de theta, nos DOIS lados, no MESMO instrumento.
#
# Substitui o desenho de `melhor - theta*` do §5.17. Critica do usuario em
# 10/set: robustez e propriedade da DISTRIBUICAO, nao do melhor -- e o melhor
# ainda era escolhido na propria medicao reportada, entao era um teto enviesado.
#
# Duas coisas que o controle de 10/set descobriu e que este script existe para
# consertar:
#   1. `theta_*_mlp.csv` usa o MLP DENSO (`fit` de evaluate_redundancy_selection
#      _curve, §5.9), nao o BranchMLP do pipeline -- 2,3pp de diferenca no
#      Hilbert em theta*. As varreduras usam sonda LDA. Eram TRES instrumentos.
#   2. Interpolar a grade a partir das 3-4 ancoras do pipeline SUBESTIMA a
#      dispersao em 3x (sd 0,595 contra 1,773 real) e ainda encolhe a barra de
#      erro aparente de 41% para 16%. Numero errado com cara de preciso.
#
# `--isolated` roda a MESMA view sozinha pelo MESMO `evaluate_mlp`.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.ABSvenv/bin/python3}

HILBERT=""; for n in 32 64 128 256 512; do for b in 4 8 16 32; do HILBERT="$HILBERT ${n}x${b}"; done; done
GABOR="";   for t in 32 64 128 256 512; do for n in 2 4 6 8;   do GABOR="$GABOR ${t}x${n}";     done; done
FOURIER=""; for s in 64 128 256 512;    do for b in 6 12 18 24; do FOURIER="$FOURIER ${s}x${b}"; done; done

corre() {  # nome script args...
  local rot=$1 tag=$2 minimo=$3; shift 3
  local out="results/theta_grid_${rot}.csv"
  if [ -f "$out" ] && [ "$(( $(wc -l < "$out") - 1 ))" -ge "$minimo" ]; then
    echo "[skip] $rot ja completo"; return
  fi
  echo "=== $rot  ($(date +%H:%M:%S)) ==="
  $PY "experiments/$tag" "$@" --folds 5 --output "$out" \
    || echo "FALHOU: $rot" | tee -a logs/theta_grid_failures.txt
}

for lado in pipeline isolated; do
  FLAG=""; [ "$lado" = isolated ] && FLAG="--isolated"
  corre "hilbert_$lado"  evaluate_theta_hilbert_pipeline.py  105 --thetas $HILBERT $FLAG
  corre "gabor_$lado"    evaluate_theta_gabor_pipeline.py    105 --thetas $GABOR   $FLAG
  corre "fourier_$lado"  evaluate_theta_fourier_pipeline.py   85 --thetas $FOURIER $FLAG
  corre "slic_$lado"     evaluate_theta_segmenters_pipeline.py 30 --method slic         $FLAG
  corre "felz_$lado"     evaluate_theta_segmenters_pipeline.py 30 --method felzenszwalb $FLAG
done
echo "=== fim ($(date +%H:%M:%S)) ==="
