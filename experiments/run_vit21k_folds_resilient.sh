#!/usr/bin/env bash
# Folds 2-5 do ViT IN-21k com receita forte, em sequencia, depois que o fold 1
# terminar. Escrito como arquivo (nao encadeado no shell) pelo gotcha do §8 do
# CLAUDE.md, e a espera usa /proc em vez de pgrep pelo mesmo motivo.
#
# Retomavel: um fold cujo `fold_0N_posteriors.npz` ja existe e pulado, entao um
# SIGKILL de madrugada nao perde os folds anteriores. Cada fold tenta ate 3x.
#
# `fold_metrics.csv` e sobrescrito a cada invocacao do treinador (ele so conhece
# os folds da propria rodada), por isso o snapshot por fold e a agregacao final.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
OUT=outputs/multitask_vit21k_recipe_acc
LOG=logs/multitask_vit21k_folds.log
WAIT_PID=${1:-0}

say () { echo "=== [$(date '+%d/%b %H:%M:%S')] $*" | tee -a "$LOG"; }

if [ "$WAIT_PID" != 0 ]; then
  say "esperando o fold 1 (PID $WAIT_PID) terminar"
  while [ -d "/proc/$WAIT_PID" ]; do sleep 60; done
  say "fold 1 terminou"
fi
[ -f "$OUT/fold_metrics.csv" ] && cp "$OUT/fold_metrics.csv" "$OUT/fold_metrics_fold01.csv"

for fold in 2 3 4 5; do
  tag=$(printf '%02d' "$fold")
  if [ -f "$OUT/fold_${tag}_posteriors.npz" ]; then
    say "fold $fold ja existe, pulando"; continue
  fi
  for attempt in 1 2 3; do
    say "fold $fold, tentativa $attempt"
    $PY src/baselines/train_resnet_wikiart.py \
      --model vit_b_16_in21k --init imagenet \
      --image-dir /tmp/prism_data/multitask/images \
      --meta data/multitask_meta.csv --task style \
      --restrict-stems data/multitask_25class_cohort_stems.csv \
      --no-class-weight --select-by accuracy \
      --augment strong --label-smoothing 0.1 --weight-decay 0.05 \
      --lr 1e-4 --batch-size 64 --epochs 40 --patience 10 --workers 12 \
      --only-fold "$fold" --folds 5 --output "$OUT" \
      >> "logs/multitask_vit21k_fold${tag}.log" 2>&1
    if [ -f "$OUT/fold_${tag}_posteriors.npz" ]; then
      cp "$OUT/fold_metrics.csv" "$OUT/fold_metrics_fold${tag}.csv"
      say "fold $fold OK"; break
    fi
    say "fold $fold falhou na tentativa $attempt"; sleep 30
  done
done

say "agregando"
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, pandas as pd
fs = sorted(glob.glob("outputs/multitask_vit21k_recipe_acc/fold_metrics_fold*.csv"))
if fs:
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(subset="fold")
    d = d.sort_values("fold")
    d.to_csv("outputs/multitask_vit21k_recipe_acc/fold_metrics_all.csv", index=False)
    print(d.to_string(index=False))
    for m in ("accuracy", "balanced_accuracy", "macro_f1"):
        if m in d: print(f"{m:20s} {d[m].mean()*100:6.2f} +- {d[m].std(ddof=0)*100:4.2f}  (n={len(d)})")
PYEOF
say "FIM"
