#!/usr/bin/env bash
# WND-CHARM tiled contra hand-8 nos QUATRO corpora de split publicado, agora com
# o protocolo publicado em vez de 5-fold CV.
#
# POR QUE REFAZER. As seis linhas medidas em 29-30/set rodaram
# `StratifiedKFold(5)` sobre a coorte inteira nos quatro -- e o bug que o §8
# registra no `evaluate_preprocess_effect_oilpainting` ("a constante OFFICIAL
# existia e nunca era lida"). O custo nao e cosmetico:
#   * places_lt: a CV derruba o desbalanceamento de 996x para 41x e apaga o
#     regime que motiva o corpus (§7.5);
#   * nct_crc:   a CV e INTERNA, e o §7.14 diz explicitamente que ela "nao e o
#     numero principal porque nao controla paciente/slide" -- o teste externo
#     CRC-VAL-HE-7K derruba o hand-8 de 99 para 87,6 (11pp).
#
# Saidas NOVAS (`_official.csv`): as de CV ficam em disco, porque o contraste
# entre os dois protocolos e informativo e ja esta pago.
#
# SEQUENCIAL: uma GPU. Espera o marginal WL do food101, que esta na frente.
# Lock de diretorio -- §8, nona vez.
set -u
cd "$(dirname "$0")/.."

LOCK=/tmp/prism_wndcharm_official.lock
PY=.venv-rtx3090-py312/bin/python3

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "=== lock $LOCK existe -- outra instancia no ar, saindo ==="; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }

# Espera por MARCADOR, nao por PID (§8: PID some num restart, e `pgrep -f` casa
# o proprio shell que o contem).
wait_marker() {  # $1 = marcador, $2 = log, $3 = padrao do processo
  while ! grep -q "$1" "$2" 2>/dev/null; do
    if ! pgrep -f "$3" | grep -qv "^$$\$"; then
      say "'$3' nao esta rodando e '$1' nao apareceu em $2 -- seguindo"; return 1
    fi
    sleep 60
  done
  return 0
}

say "esperando o marginal WL do food101 liberar a GPU"
wait_marker "WL real contra o PLACEBO" logs/wl_pipeline_food101.log \
            "evaluate_wl_pipeline_multitask.py --dataset food101" || true

say "esperando a extracao tiled do nct_crc_val"
wait_marker "gravado" logs/wndcharm_tiled_nct_crc_val.log \
            "extract_wndcharm_tiled_features.py --dataset nct_crc_val" || true

# Ordem: do mais barato ao mais caro, para que uma interrupcao deixe o maximo
# de linhas prontas. minc2500 ~50min, nct_crc ~1h40, food101 ~1h42,
# places_lt ~2h (365 classes).
for ds in minc2500 nct_crc food101 places_lt; do
  out=results/wndcharm_tiled_vs_hand8_${ds}_official.csv
  if [ -s "$out" ] && grep -q "hand-8 + WND-CHARM tiled (Fisher top15%)" "$out"; then
    say "$ds ja completo em $out -- pulando"; continue
  fi
  say "$ds: avaliando no split publicado"
  "$PY" experiments/evaluate_wndcharm_tiled_vs_hand8.py \
    --dataset "$ds" --tiled-dir "outputs/${ds}_wndcharm_tiled" \
    --output "$out" > "logs/wndcharm_official_${ds}.log" 2>&1
  rc=$?
  say "$ds terminou (rc=$rc), log em logs/wndcharm_official_${ds}.log"
  tail -3 "logs/wndcharm_official_${ds}.log"
done

say "FIM DA FILA OFICIAL"
