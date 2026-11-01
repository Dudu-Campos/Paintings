#!/usr/bin/env bash
# ResNet treinada DO ZERO nos sete corpora.
#
# POR QUE. E a unica baseline com o MESMO orcamento de supervisao que o nosso
# banco: zero rotulo de fora do corpus (§4.1). A congelada carrega 1,28M de
# rotulos do ImageNet e a fine-tuned carrega esses mais os do alvo. Hoje a linha
# do zero existe so para o WikiArt (`outputs/cnn_scratch`, 42,34 contra os
# nossos 52,10), e as imagens daquele corpus nao existem mais -- entao as
# tabelas dos SETE nao tem o degrau de baixo da escada de supervisao.
#
# RECEITA, e ela difere do fine-tuning de proposito. Treino do zero precisa de
# mais orcamento e mais regularizacao; dar a ele a receita do fine-tuning seria
# subtreinar o adversario, que e a direcao errada (§9.9: "a mudanca fortalece a
# baseline, entao corre contra nos -- que e a direcao certa").
#   200 epocas          contra as 75 do fine-tuning
#   paciencia == epocas o cosseno roda inteiro e a epoca sai por BAcc de
#                       validacao; o §7.16 mede que paciencia curta briga com
#                       o cosseno (o MINC morria na epoca 16 com LR em 89%)
#   lr 0,1 @ batch 256  regra de escala linear de Goyal et al. 2017
#   wd 5e-4, FORA de BN e bias          |  gamma da ultima BN de cada bloco em 0
#   warmup 5 epocas  |  label smoothing 0,1  |  Mixup 0,2  |  AMP
#   --augment strong  a MESMA do fine-tuning, para a comparacao ficar de pe
#
# IMAGENS VEM DO NFS, nao de /tmp. As copias locais sumiram no reboot de 01/out
# e o NVMe esta 98% cheio (dados de outros usuarios); o unico disco com espaco
# e rotacional, pior que NFS para arquivo pequeno aleatorio. Medido aqui: a 1a
# epoca do oilpainting custa 157s lendo do NFS e as seguintes 11s, porque o
# page cache segura o corpus inteiro (115 GB livres contra 48 GB do maior).
#
# Uso:  bash experiments/run_scratch_grid_resilient.sh [--units N] [--dry-run]
set -u
cd "$(dirname "$0")/.."

LOCK=/tmp/prism_scratch_grid.lock
PY=${PY:-.venv-rtx3090-py312/bin/python3}
ROOT_OUT=outputs/scratch_grid
LOG_DIR=logs/scratch_grid
DRY=0
WAIT_FOR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --units) MAX_UNITS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --wait-for) WAIT_FOR="$2"; shift 2 ;;   # padrao de processo a esperar
    *) echo "flag desconhecida: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$ROOT_OUT" "$LOG_DIR"
if [ "$DRY" = "0" ]; then
  if ! mkdir "$LOCK" 2>/dev/null; then
    echo "=== lock $LOCK existe -- outra instancia no ar, saindo ==="; exit 0
  fi
  trap 'rmdir "$LOCK" 2>/dev/null' EXIT
fi

say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*" | tee -a "$LOG_DIR/driver.log"; }

# FONTE UNICA DAS FLAGS: extrai `dataset_args` do driver de fine-tuning em vez
# de copiar o `case`. Duplicar divergiria em silencio -- o §8 registra o caso
# (`evaluate_decision_fusion_official.py:148`, logica de split que divergiu
# entre dois produtores). So o --image-dir e trocado, logo abaixo.
ds_args_of() {
  bash -c "$(sed -n '/^dataset_args()/,/^}/p' experiments/run_finetune_grid_resilient.sh)
           dataset_args $1"
}

# Os originais no NFS. As copias de /tmp que o driver de fine-tuning usa nao
# existem mais nesta maquina.
nfs_dir_of() {
  case "$1" in
    oilpainting)  echo data/oilpainting/image ;;
    rijksdraw)    echo data/rijksmuseum/drawing ;;
    places_lt)    echo data/places_lt/images ;;
    minc2500)     echo data/minc2500/minc-2500/images ;;
    food101)      echo data/food101/food-101/images ;;
    nct_crc)      echo data/nct_crc/images ;;
    multitask125) echo data/multitask_painting100k/original/images ;;
    *) echo "sem diretorio NFS para $1" >&2; return 2 ;;
  esac
}

is_official() { case "$1" in minc2500|places_lt|food101|nct_crc) return 0 ;; *) return 1 ;; esac; }

# ARQUITETURA UNICA: resnet18 (decisao do usuario, 05/out). O argumento e que
# ninguem questiona uma rede PEQUENA em corpus pequeno -- a objecao natural a
# 15.829 imagens e que uma resnet50 do zero esta faminta de dado, nao o
# contrario. E os cinco corpora grandes sao 87% do custo, entao a troca corta a
# grade pela metade e e ela que torna o PROTOCOLO CHEIO possivel.
# Medido no oilpainting antes de decidir: rn18 54,17 contra rn50 53,58.
ARCH=${ARCH:-resnet18}

# PROTOCOLO IGUAL AO DO GRID DE FINE-TUNING (§9.1), que era a assimetria que o
# usuario apontou: la sao 5 folds x 5 sementes = 25 unidades nos corpora de CV e
# 1 particao x 5 sementes = 5 nos de split oficial. Reportar o do zero com UMA
# unidade contra aquelas 25 seria pior do que parece -- a CNN do zero esta
# GANHANDO de nos no oilpainting, e anunciar a vitoria de um adversario a partir
# de uma semente so nao se defende.
#   95 unidades, 185 h medidas (resnet18, 200 epocas, cache quente)

# LARGURA PRIMEIRO. O laco externo e o INDICE da unidade e o interno e o corpus,
# nao o contrario: assim uma interrupcao deixa todos os corpora na mesma
# profundidade em vez de alguns completos e outros zerados. Em 16 h ha 1 unidade
# de cada; em 81 h os quatro oficiais FECHAM e os de arte estao em 5 de 25.
# Corpora na ordem do mais barato para o mais caro dentro de cada passada.
CORPORA=${CORPORA:-"oilpainting rijksdraw minc2500 places_lt food101 multitask125 nct_crc"}
MAX_UNITS=${MAX_UNITS:-25}

if [ -n "$WAIT_FOR" ]; then
  say "esperando '$WAIT_FOR' liberar a GPU"
  while pgrep -f "$WAIT_FOR" | grep -qv "^$$\$"; do sleep 120; done
  say "liberou"
fi

for i in $(seq 0 $((MAX_UNITS - 1))); do
  for ds in $CORPORA; do
    if is_official "$ds"; then
      # 1 particao x 5 sementes; nao ha fold para variar
      [ "$i" -ge 5 ] && continue
      seed=$((42 + i)); fold=1
    else
      # 5 folds x 5 sementes: as 5 primeiras unidades sao os 5 folds da semente
      # 42, que cobrem a coorte inteira OOF -- o recorte mais util se parar ali
      seed=$((42 + i / 5)); fold=$((1 + i % 5))
    fi

    base_args="$(ds_args_of "$ds")" || { say "pulando $ds (sem args)"; continue; }
    nfs="$(nfs_dir_of "$ds")" || continue
    args="$(echo "$base_args" | sed -E "s#--image-dir [^ ]+#--image-dir $nfs#")"

    out="$ROOT_OUT/$ds/$ARCH/seed_${seed}/fold_${fold}"
    log="$LOG_DIR/${ds}_${ARCH}_s${seed}_f$(printf %02d "$fold").log"
    marker="$out/fold_$(printf %02d "$fold")_posteriors.npz"
    legado="$ROOT_OUT/$ds/$ARCH/seed_${seed}/fold_$(printf %02d "$fold")"
    if [ -d "$legado" ] && [ ! -d "$out" ]; then mv "$legado" "$out"; say "normalizado: $legado"; fi
    if [ -f "$marker" ]; then continue; fi

    cmd="$PY src/baselines/train_resnet_wikiart.py $args \
      --init scratch --model $ARCH --output $out \
      --seed $seed --only-fold $fold \
      --epochs 200 --patience 200 --batch-size 256 \
      --lr 0.1 --weight-decay 5e-4 --warmup-epochs 5 \
      --label-smoothing 0.1 --mixup 0.2 \
      --no-wd-on-norm --zero-init-residual \
      --augment strong --select-by bacc --require-cuda"

    if [ "$DRY" = "1" ]; then echo "--- [u$i] $ds/$ARCH s$seed f$fold"; continue; fi
    mkdir -p "$out"
    say "[unidade $i] $ds/$ARCH semente $seed fold $fold -> $log"
    eval "$cmd" > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "$ds" "$ARCH" "$seed" "$fold" "$rc" >> "$LOG_DIR/failures.tsv"
      say "FALHOU $ds/$ARCH s$seed f$fold (rc=$rc)"; tail -3 "$log"
    else
      say "ok $ds/$ARCH s$seed f$fold: $(grep -oE 'BAcc=[0-9.]+' "$log" | tail -1)"
    fi
  done
done
say "FIM DA GRADE DO ZERO"
