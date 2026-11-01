#!/usr/bin/env bash
# A noite de 12/out, na ordem pedida:
#   1) controle INDUTIVO do Fisher vector no rijksdraw
#   2) MLP denso-512 para os tres baselines nos sete corpora
#   3) fusao hand-8 + Fisher vector, com predicao em disco para a sobreposicao
set -u
cd /exp/duducampos/Paintings || exit 1
PY=.venv-rtx3090-py312/bin/python3
L=logs/noite_12out.log; : > "$L"
diz() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

# esperar o que estiver rodando (as 4 sementes do FV transdutivo)
while pgrep -u "$USER" -x python3 >/dev/null; do sleep 30; done
diz "maquina livre"

# ---------------------------------------------------------------- (1) INDUTIVO
# O codebook do cache `_pub` foi ajustado numa amostra da COORTE INTEIRA -- o
# metadado do rijksdraw nao tem coluna `split`. Isso e transdutivo (PCA e GMM
# nao veem rotulo) e so virou decisivo porque o FV EMPATOU com o hand-8 ali.
# Aqui o pool e so o treino do fold 1, e a diferenca mede o tamanho da
# transducao em vez de declara-la.
if [ ! -f outputs/rijksdraw_fisher_vector_ind/features.npy ]; then
  diz "(1) extracao INDUTIVA do rijksdraw"
  "$PY" -u src/extractors/extract_fisher_vector_gpu.py --dataset rijksdraw \
    --image-dir data/rijksmuseum/drawing \
    --out-dir outputs/rijksdraw_fisher_vector_ind \
    --max-side 512 --upscale 2 --stride 3 --bin-sizes 4,6,8,11,16,23,32 \
    --pca-dim 80 --gmm-k 256 --augment-xy --intra-norm --max-desc 0 \
    --fit-stems data/rijksdraw_fold1_train_stems.txt --workers 16 \
    > logs/fisher_vector/rijksdraw_ind.log 2>&1
  diz "(1) extracao rc=$?"
fi
# SO o fold 1: o codebook so e honesto nele. Rodar os outros quatro seria
# transdutivo de novo, e sem nada na saida denunciando -- por isso --folds 1.
diz "(1) sonda indutiva, fold 1, 5 sementes"
"$PY" -u experiments/evaluate_banks_dense_mlp.py --datasets rijksdraw \
  --bancos "Fisher vector" --arquiteturas denso-128 linear --folds 1 \
  --fv-sufixo _ind --output results/rijksdraw_banks_dense_mlp.csv \
  > logs/banks_dense_fvind_rijksdraw.log 2>&1
diz "(1) sonda rc=$?"

# ------------------------------------------------------------- (2) DENSO-512
# A contrapartida justa das branches do hand-8: o denso-128 comprime o
# WND-CHARM 22,6x contra 4,8x do hand-8, e nos dois corpora onde o denso-512 ja
# existia ele VENCE para eles (rijksdraw 48,97 contra 47,08; places_lt 8,35
# contra 5,62). Sem ele a regra "melhor configuracao de cada metodo" nao esta
# cumprida do lado deles.
for ds in oilpainting rijksdraw multitask nct_crc minc2500 food101 places_lt; do
  diz "(2) denso-512 $ds"
  "$PY" -u experiments/evaluate_banks_dense_mlp.py --datasets "$ds" \
    --bancos Haralick HOG WND-CHARM --arquiteturas denso-512 \
    --output "results/${ds}_banks_dense_mlp.csv" \
    >> "logs/banks_d512_${ds}.log" 2>&1
  diz "(2) denso-512 $ds rc=$?"
done

# ---------------------------------------------------------------- (3) FUSAO
# MESMA sonda nos dois lados, de proposito e ao contrario da regra da tabela
# principal: ali a pergunta e "quanto cada metodo entrega" e cada um leva a sua
# melhor cabeca; aqui e "eles erram nas mesmas imagens", e sonda diferente
# mediria a sonda.
diz "(3) fusao hand-8 + Fisher vector (e predicoes para a sobreposicao)"
"$PY" -u experiments/evaluate_banks_dense_mlp.py --datasets rijksdraw \
  --bancos hand-8 "Fisher vector" --fusoes "Fisher vector pub" \
  --arquiteturas denso-128 --folds 1 --fv-sufixo _pub \
  --pred-dir outputs/banks_mlp_pred \
  --output results/rijksdraw_banks_dense_mlp.csv \
  > logs/banks_fusao_rijksdraw.log 2>&1
diz "(3) fusao rc=$?"
diz "noite concluida"
