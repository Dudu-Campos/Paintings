#!/usr/bin/env bash
# Controles da ablacao de captioner (§5.18): vazamento lexical e teto de vocabulario.
#
# O braco principal mediu Qwen ~28,7 contra 16,08 do estudante, sob o mesmo
# protocolo. Duas objecoes ficaram declaradas e este script as fecha:
#
#  1. VAZAMENTO. As stop words casam token EXATO, entao forma derivada escapa
#     (§5.18b). A auditoria contou `classical` 2.571 no Qwen contra 14 no
#     estudante, e `impressionistic` 648 contra ZERO. O modo `root` poda por
#     radical e erra para o lado de remover demais -- se o numero sobrevive
#     a esta poda, sobrevive a qualquer poda mais cirurgica.
#
#  2. TETO DE VOCABULARIO. O Qwen satura os 60.000 termos e o estudante realizou
#     31.163, entao a comparacao nao e pareada em orcamento. O braco de 31.163
#     da ao Qwen exatamente o vocabulario que o estudante usou.
#
# Sequencial de proposito: sao tres LinearSVC sobre ~100k linhas e disputar CPU
# entre eles invalidaria nada, mas atrasa os tres (§8: maquina quieta).
set -u
cd /exp/duducampos/Paintings
PY=.ABSvenv/bin/python3
DS=multitask

rodar () {                      # nome  legendas  modo  teto
  local nome="$1" caps="$2" modo="$3" teto="$4"
  local saida="results/text_scores_${nome}"
  if [ -f "${saida}/${DS}_decisions.npy" ]; then
    echo "== ${nome}: ja existe, pulando"
    return 0
  fi
  echo "== ${nome}: modo=${modo} teto=${teto}"
  $PY experiments/build_text_scores.py --dataset "$DS" \
      --captions "$caps" --output-dir "$saida" \
      --stopword-mode "$modo" --max-features "$teto" || {
        echo "FALHOU: ${nome}"; return 1; }
}

# 1. Qwen com poda por radical -- o controle de vazamento
rodar qwen_root    results/${DS}_qwen_captions.jsonl    root  60000
# 2. Estudante com a MESMA poda -- sem isso a comparacao deixa de ser pareada
rodar student_root results/${DS}_student_captions.jsonl root  60000
# 3. Qwen no orcamento de vocabulario do estudante
rodar qwen_cap31k  results/${DS}_qwen_captions.jsonl    exact 31163

echo "== fim"
