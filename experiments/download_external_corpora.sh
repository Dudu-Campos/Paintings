#!/usr/bin/env bash
# Rebaixa MINC-2500 e NCT-CRC PARA O NFS.
#
# POR QUE: os dois foram baixados direto para /tmp/prism_data em 26/ago e nunca
# replicados no NFS. Dois reboots (04/set e a queda de luz de 05/set) limparam o
# /tmp e levaram as imagens junto -- as features sobreviveram, mas a capacidade
# de extrair QUALQUER COISA NOVA (legendas, o ViT congelado de nct_crc_val,
# outra resolucao) foi perdida. O §1 manda "tars originais no NFS"; isso nao foi
# seguido para estes dois.
#
# `--continue` em tudo: sao 15 GB e a maquina ja caiu duas vezes esta semana.
set -u
cd "$(dirname "$0")/.."

get() {  # url destino
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "  ja existe: $dest"; return; fi
  echo "=== [$(date '+%F %T')] baixando $(basename "$dest")"
  # `--retry-all-errors` e obrigatorio: sem ele o `--retry` NAO cobre o erro 18
  # (transfer closed), que foi como o opensurfaces derrubou o minc a 55 MB do
  # fim. `--continue-at -` retoma do .part em vez de recomecar os 2,2 GB.
  for attempt in 1 2 3 4 5; do
    curl -L --continue-at - --retry 5 --retry-delay 10 --retry-all-errors \
         --fail -o "$dest.part" "$url" && { mv "$dest.part" "$dest"; return; }
    echo "  tentativa $attempt falhou; retomando em 15s"
    sleep 15
  done
  echo "  DESISTIU: $dest"
}

# NCT-CRC: a versao CANONICA (com normalizacao de coloracao), que e a nomeada no
# §7.14. NAO baixar a NONORM -- features extraidas dela nao casariam com as que
# ja estao em disco.
get "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip?download=1" \
    data/nct_crc/archive/NCT-CRC-HE-100K.zip
get "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1" \
    data/nct_crc/archive/CRC-VAL-HE-7K.zip
get "http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz" \
    data/minc2500/minc-2500.tar.gz

echo "=== [$(date '+%F %T')] FIM"
ls -lh data/nct_crc/archive/ data/minc2500/ 2>/dev/null
