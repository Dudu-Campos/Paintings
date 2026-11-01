#!/usr/bin/env bash
# Reproduz a queda pelo gatilho REAL: a SAIDA do processo, nao a carga.
#
# A telemetria de 14/out (logs/gpu_watchdog/incidente_20261014_141800/ANALISE.md)
# mostra a placa sobrevivendo 40 min a 350 W, 82-83 C, `sw_power_cap` ativo --
# e morrendo 10 s DEPOIS que a carga parou, em repouso a ~46 W, 1 a 3 s apos o
# link PCIe completar Gen 4 -> Gen 2 -> Gen 1. Depois disso o `lspci` ainda
# enumera a placa e o config space responde, mas `current_link_speed` le
# "Unknown": quem colapsou foi o LINK, nao a alimentacao.
#
# Logo o evento letal e a transicao para o repouso profundo, e ela so acontece
# quando o ULTIMO cliente CUDA sai (sem persistence mode o driver comeca a
# desinicializar ai). Cada ciclo aqui e: carregar um pouco, SAIR, esperar a
# placa chegar em P8/Gen1, repetir. Se a hipotese estiver certa ela morre em
# poucos minutos em vez de 40, e isso torna o teste utilizavel como BANCADA
# para validar um conserto.
#
# USO (com o watchdog ja no ar):
#   setsid nohup bash experiments/gpu_stress_ciclos.sh > /dev/null 2>&1 < /dev/null &
#
# VALIDAR UM CONSERTO: aplicar um dos tres abaixo e rodar isto de novo. Se
# sobreviver a 20 ciclos onde antes morria em 3, o conserto e esse.
#   sudo nvidia-smi -pm 1                  persistence mode: o driver nao descarrega
#   BIOS: PCIe -> Gen 3                    o remedio padrao para falha de
#                                          treinamento de link com riser
#   pcie_aspm=off na linha do kernel       desliga o gerenciamento de energia do link
set -u
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$RAIZ/.venv-rtx3090-py312/bin/python3}"
CICLOS="${CICLOS:-20}"
CARGA="${CARGA:-45}"        # segundos de carga por ciclo
REPOUSO="${REPOUSO:-30}"    # segundos em repouso -- tem de passar de P8/Gen1
LOG="$RAIZ/logs/gpu_watchdog/ciclos.log"
mkdir -p "$(dirname "$LOG")"

diz() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
estado() {
    nvidia-smi --query-gpu=pstate,pcie.link.gen.current,power.draw.instant \
               --format=csv,noheader,nounits 2>&1 | head -1
}

diz "=== ciclos iniciados: $CICLOS x (carga ${CARGA}s + repouso ${REPOUSO}s) ==="
for i in $(seq 1 "$CICLOS"); do
    diz "ciclo $i/$CICLOS · carga"
    if ! "$PY" "$RAIZ/experiments/gpu_stress.py" \
            --degraus 8192 --dwell "$CARGA" --ciclos 0 >> "$LOG" 2>&1; then
        diz "!! MORREU DURANTE A CARGA no ciclo $i -- estado: $(estado)"
        exit 1
    fi
    # A ESPERA E O EXPERIMENTO. Sem ela a placa nao chega a P8/Gen1 e o ciclo
    # nao testa nada: o gatilho e justamente o rebaixamento completo.
    sleep "$REPOUSO"
    e=$(estado)
    diz "ciclo $i/$CICLOS · repouso · $e"
    if [[ "$e" =~ (No\ devices|Unable|NVML) ]]; then
        diz "!! MORREU NO REPOUSO depois do ciclo $i -- e o gatilho previsto"
        exit 1
    fi
done
diz "=== sobreviveu a $CICLOS ciclos ==="
diz "se antes morria em poucos ciclos, o conserto aplicado e o responsavel"
