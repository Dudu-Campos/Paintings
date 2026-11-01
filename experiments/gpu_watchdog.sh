#!/usr/bin/env bash
# Captura o que a GPU estava fazendo nos minutos ANTES de cair.
#
# POR QUE EXISTE. O §8 registra que quando a RTX 3090 cai o torch volta para a
# CPU e o job CONTINUA, produzindo numeros validos ~14x mais lentos, sem nada no
# CSV denunciando -- em 12/out foram 67 celulas do multitask calculadas na CPU.
# E o diagnostico canonico da NVIDIA (os erros Xid) vai para o `kern.log`, que
# o usuario NAO consegue ler: ele nao esta no grupo `adm`. Entao a unica
# evidencia obtivel sem root e telemetria amostrada de fora, e e isso aqui.
#
# A HIPOTESE QUE ELE EXISTE PARA TESTAR: o usuario resolve a queda **tirando e
# recolocando o cabo da GPU**. Isso aponta para contato/alimentacao, nao para
# driver -- e a assinatura seria a queda acontecer num PICO DE POTENCIA. A 3090
# e 350 W nominais com transientes muito acima disso, entao `power.draw.instant`
# nos segundos anteriores e a medida que decide. Se a queda vier em potencia
# baixa, a hipotese do cabo de forca cai e sobra slot/riser ou driver.
#
# JA DESCARTADO, antes de escrever isto:
#   Runtime D3 ............ "Disabled by default"  -> nao e suspensao do driver
#   HW Thermal Slowdown ... Not Active             -> nao e teto termico (em idle)
#   ECC / Retired Pages ... sem pendencia          -> nao e memoria degradada
#
# COMO USAR (nao precisa de root, nao precisa de GPU livre):
#   setsid nohup bash experiments/gpu_watchdog.sh > /dev/null 2>&1 < /dev/null &
#
# Ele grava em logs/gpu_watchdog/:
#   telemetria.csv       amostra continua, rotacionada
#   incidente_<data>/    congelado QUANDO CAI: a janela anterior + tudo que
#                        der para ler no momento da falha
set -u

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAIDA="${GPU_WATCHDOG_DIR:-$RAIZ/logs/gpu_watchdog}"
PY="${PY:-$RAIZ/.venv-rtx3090-py312/bin/python3}"
INTERVALO="${GPU_WATCHDOG_INTERVALO:-2}"      # amostra de telemetria, segundos
# A SONDA E CARA EM POTENCIA, nao em tempo: inicializar o CUDA leva a placa de
# ~29 W para ~118 W por alguns segundos (medido). Se a hipotese sob teste e
# transiente de alimentacao, sondar de minuto em minuto injeta justamente o
# estimulo que se quer observar. Por isso ela e RARA (5 min) e quem detecta
# rapido e o proprio nvidia-smi, que e de graca e ja roda a cada 2 s.
PROBE_CADA="${GPU_WATCHDOG_PROBE:-300}"       # sonda de CUDA, segundos
JANELA=900                                     # segundos de historico congelados
mkdir -p "$SAIDA"
CSV="$SAIDA/telemetria.csv"

CAMPOS=timestamp,power.draw.instant,power.draw,temperature.gpu,fan.speed,\
clocks.sm,clocks.mem,utilization.gpu,memory.used,pstate,\
pcie.link.gen.current,pcie.link.width.current,\
clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_power_brake_slowdown,\
clocks_event_reasons.sw_power_cap

[ -s "$CSV" ] || echo "$CAMPOS,sonda" > "$CSV"

log() { echo "[$(date '+%F %T')] $*" >> "$SAIDA/watchdog.log"; }

# A SONDA roda em SUBPROCESSO, de proposito, por duas razoes. Se a GPU cair, a
# falha de CUDA mata o filho e nao o vigia. E assim ela nao segura um contexto
# CUDA: segurar manteria a placa fora do P8 o tempo todo, mudando a potencia de
# repouso -- ou seja, mudando justamente o sistema sob suspeita.
sonda() {
    timeout 90 "$PY" - <<'PYEOF' 2>&1 | tail -1
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("MORTA:cuda_indisponivel"); sys.exit(0)
    torch.zeros(8, device="cuda").sum().item()
    print("viva")
except RuntimeError as e:
    m = str(e).lower()
    # ficar sem memoria significa que a GPU esta VIVA e ocupada -- tratar isso
    # como queda dispararia incidente a cada job grande nosso.
    print("viva:sem_memoria" if "out of memory" in m else f"MORTA:{str(e)[:120]}")
except Exception as e:
    print(f"MORTA:{type(e).__name__}:{str(e)[:120]}")
PYEOF
}

incidente() {
    local motivo="$1"
    local dir="$SAIDA/incidente_$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$dir"
    log "!! INCIDENTE: $motivo -> $dir"
    echo "$motivo" > "$dir/motivo.txt"
    date '+%F %T %z' > "$dir/quando.txt"
    uptime > "$dir/uptime.txt" 2>&1
    # a JANELA ANTERIOR, que e o ponto de tudo isto
    tail -n $((JANELA / INTERVALO + 60)) "$CSV" > "$dir/telemetria_antes.csv"
    # e tudo que ainda der para ler NO MOMENTO da falha
    nvidia-smi               > "$dir/nvidia-smi.txt"       2>&1
    nvidia-smi -q            > "$dir/nvidia-smi-q.txt"     2>&1
    cat /proc/driver/nvidia/gpus/*/information \
                             > "$dir/driver_info.txt"      2>&1
    cat /proc/driver/nvidia/gpus/*/power \
                             > "$dir/driver_power.txt"     2>&1
    lspci -nn | grep -i nvidia > "$dir/lspci.txt"          2>&1
    # os dois abaixo quase certamente falham sem root; ficam registrados COM a
    # mensagem de erro, que e o que prova que falta permissao e nao evidencia.
    dmesg 2>&1 | tail -200    > "$dir/dmesg.txt"
    journalctl -k -n 200 --no-pager 2>&1 | tail -200 > "$dir/journal.txt"
    # os jobs nossos que estavam rodando, para casar a queda com a carga
    ps -eo pid,etime,pcpu,rss,cmd 2>/dev/null | grep -E "python3" | grep -v grep \
                             > "$dir/processos.txt"
}

VEREDITO="$SAIDA/.veredito"
EM_CURSO="$SAIDA/.sondando"
rm -f "$VEREDITO" "$EM_CURSO"
trap 'rm -f "$VEREDITO" "$EM_CURSO"; log "watchdog encerrado"; exit 0' TERM INT

log "watchdog iniciado (intervalo ${INTERVALO}s, sonda a cada ${PROBE_CADA}s, PY=$PY)"
ultimo_probe=0
estado="viva"
while true; do
    agora=$(date +%s)
    linha=$(nvidia-smi --query-gpu="$CAMPOS" --format=csv,noheader,nounits 2>&1 | head -1)

    # DETECTOR RAPIDO, de graca: quando a placa cai do barramento o nvidia-smi
    # responde "No devices were found" -- foi literalmente o que o §8 registrou
    # em 12/out. Ele nao substitui a sonda (ha o caso de GPU presente com
    # contexto CUDA quebrado, o `cudaErrorLaunchFailure` que veio ANTES naquele
    # episodio), mas pega o caso duro em 2 s em vez de em 5 min.
    if [[ "$linha" =~ (No\ devices|Unable\ to\ determine|fallen\ off\ the\ bus|NVML|Failed\ to\ initialize) ]]; then
        if [[ "$estado" == "viva" ]]; then
            estado="morta"
            incidente "nvidia-smi: $linha"
        fi
        echo "$(date '+%Y/%m/%d %H:%M:%S.000'),,,,,,,,,,,,,,,SMI_FALHOU" >> "$CSV"
        sleep "$INTERVALO"; continue
    fi

    # A SONDA E ASSINCRONA. Sincrona ela custava ~5s (import do torch) e
    # BLOQUEAVA a amostragem -- a telemetria saia com buracos de 5s a cada
    # ciclo, e um buraco desses pode cair exatamente em cima da queda, que e a
    # unica janela que este script existe para preservar. Agora ela escreve o
    # veredito num arquivo e o laco le sem esperar.
    marca="-"
    if [ -f "$VEREDITO" ]; then
        r=$(cat "$VEREDITO"); rm -f "$VEREDITO"
        marca="${r%%:*}"
        if [[ "$r" == MORTA* ]]; then
            if [[ "$estado" == "viva" ]]; then
                estado="morta"
                incidente "$r"
            fi
        elif [[ "$estado" == "morta" ]]; then
            estado="viva"
            log "GPU voltou (${r}) -- provavelmente alguem recolocou o cabo"
        fi
    fi
    if (( agora - ultimo_probe >= PROBE_CADA )) && [ ! -f "$EM_CURSO" ]; then
        ultimo_probe=$agora
        ( : > "$EM_CURSO"; sonda > "$VEREDITO.tmp" 2>&1
          mv -f "$VEREDITO.tmp" "$VEREDITO"; rm -f "$EM_CURSO" ) &
    fi
    echo "$linha,$marca" >> "$CSV"

    # rotacao simples: o CSV nunca passa de ~200 MB
    if [ "$(stat -c%s "$CSV" 2>/dev/null || echo 0)" -gt 200000000 ]; then
        mv "$CSV" "$CSV.1"; echo "$CAMPOS,sonda" > "$CSV"
    fi
    sleep "$INTERVALO"
done
