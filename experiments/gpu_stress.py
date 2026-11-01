#!/usr/bin/env python3
"""Reproduz a queda da GPU de proposito, com rampa de potencia instrumentada.

Pedido do usuario em 14/out: *"vamos rodar o fine-tunning da cnn so para
estressar a maquina e derrubar outra vez"* -- com o `gpu_watchdog.sh` ja no ar,
a reproducao deliberada e o experimento certo.

**POR QUE NAO E O FINE-TUNING.** Nenhum corpus tem copia local (o `/` esta 100%
cheio) e ler do NFS da ~16 arquivos/s (§1), entao o treino ficaria limitado por
I/O com a GPU OCIOSA -- estressaria a rede, nao a placa. Carga sintetica nao
precisa de dado nenhum e peg a a placa no teto de potencia em segundos.

**E ELA RESPONDE MAIS QUE "caiu".** A hipotese sob teste e alimentacao/contato
-- o usuario conserta tirando e recolocando o cabo. Ela se distingue das outras
por ONDE a queda acontece:

    FASE 1, carga sustentada crescente   -> se cair aqui, e teto de POTENCIA
                                            (a 3090 e 350 W nominais) e o limiar
                                            fica registrado em watts
    FASE 2, liga-desliga a cada 2 s      -> se so cair aqui, e TRANSIENTE: o
                                            dI/dt de ligar e desligar e o que
                                            mata contato marginal, e carga
                                            constante e mais BENIGNA que isso
    nao cair em nenhuma                  -> a causa nao e carga; sobra slot,
                                            riser, ou reinicializacao do driver
                                            (hoje `Persistence Mode: Disabled`)

O `gpu_watchdog.sh` grava a telemetria; este script so produz o estimulo e
anota em que degrau estava quando morreu.

RESSALVA HONESTA: derrubar de proposito nao e de graca. Se a causa for contato
no conector, cada queda e um evento de arco eletrico ali, e repetir degrada. O
usuario ja derruba isto rotineiramente, entao o custo marginal e baixo -- mas
UMA reproducao com instrumento vale mais que cinco sem, e o objetivo e parar na
primeira.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def watts() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw.instant",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().split("\n")[0])
    except Exception:
        return float("nan")


def log(m: str, arq: Path) -> None:
    linha = f"[{time.strftime('%F %T')}] {m}"
    print(linha, flush=True)
    with arq.open("a") as f:
        f.write(linha + "\n")


def queima(n: int, segundos: float, duty: float = 1.0) -> float:
    """Matmuls FP16 por `segundos`, com ciclo de trabalho `duty`.

    FP16 porque e o que acorda os tensor cores e leva a placa ao teto de
    potencia; FP32 fica bem abaixo e nao testaria a hipotese.
    """
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    fim = time.time() + segundos
    pico = 0.0
    while time.time() < fim:
        t0 = time.time()
        while time.time() - t0 < 2.0 * duty:
            a = (a @ b).clamp_(-8, 8)          # clamp evita inf e manter o loop
        torch.cuda.synchronize()
        pico = max(pico, watts())
        if duty < 1.0:
            # a PAUSA e o ponto da fase 2: e o dI/dt que mata contato marginal
            time.sleep(2.0 * (1.0 - duty))
    del a, b
    torch.cuda.empty_cache()
    return pico


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--degraus", nargs="+", type=int,
                    default=[2048, 4096, 8192, 12288, 16384],
                    help="lado da matriz em cada degrau da fase 1")
    ap.add_argument("--dwell", type=float, default=60.0,
                    help="segundos em cada degrau")
    ap.add_argument("--ciclos", type=float, default=600.0,
                    help="segundos de liga-desliga na fase 2")
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--saida", type=Path,
                    default=ROOT / "logs/gpu_watchdog/stress.log")
    args = ap.parse_args()
    args.saida.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("sem CUDA -- a GPU ja esta caida")
    nome = torch.cuda.get_device_name(0)
    log(f"=== stress iniciado em {nome} ===", args.saida)
    log(f"repouso: {watts():.1f} W", args.saida)

    estado = {"fase": None, "degrau": None, "pico_w": 0.0}
    try:
        for n in args.degraus:
            estado.update(fase=1, degrau=n)
            p = queima(n, args.dwell, 1.0)
            estado["pico_w"] = max(estado["pico_w"], p)
            log(f"fase 1 · matriz {n}x{n} · {args.dwell:.0f}s · pico {p:.1f} W",
                args.saida)

        n = args.degraus[-1]
        estado.update(fase=2, degrau=n)
        log(f"fase 2 · liga-desliga {args.duty:.0%} em {n}x{n} por "
            f"{args.ciclos:.0f}s -- e aqui que transiente mata", args.saida)
        p = queima(n, args.ciclos, args.duty)
        estado["pico_w"] = max(estado["pico_w"], p)
        log(f"fase 2 concluida · pico {p:.1f} W", args.saida)
        log("=== SOBREVIVEU as duas fases ===", args.saida)
        log("a causa nao e carga: sobra slot/riser ou reinicializacao do "
            "driver (Persistence Mode esta Disabled)", args.saida)
    except Exception as e:
        log(f"!! MORREU na fase {estado['fase']}, degrau {estado['degrau']}, "
            f"pico ate aqui {estado['pico_w']:.1f} W", args.saida)
        log(f"!! {type(e).__name__}: {str(e)[:300]}", args.saida)
        log(f"!! nvidia-smi agora: {watts()}", args.saida)
        estado["erro"] = f"{type(e).__name__}: {str(e)[:300]}"
        (args.saida.parent / "stress_estado.json").write_text(
            json.dumps(estado, indent=2))
        raise


if __name__ == "__main__":
    main()
