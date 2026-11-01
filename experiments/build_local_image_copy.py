"""Reconstroi a copia local reduzida do corpus (lado menor 256), em paralelo.

Por que reduzida e nao copia fiel: o §9.0 registra que
`/tmp/prism_data/multitask_painting100k/images` SEMPRE foi derivada -- lado menor
forcado a 256, 4,1 GB contra 48 GB dos originais -- e foi dela que saiu o cache
CLIP de producao (`outputs/multitask_clip_vitb32.npy`), logo o 12,74 do §5.18f.
Reconstruir a mesma derivacao faz o braco de controle do B/32 VALIDAR a
reconstrucao: se ele nao reproduzir 12,74, a copia nao e a mesma e eu fico
sabendo, em vez de comparar escalas de CLIP sobre fontes diferentes (§8: esse
tipo de par ja valeu 4,91pp).

Para CLIP a 224 px o lado menor 256 e o ponto de operacao canonico, entao a
reducao nao joga fora informacao que o modelo usaria.

O gargalo e o NFS -- medido em 6 MB/s e 16 arquivos/s numa leitura sequencial,
contra os 12x de NVMe local que o §1 registra --, entao o paralelismo aqui e de
I/O e nao de CPU: os workers passam a maior parte do tempo bloqueados.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # varredura de museu estoura o limite (§8)


def uma(par: tuple[str, str, int, int]) -> tuple[bool, int]:
    origem, destino, lado, qualidade = par
    d = Path(destino)
    if d.is_file() and d.stat().st_size > 0:
        return True, 0                  # retomavel: nao refaz o que existe
    try:
        with Image.open(origem) as img:
            img = img.convert("RGB")
            w, h = img.size
            menor = min(w, h)
            if menor > lado:
                escala = lado / menor
                img = img.resize((max(1, round(w * escala)), max(1, round(h * escala))),
                                 Image.BILINEAR)
            tmp = d.with_suffix(d.suffix + ".tmp")
            img.save(tmp, "JPEG", quality=qualidade)
            os.replace(tmp, d)          # atomico: nunca deixa arquivo pela metade
        return True, d.stat().st_size
    except Exception as exc:
        print(f"[falha] {origem}: {exc}", flush=True)
        return False, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=Path("data/multitask_painting100k/original/images"))
    ap.add_argument("--dest", type=Path,
                    default=Path("/tmp/prism_data/multitask_painting100k/images"))
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    fontes = [f for f in args.source.rglob("*") if f.is_file() and f.suffix.lower() in
              (".jpg", ".jpeg", ".png")]
    print(f"[{time.strftime('%H:%M:%S')}] {len(fontes):,} arquivos na origem", flush=True)

    # Destino PLANO, como a copia antiga. Seguro porque os nomes sao unicos na
    # origem -- conferido: 103.250 arquivos, 0 nomes ambiguos. Se algum dia
    # deixar de ser, isto sobrescreve em silencio, entao a checagem fica aqui.
    nomes = {f.name for f in fontes}
    if len(nomes) != len(fontes):
        raise SystemExit(f"{len(fontes)-len(nomes)} nomes duplicados: destino plano "
                         "sobrescreveria em silencio (§8)")

    tarefas = [(str(f), str(args.dest / f.name), args.short_side, args.quality)
               for f in fontes]
    t0, ok, bytes_ = time.time(), 0, 0
    with ProcessPoolExecutor(args.workers) as pool:
        futuros = [pool.submit(uma, t) for t in tarefas]
        for i, fut in enumerate(as_completed(futuros), 1):
            sucesso, n = fut.result()
            ok += sucesso
            bytes_ += n
            if i % 5000 == 0:
                dt = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}]   {i:,}/{len(tarefas):,}  "
                      f"{i/dt:.0f} img/s  faltam {(len(tarefas)-i)/max(i/dt,1e-9)/60:.1f} min",
                      flush=True)
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] fim: {ok:,}/{len(tarefas):,} ok, "
          f"{bytes_/1e9:.1f} GB escritos, {dt/60:.1f} min", flush=True)
    if ok != len(tarefas):
        raise SystemExit(f"{len(tarefas)-ok} falhas -- ver [falha] acima")


if __name__ == "__main__":
    main()
