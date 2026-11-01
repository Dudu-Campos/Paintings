#!/usr/bin/env python3
"""Funde um remendo de `--only-stems` no cache WND-CHARM tiled de um corpus.

Existe porque o extrator indexa shards por POSICAO na lista da coorte: uma
corrida feita com 528 stems a menos tem shards que nao podem ser reaproveitados
quando a coorte volta ao tamanho cheio -- reusa-los daria bloco desalinhado, o
modo de falha do §5.18b. Em vez de repetir 4,5 h de extracao, extrai-se o
subconjunto num diretorio proprio e funde-se AQUI, por stem.

Reconstroi os TRES arrays, e nao so o `pooled.npy`: se so o pooled fosse
atualizado, apagar o cache faria o repool voltar a sair sem os stems novos, em
silencio. Armadilha latente e exatamente o que o §8 mede.

GUARDAS, e aborta em qualquer uma:
  1. os metadados de extracao das duas corridas tem de ser identicos nas chaves
     de protocolo (inclusive os 2.895 nomes de feature).
  2. nenhum stem em comum entre base e remendo.
  3. as linhas da BASE no arquivo novo tem de sair BIT A BIT iguais as antigas.
  4. as linhas do REMENDO idem.
  5. o `pooled` novo tem de reproduzir o antigo BIT A BIT nos stems da base.
  6. a coorte do registry tem de ficar 100% coberta no fim.
Escreve em .tmp e renomeia, entao uma interrupcao nao deixa cache meio-feito.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHAVES = ("grid", "standard_side", "tile_side", "protocol", "library",
          "grayscale", "color_transform", "raw_names")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--patch", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # (1) mesma configuracao de extracao
    mb = json.loads((args.base / "meta.json").read_text())
    mp = json.loads((args.patch / "meta.json").read_text())
    for k in CHAVES:
        if mb.get(k) != mp.get(k):
            raise SystemExit(f"meta.json divergiu em {k!r} -- corridas nao comparaveis")
    print(f"guarda 1: meta.json identico nas {len(CHAVES)} chaves de protocolo")

    sb = np.load(args.base / "stems.npy", allow_pickle=True).astype(str)
    sp_ = np.load(args.patch / "stems.npy", allow_pickle=True).astype(str)
    fb = np.load(args.base / "features.npy", mmap_mode="r")
    fp = np.load(args.patch / "features.npy", mmap_mode="r")
    if fb.shape[1:] != fp.shape[1:]:
        raise SystemExit(f"forma divergente: {fb.shape} vs {fp.shape}")
    print(f"base {fb.shape}  remendo {fp.shape}")

    # (2) sem sobreposicao
    comum = set(sb) & set(sp_)
    if comum:
        raise SystemExit(f"{len(comum)} stems em comum (ex.: {sorted(comum)[:3]}) -- abortando")
    print(f"guarda 2: nenhum stem em comum")

    if args.dry_run:
        print(f"dry-run: resultaria em {len(sb) + len(sp_):,} stems. Nada escrito.")
        return

    n = len(sb) + len(sp_)
    tmp_f = args.base / "features.npy.tmp"
    novo = np.lib.format.open_memmap(tmp_f, mode="w+", dtype=np.float32,
                                     shape=(n, *fb.shape[1:]))
    for i in range(0, len(sb), 2048):
        # a fatia do DESTINO tem de ser limitada ao tamanho da base: `novo` e mais
        # longo, entao `novo[i:i+2048]` devolve mais linhas que `fb[i:i+2048]` no
        # ultimo pedaco e o broadcast falha.
        j = min(i + 2048, len(sb))
        novo[i:j] = fb[i:j]
    novo[len(sb):] = fp[:]
    novo.flush()

    # (3) e (4) bit a bit
    chk = np.load(tmp_f, mmap_mode="r")
    # `array_equal(equal_nan=True)` e NAO diferenca maxima: o WND-CHARM poolado tem
    # NaN e inf legitimos (razoes e logs de estatistica de tile), e em Python
    # `max(0.0, nan)` devolve 0.0 -- uma guarda por maxdiff ENGOLE a divergencia.
    for rotulo, src, lo, hi in (("base", fb, 0, len(sb)), ("remendo", fp, len(sb), n)):
        for i in range(lo, hi, 2048):
            j = min(i + 2048, hi)
            if not np.array_equal(np.asarray(chk[i:j]),
                                  np.asarray(src[i - lo:j - lo]), equal_nan=True):
                raise SystemExit(f"linhas de {rotulo} divergiram em [{i}:{j}] -- abortando")
        print(f"guarda {3 if rotulo == 'base' else 4}: linhas de {rotulo} BIT A BIT")

    # pooled novo, e (5) confere contra o antigo nos stems da base
    pooled_velho = args.base / "pooled.npy"
    pv = np.load(pooled_velho, mmap_mode="r") if pooled_velho.exists() else None
    pool = np.empty((n, fb.shape[2]), dtype=np.float32)
    for i in range(0, n, 2048):
        pool[i:i + 2048] = np.asarray(chk[i:i + 2048], dtype=np.float32).mean(axis=1)
    if pv is not None:
        at = {s: i for i, s in enumerate(
            np.load(args.base / "pooled_stems.npy", allow_pickle=True).astype(str))}
        idx = np.array([at[s] for s in sb if s in at])
        onde = [i for i, s in enumerate(sb) if s in at]
        if not np.array_equal(pool[:len(sb)][onde], np.asarray(pv[idx]), equal_nan=True):
            raise SystemExit("pooled da base divergiu -- abortando")
        print(f"guarda 5: pooled da base BIT A BIT em {len(idx):,} stems")

    stems = np.concatenate([sb, sp_])
    up = np.concatenate([np.load(args.base / "upscaled.npy"),
                         np.load(args.patch / "upscaled.npy")])
    # gravar por HANDLE: `np.save` acrescenta `.npy` a um caminho que nao termina
    # nisso, entao `pooled.npy.tmp` sairia como `pooled.npy.tmp.npy` e o rename
    # abaixo falharia com metade dos arquivos ja trocados (§8).
    for nome, arr in (("pooled.npy", pool), ("pooled_stems.npy", stems),
                      ("stems.npy", stems), ("upscaled.npy", up)):
        with open(args.base / f"{nome}.tmp", "wb") as handle:
            np.save(handle, arr)
    del novo, chk, pv
    for nome in ("features.npy", "pooled.npy", "pooled_stems.npy", "stems.npy", "upscaled.npy"):
        (args.base / f"{nome}.tmp").rename(args.base / nome)
    print(f"gravado: {n:,} stems")

    # (6) a coorte fecha?
    sys.path.insert(0, str(ROOT / "src" / "pipeline"))
    from registry import load_blocks
    coorte = load_blocks(args.dataset)
    tem = set(np.load(args.base / "stems.npy", allow_pickle=True).astype(str))
    falta = [s for s in (str(x) for x in coorte.stems) if s not in tem]
    if falta:
        raise SystemExit(f"guarda 6 FALHOU: {len(falta)} stems da coorte ainda ausentes")
    print(f"guarda 6: a coorte de {len(coorte.stems):,} esta 100% coberta")


if __name__ == "__main__":
    main()
