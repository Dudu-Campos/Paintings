"""Monta as 62.500 imagens de TREINO do Places-LT a partir do tar do Places365.

**Por que existe.** `data/places_lt/images` tinha so 67.821 arquivos contra as
106.300 linhas do metadado -- as `Places365_val_*` (de onde saem o teste e o val
do LT) estavam la, mas o treino nao. O §9.1 registra que os tars foram extraidos
em `/tmp/prism_data/places_lt/raw`, e o /tmp foi limpo desde entao. O
`train_256.tar` (26 GB) segue em `data/places_lt/tars/`, entao nao ha o que
baixar.

**Extrai POR MEMBRO, nao o tar inteiro.** Descompactar 26 GB para usar 62.500 de
~1,8 milhao de imagens desperdicaria ~25 GB de disco -- e acabamos de liberar
149G justamente para nao viver assim. `tarfile` le sequencialmente e so
materializa os membros pedidos.

**Convencao de nome, conferida contra o metadado:** a lista do LT diz
`train/airfield/00004851.jpg` e `categories_places365.txt` diz `/a/airfield 0`,
enquanto `data/places_lt_meta.csv` espera `style=a_airfield` e
`filename=00004851.jpg`. O destino e portanto `images/a_airfield/00004851.jpg`,
que e o par `(pasta, arquivo)` que o resolvedor de legendas casa (§8: indexar so
por nome daria legenda errada em silencio -- aqui `00000004.jpg` se repete em
varias classes, e a rodada de 08/set recusou 4.985 nomes ambiguos por isso).
"""
from __future__ import annotations

import argparse
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tar", type=Path, default=ROOT / "data/places_lt/tars/train_256.tar")
    ap.add_argument("--list", type=Path, default=ROOT / "data/places_lt/Places_LT_train.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "data/places_lt/images")
    args = ap.parse_args()

    wanted = {}
    for line in args.list.read_text().splitlines():
        if not line.strip():
            continue
        rel = line.split()[0]                      # train/airfield/00004851.jpg
        parts = rel.split("/")
        category, filename = parts[-2], parts[-1]
        # `/a/airfield` -> `a_airfield`: a inicial vem do primeiro caractere da
        # categoria, que e como o metadado foi construido.
        # TRES convencoes para a mesma categoria, e o hifen do LT e AMBIGUO:
        #   apartment_building-outdoor  ->  tar: apartment_building/outdoor
        #   desert-sand                 ->  tar: desert_sand          (nome!)
        # Adivinhar dava 528 imagens perdidas concentradas em 5 categorias --
        # `desert_road` caiu de 176 para 1 e duas classes sumiram inteiras.
        # Por isso as DUAS chaves entram e o casamento decide pelo que existe
        # no tar, em vez de supor qual leitura vale.
        style = f"{category[0]}_{category.replace('-', '_')}"
        for inner in {category, category.replace("-", "/"),
                      category.replace("-", "_")}:
            wanted[f"data_256/{category[0]}/{inner}/{filename}"] = (style, filename)
    log(f"lista do LT: {len(wanted):,} imagens de treino")

    done = sum(1 for s, f in wanted.values() if (args.out / s / f).exists())
    log(f"ja em disco: {done:,}")
    if done == len(wanted):
        log("nada a fazer"); return

    written, seen = 0, 0
    started = time.perf_counter()
    with tarfile.open(args.tar, "r|") as tar:        # streaming: nao le o indice inteiro
        for member in tar:
            seen += 1
            if seen % 200_000 == 0:
                log(f"  varridos {seen:,} membros, escritos {written:,}")
            name = member.name.lstrip("./")
            hit = wanted.get(name)
            if hit is None or not member.isfile():
                continue
            style, filename = hit
            target = args.out / style / filename
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            target.write_bytes(src.read())
            written += 1
    log(f"escritos {written:,} em {(time.perf_counter() - started) / 60:.1f} min "
        f"(varridos {seen:,} membros)")

    falta = [f"{s}/{f}" for s, f in wanted.values() if not (args.out / s / f).exists()]
    log(f"FIM: faltam {len(falta):,}" + (f" -- ex: {falta[:3]}" if falta else " (completo)"))


if __name__ == "__main__":
    main()
