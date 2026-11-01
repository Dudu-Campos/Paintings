"""Instala as 5.520 imagens de `val` do Places-LT que faltavam no NFS.

O §4.1f item 6 nao pode rodar a rota do `val` no Places-LT porque so 1.780 das
7.300 linhas tinham imagem em disco. O padrao NAO era perda aleatoria: **89
classes com val 100% presente e 276 com 0%**, 20 por classe nas 365 -- ou seja
276 classes nunca foram materializadas, o que casa com o §9.1 ("a primeira versao
mapeou classes com subpastas de forma ingenua").

**A FONTE E A LISTA OFICIAL, NAO O METADADO**, e essa e a licao do §5.18d. Na
convencao do Places-LT o **hifen e separador de diretorio** e o underscore faz
parte do nome:

    apartment_building-outdoor  ->  a/apartment_building/outdoor
    desert-sand                 ->  d/desert/sand
    desert_road                 ->  d/desert_road          (UM nivel)

Quem resolveu isso adivinhando permutou 5 classes (§5.18d). Aqui a ambiguidade
nao existe porque o mapa vem do **indice numerico**, que esta nos dois arquivos:
`categories_places365.txt` da indice -> caminho no tar, e `Places_LT_val.txt` da
indice -> arquivos. A `style` de destino e o caminho com `/` -> `_`, derivacao
que bate com o metadado nas 365.

Conferido antes de extrair: o `val` do metadado bate **100% com a lista oficial
nas 365 classes**, entao a permutacao do §5.18d e so do treino, como aquela
secao afirma. Nenhum rotulo de val esta em disputa.

**GUARDAS**, nesta ordem, e o script aborta em qualquer uma:
  1. os 20 arquivos de conferencia -- que JA existiam no NFS -- tem de sair do
     tar **byte a byte identicos**. Sem isso nao ha como saber se o tar e a mesma
     fonte da extracao original (§9.0: copia pode ser DERIVADA, nao copia).
  2. nunca sobrescreve: um destino que ja exista aborta em vez de ser trocado.
  3. a contagem final por classe tem de dar 20 nas 365.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, required=True,
                    help="diretorio onde o tar foi extraido (estrutura data_256/...)")
    ap.add_argument("--members", type=Path, required=True,
                    help="CSV com as colunas member,dest,exists")
    ap.add_argument("--images", type=Path, default=ROOT / "data/places_lt/images")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t = pd.read_csv(args.members)
    print(f"val total {len(t):,} | ja no NFS {t.exists.sum():,} | a instalar {(~t.exists).sum():,}")

    # (1) CONFERENCIA byte a byte nos que ja existiam.
    check = t[t.exists].head(20)
    same = 0
    for _, r in check.iterrows():
        staged, live = args.stage / r.member, args.images / r.dest
        if not staged.is_file():
            raise SystemExit(f"conferencia impossivel: {r.member} nao saiu do tar")
        if digest(staged) != digest(live):
            raise SystemExit(
                f"O TAR NAO E A MESMA FONTE: {r.dest} difere do que ja esta no NFS.\n"
                "Instalar misturaria duas origens de imagem, e o §8 mede que "
                "pre-processamento diferente vale 4,91pp.")
        same += 1
    print(f"conferencia: {same}/{len(check)} arquivos ja existentes batem BYTE A BYTE com o tar")

    # (2) instalar, sem sobrescrever nada.
    todo = t[~t.exists]
    missing = [r.member for _, r in todo.iterrows() if not (args.stage / r.member).is_file()]
    if missing:
        raise SystemExit(f"{len(missing):,} membros nao sairam do tar (ex: {missing[0]})")
    clash = [r.dest for _, r in todo.iterrows() if (args.images / r.dest).exists()]
    if clash:
        raise SystemExit(f"{len(clash):,} destinos JA existem (ex: {clash[0]}) -- abortando")
    if args.dry_run:
        print(f"dry-run: {len(todo):,} arquivos seriam instalados. Nada escrito.")
        return
    for _, r in todo.iterrows():
        target = args.images / r.dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.stage / r.member, target)
    print(f"instalados {len(todo):,} arquivos")

    # (3) contagem final por classe.
    t["now"] = [(args.images / d).is_file() for d in t.dest]
    por_classe = t.assign(style=t.dest.str.split("/").str[0]).groupby("style").now.sum()
    print(f"val presente agora: {t.now.sum():,} de {len(t):,}")
    print(f"classes com as 20: {(por_classe == 20).sum()} de {len(por_classe)}")
    if not (por_classe == 20).all():
        raise SystemExit("alguma classe nao ficou com 20 -- conferir antes de usar")
    print("FIM: as 365 classes com 20 imagens de val cada")


if __name__ == "__main__":
    main()
