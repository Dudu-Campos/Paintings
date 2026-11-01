#!/usr/bin/env python3
"""Reinstala as 528 imagens de TREINO do Places-LT que o NFS perdeu.

CONTEXTO, e ele corrige o §5.18d. Aquela secao afirma que
`data/places_lt_meta.csv` tem 5 classes PERMUTADAS e que 528 imagens de treino
estao com rotulo trocado. **Medido em 08/out: o metadado esta CORRETO.** Por
classe e por split, os conjuntos de nome de arquivo casam a lista oficial
exatamente (28/20/100, 128/20/100, 176/20/100, 85/20/100, 118/20/100), e o
teste de subconjunto identifica a classe oficial certa nas 15 celulas.

O bug do §5.18d foi reconstruido 5/5: ele pareou CAMINHOS oficiais ordenados
contra NOMES do metadado ordenados, POSICIONALMENTE. Como "/" (0x2F) ordena
antes de "_" (0x5F), a ordem oficial e
    d/desert/sand  <  d/desert/vegetation  <  d/desert_road
e a do metadado e
    d_desert_road  <  d_desert_sand  <  d_desert_vegetation
Zipar os dois produz exatamente o ciclo que aquela secao reporta. E o gotcha do
proprio §10.8 ("casar atribuicoes entre scripts por posicao e fragil e
silencioso"), cometido pela secao que acusou o metadado de adivinhar.

O DEFEITO REAL e outro: as cinco classes perderam 528 ARQUIVOS de treino no
NFS. Dos 535 oficiais, 7 estao em disco. Os cinco diretorios tem 1.135 arquivos,
dos quais so 607 sao stems do metadado (500 teste + 100 val + 7 treino); os
outros 528 sao imagem de OUTRA classe, materializada sob o nome errado pelo mapa
ingenuo do §9.1. Ou seja: o mapa ingenuo estragou a MATERIALIZACAO, nao o
metadado.

E NADA do que foi publicado esta contaminado -- isto foi medido, nao suposto:
  * as 528 linhas de feature do `hand-8` reproduzem a imagem OFICIAL do tar
    BIT A BIT (528/528, via Fourier determinístico), entao a extracao leu a
    imagem certa, antes de o arquivo ser trocado em disco;
  * o split de TESTE bate byte a byte com `val_256.tar` (500/500), e os nomes
    de teste sao globalmente unicos (`Places365_val_*`), portanto imunes ao bug
    de nome de diretorio;
  * o `val` bate byte a byte com `train_256.tar` (100/100).

GUARDAS, nesta ordem, e o script aborta em qualquer uma:
  1. os arquivos que JA existem (7 de treino + 100 de val) tem de sair do stage
     byte a byte identicos -- senao o stage nao e a mesma fonte (§9.0).
  2. o Fourier de cada arquivo a instalar tem de reproduzir a linha do parquet
     do `hand-8` BIT A BIT. E a guarda mais forte que o §4.1f nao tinha: ela
     prova que o arquivo instalado E o arquivo de que as features publicadas
     sairam, em vez de so provar que a fonte e plausivel.
  3. nunca sobrescreve: destino existente aborta.
  4. contagem final por classe contra a lista oficial.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "extractors" / "dormant"))
import extract_fourier_spectrum_features as spectrum  # noqa: E402

# indice oficial -> (caminho no tar, style do metadado)
ALVO = {
    116: ("d/desert/sand", "d_desert_sand"),
    117: ("d/desert/vegetation", "d_desert_vegetation"),
    118: ("d/desert_road", "d_desert_road"),
    141: ("f/field/wild", "f_field_wild"),
    142: ("f/field_road", "f_field_road"),
}


def digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def fourier(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return spectrum.image_features(image.convert("RGB"), "compact")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=Path, required=True,
                    help="diretorio com train_256.tar extraido (estrutura data_256/...)")
    ap.add_argument("--images", type=Path, default=ROOT / "data/places_lt/images")
    ap.add_argument("--lists", type=Path, default=ROOT / "data/places_lt")
    ap.add_argument("--meta", type=Path, default=ROOT / "data/places_lt_meta.csv")
    ap.add_argument("--fourier", type=Path,
                    default=ROOT / "results/fourier_places_lt_features.parquet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # lista oficial: style -> {split: {nome de arquivo}}
    oficial: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for split in ("train", "val", "test"):
        for linha in (args.lists / f"Places_LT_{split}.txt").read_text().splitlines():
            if not linha.strip():
                continue
            caminho, idx = linha.rsplit(" ", 1)
            idx = int(idx)
            if idx in ALVO:
                oficial[ALVO[idx][1]][split].add(Path(caminho).name)

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    meta = meta[meta["style"].isin(oficial)]

    # o metadado bate a lista oficial? (a correcao do §5.18d, conferida aqui)
    for style, splits in oficial.items():
        for split, nomes in splits.items():
            tem = set(meta[(meta["style"] == style) & (meta["split"] == split)]["filename"])
            if tem != nomes:
                raise SystemExit(
                    f"metadado divergiu da lista oficial em {style}/{split}: "
                    f"{len(tem)} contra {len(nomes)} -- parar e reconferir o §5.18d")
    print(f"metadado CONFERE com a lista oficial nas {len(oficial)} classes x 3 splits")

    path_de = {s: p for p, s in ALVO.values()}
    linhas = []
    for row in meta.itertuples(index=False):
        linhas.append({
            "stem": row.stem,
            "style": row.style,
            "split": row.split,
            "member": f"data_256/{path_de[row.style]}/{row.filename}",
            "dest": f"{row.style}/{row.filename}",
        })
    tab = pd.DataFrame(linhas)
    tab["no_stage"] = [(args.stage / m).is_file() for m in tab["member"]]
    tab["no_nfs"] = [(args.images / d).is_file() for d in tab["dest"]]
    print(f"linhas nas cinco classes {len(tab):,} | no stage {tab.no_stage.sum():,} "
          f"| no NFS {tab.no_nfs.sum():,}")

    # (1) byte a byte nos que JA existem e estao no stage
    confere = tab[tab.no_nfs & tab.no_stage]
    iguais = sum(digest(args.stage / r.member) == digest(args.images / r.dest)
                 for r in confere.itertuples(index=False))
    if iguais != len(confere):
        raise SystemExit(
            f"O STAGE NAO E A MESMA FONTE: {len(confere) - iguais} de {len(confere)} "
            "arquivos ja existentes divergem. Instalar misturaria duas origens, e o "
            "§8 mede que pre-processamento diferente vale 4,91pp.")
    print(f"guarda 1: {iguais}/{len(confere)} arquivos ja existentes batem BYTE A BYTE")

    todo = tab[(~tab.no_nfs) & tab.no_stage]
    falta_stage = tab[(~tab.no_nfs) & (~tab.no_stage)]
    if len(falta_stage):
        raise SystemExit(f"{len(falta_stage):,} a instalar nao sairam do tar "
                         f"(ex: {falta_stage.iloc[0]['member']})")

    # (2) o Fourier de cada um reproduz o parquet do hand-8 bit a bit?
    par = pd.read_parquet(args.fourier).set_index("stem")
    cols = [c for c in par.columns if c != "style"]
    bate = fora = 0
    for r in todo.itertuples(index=False):
        if r.stem not in par.index:
            raise SystemExit(f"{r.stem} nao esta no parquet do Fourier -- coorte divergente")
        v = fourier(args.stage / r.member)
        g = par.loc[r.stem, cols].to_numpy(float)
        if np.abs(v - g).max() == 0.0:
            bate += 1
        else:
            fora += 1
    if fora:
        raise SystemExit(
            f"{fora} de {len(todo)} arquivos NAO reproduzem a feature publicada. "
            "O arquivo do stage nao e o que a extracao leu -- nao instalar.")
    print(f"guarda 2: {bate}/{len(todo)} reproduzem a feature publicada BIT A BIT")

    # (3) nunca sobrescreve
    choque = [r.dest for r in todo.itertuples(index=False) if (args.images / r.dest).exists()]
    if choque:
        raise SystemExit(f"{len(choque):,} destinos JA existem (ex: {choque[0]}) -- abortando")

    if args.dry_run:
        print(f"dry-run: {len(todo):,} arquivos seriam instalados. Nada escrito.")
        return

    for r in todo.itertuples(index=False):
        alvo = args.images / r.dest
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.stage / r.member, alvo)
    print(f"instalados {len(todo):,} arquivos")

    # (4) contagem final por classe contra a oficial
    tab["agora"] = [(args.images / d).is_file() for d in tab["dest"]]
    ok = True
    for style, splits in sorted(oficial.items()):
        for split in ("train", "val", "test"):
            # acesso por colchete: `tab.style` e o acessor Styler do pandas (§8)
            sel = tab[(tab["style"] == style) & (tab["split"] == split)]
            tem, quer = int(sel["agora"].sum()), len(splits[split])
            marca = "" if tem == quer else "   <-- FALTA"
            if tem != quer:
                ok = False
            print(f"  {style:22s} {split:6s} {tem:4d} de {quer:4d}{marca}")
    if not ok:
        raise SystemExit("alguma classe/split nao fechou contra a lista oficial")
    print("FIM: as cinco classes completas contra a lista oficial")


if __name__ == "__main__":
    main()
