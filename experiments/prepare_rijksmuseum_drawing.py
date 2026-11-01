"""Meta e rótulos do braço de desenhos do Rijksmuseum.

Por que um dataset separado das pinturas em vez de juntar os dois: misturar
desenho e pintura no mesmo alvo de período transformaria parte da tarefa em
**detecção de meio** (papel/grafite contra tela/óleo separam-se trivialmente),
e aí o número não responderia mais a pergunta registrada. Desenhos sozinhos
são o mesmo museu, a mesma tarefa de meio-século e ~9x os dados das pinturas
-- só o `n` muda, que é a variável do teste.

Rótulo: meio-século extraído do primeiro ano de quatro dígitos em `date`.
Classes com menos de `--min-count` obras saem, como já era feito nas pinturas.

Saída: data/rijksdraw_meta.csv (stem,filename,style) e
data/rijksdraw_labels.csv (stem,period,artist,date).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "rijksmuseum/metadata.csv")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "rijksmuseum/drawing")
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--meta-out", type=Path, default=ROOT / "data/rijksdraw_meta.csv")
    parser.add_argument("--labels-out", type=Path, default=ROOT / "data/rijksdraw_labels.csv")
    args = parser.parse_args()

    catalog = pd.read_csv(args.metadata)
    catalog = catalog[catalog["type"] == "drawing"].drop_duplicates("id")

    # O CSV lista o que foi catalogado; o disco manda. `AK-MAK-1541` e
    # `SK-A-2082` aparecem em duas facetas, então o disco é a fonte da verdade
    # sobre o que existe aqui.
    on_disk = {path.stem for path in args.image_dir.glob("*.jpg")}
    catalog = catalog[catalog["id"].isin(on_disk)].copy()

    year = pd.to_numeric(
        catalog["date"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    catalog["period"] = (year // 50) * 50
    catalog = catalog[catalog["period"].notna()]

    counts = catalog["period"].value_counts()
    keep = counts[counts >= args.min_count].index
    catalog = catalog[catalog["period"].isin(keep)]

    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    # `style` guarda o subdiretório, não o rótulo -- é assim que os extratores
    # resolvem o caminho (`{image_dir}/{style}/{filename}`). O alvo real mora
    # no arquivo de rótulos, exatamente como no braço das pinturas.
    pd.DataFrame({
        "stem": catalog["id"],
        "filename": catalog["id"] + ".jpg",
        "style": "drawing",
    }).to_csv(args.meta_out, index=False)

    pd.DataFrame({
        "stem": catalog["id"],
        "period": catalog["period"],
        "artist": catalog["artist"],
        "date": catalog["date"],
    }).to_csv(args.labels_out, index=False)

    print(f"{len(catalog):,} desenhos, {catalog['period'].nunique()} periodos")
    print(catalog["period"].value_counts().sort_index().to_string())
    print(f"salvo: {args.meta_out}  {args.labels_out}")


if __name__ == "__main__":
    main()
