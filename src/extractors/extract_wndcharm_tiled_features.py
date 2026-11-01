"""WND-CHARM no PROTOCOLO CANONICO de Shamir et al. (ACM TAP 7(2), 2010).

A primeira corrida (`extract_wndcharm_features.py`) usou tile unico a 256 px, e a
auditoria do artigo de arte mostrou que isso divergia do metodo em cinco pontos.
Este extrator implementa quatro deles; o quinto (transformada de cor) nao e
possivel com a implementacao de referencia -- ver LIMITACAO abaixo.

Citacoes do artigo que definem o protocolo:

    "first downsampled each image such that the smallest side of the image is
     600 pixels ... resulting in a dataset of images with a standard size of
     600x600 pixels"
    "each image was divided into 16 equal-sized (150x150 pixels) tiles such that
     features are computed on each of the 16 tiles, providing 16 feature vectors"
    "the weakest 85% of the features (with the lowest Fisher scores) are
     rejected, resulting in a feature space of 548 image features"
    "classified using a simple weighted nearest neighbor rule, such that the
     feature weights are the Fisher scores"
    "averaging the 16 similarity vectors of the tiles, providing a single
     similarity vector"

Este script cobre a padronizacao e o tiling; a selecao de Fisher e o WND ficam no
avaliador, porque tem de ser ajustados DENTRO de cada fold (selecionar feature na
coorte inteira seria vazamento -- o mecanismo do §6 que derrubou o SNF).

**O QUE ISSO MUDA NA NATUREZA DA COMPARACAO, e e o achado da auditoria:** o
metodo canonico opera em tiles de 150x150, ou seja e **LOCAL**. Ele cai na banda
`Local` da taxonomia do §7.11b -- a categoria em que o nosso banco global e
fraco, e o §7.11b mede o melhor `Local` classico abrindo 11,9pp sobre o melhor
`Global` no Food-101. A corrida de tile unico mediu o WND-CHARM no modo que mais
favorece o nosso lado, sem que isso fosse intencional.

**LIMITACAO DECLARADA: sem a transformada de cor.** O artigo usa uma
transformada em que cada pixel e classificado em 16 classes de cor por logica
fuzzy e recebe intensidade proporcional ao comprimento de onda. O metodo TEM
cor; a implementacao de referencia nao a expoe. Conferido em tres niveis:
`CharmFeatures(color=True)` falha porque o wrapper faz `width, height =
ndarray.shape`; nenhum dos 2.895 nomes casa `color|hue|wavelength`; pedir
`Haralick Textures (Color ())` por `f_names` aborta com assertion do Eigen em
`HSVcolor`. E o proprio C API diz:

    // capi.cpp:137
    // N.B.: Color is not supported
    matrix->remap_pix_plane(mat_ptr, width, height);

Habilitar exigiria recompilar um libcharm modificado por caminho que o upstream
desabilitou, e um plano de cor com layout errado produziria features plausiveis e
erradas em silencio. Decisao do usuario em 28/set: rodar sem cor.

**UPSCALE, medido e marcado.** O artigo assume imagens grandes. Na nossa coorte
**30,8%** tem lado menor abaixo de 600 px (medido, n=600: p05=373, p25=538,
p50=734), entao forcar 600x600 as interpola. Interpolacao nao acrescenta
informacao e para features espectrais/de textura tende a SUAVIZAR, ou seja corre
contra o WND-CHARM. Por isso cada imagem carrega um flag `upscaled`: o avaliador
reporta o subconjunto nativo como sensibilidade, o que responde a objecao com
medicao em vez de argumento.

Saida: `outputs/{dataset}_wndcharm_tiled/` com `features.npy` (n, 16, 2895)
float32, `stems.npy`, `upscaled.npy`.

Uso:
    python src/extractors/extract_wndcharm_tiled_features.py \
        --dataset multitask \
        --image-dir data/multitask_painting100k/original/images --workers 28
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

STANDARD = 600      # lado do quadrado padronizado, do artigo
GRID = 4            # 4x4 = os 16 tiles do artigo
SHARD = 1000
_WORKER: dict = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _init_worker() -> None:
    from CharmFeatures import CharmFeatures

    # Um nivel de paralelismo so, o nosso: dentro de um Pool o forking interno
    # multiplicaria 28 x N processos numa maquina compartilhada (§1).
    _WORKER["cf"] = CharmFeatures(verbosity=0, forking_executor=False,
                                  forking_gabor=False, forking_haralick=False)


def standardise(image: Image.Image) -> tuple[np.ndarray, bool]:
    """Lado menor -> 600 px, depois recorte central 600x600.

    Devolve tambem se houve UPSCALE, que e o que o flag por imagem registra.
    """
    width, height = image.size
    scale = STANDARD / min(width, height)
    upscaled = scale > 1.0
    new = (max(STANDARD, int(round(width * scale))),
           max(STANDARD, int(round(height * scale))))
    image = image.resize(new, Image.BILINEAR)
    left = (new[0] - STANDARD) // 2
    top = (new[1] - STANDARD) // 2
    square = image.crop((left, top, left + STANDARD, top + STANDARD))
    return np.asarray(square, dtype=np.float32), upscaled


def _extract_one(task: tuple[str, str]) -> tuple[str, np.ndarray | None, bool, str]:
    stem, path = task
    try:
        with Image.open(path) as handle:
            array, upscaled = standardise(handle.convert("L"))
        side = STANDARD // GRID
        out = np.empty((GRID * GRID, _WORKER["cf"].n_features), dtype=np.float32)
        position = 0
        for row in range(GRID):
            for column in range(GRID):
                tile = np.ascontiguousarray(
                    array[row * side:(row + 1) * side, column * side:(column + 1) * side])
                out[position] = np.asarray(_WORKER["cf"].get_features(tile),
                                           dtype=np.float32)
                position += 1
        return stem, out, upscaled, ""
    except Exception as error:                        # noqa: BLE001
        return stem, None, False, f"{type(error).__name__}: {error}"


def build_stem_map(image_dir: Path, meta: pd.DataFrame | None) -> dict[str, Path]:
    """stem -> caminho.

    A escolha de estrategia NAO pode depender de "ha colisao de nome no
    disco?" -- essa pergunta e ortogonal ao que importa, que e "o `stem` do
    metadado E o nome de arquivo puro?". Comprovado com 4 datasets (food101,
    minc2500, places_lt, nct_crc): todos usam `stem = f"{{classe}}__{{arquivo}}"`
    (igual o OilPainting) mas NENHUM tem colisao de nome cru entre classes --
    o caminho rapido antigo cairia sempre na chave errada (nome puro) sem
    nunca disparar o alarme de colisao, e devolveria 100% dos stems faltando
    em silencio parcial (aqui virou erro porque o main() confere e aborta,
    mas por sorte de desenho, nao por construçao). O Multitask e o unico
    caso onde `stem` == nome de arquivo puro, entao e o unico onde o caminho
    rapido e correto.
    """
    images = [p for p in image_dir.rglob("*")
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]

    has_meta_cols = meta is not None and {"stem", "style", "filename"} <= set(meta.columns)
    stem_is_bare_filename = (
        has_meta_cols
        and (meta["stem"].astype(str)
             == meta["filename"].astype(str).map(lambda f: Path(f).stem)).all()
    )

    if not has_meta_cols or stem_is_bare_filename:
        found: dict[str, Path] = {}
        duplicates: dict[str, list[Path]] = {}
        for path in sorted(images):
            if path.stem in found:
                duplicates.setdefault(path.stem, [found[path.stem]]).append(path)
            found[path.stem] = path
        if not duplicates:
            return found
        if not has_meta_cols:
            raise SystemExit(f"{len(duplicates)} stems ambiguos em {image_dir} (§8), "
                             f"e nao ha metadado (estilo,arquivo) pra desambiguar: "
                             f"{list(duplicates.items())[:3]}")
        log(f"  {len(duplicates)} stems ambiguos por nome puro (§8) -- "
            f"desambiguando por (estilo, arquivo)")

    by_pair = {(p.parent.name, p.name): p for p in images}
    return {str(r.stem): by_pair[(str(r.style), str(r.filename))]
            for r in meta.itertuples(index=False)
            if (str(r.style), str(r.filename)) in by_pair}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-stems", type=Path, default=None,
                        help="arquivo com um stem por linha: restringe a coorte a "
                             "eles, PRESERVANDO a ordem do registry. Serve para "
                             "remendar um subconjunto sem repetir a corrida inteira; "
                             "a saida vai num --out-dir proprio e e fundida POR STEM.")
    parser.add_argument("--exclude-stems", default="",
                        help="stems separados por virgula pra pular (ex.: "
                             "imagem corrompida conhecida -- o rijksdraw tem uma, "
                             "RP-T-1940-571, ver §8)")
    args = parser.parse_args()

    out_dir = args.out_dir or ROOT / "outputs" / f"{args.dataset}_wndcharm_tiled"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    from registry import LABELS, load_blocks

    log(f"coorte de producao de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset)
    stems = [str(s) for s in cohort.stems]
    if args.only_stems:
        querido = {linha.strip() for linha in args.only_stems.read_text().splitlines()
                   if linha.strip()}
        falta = querido - set(stems)
        if falta:
            raise SystemExit(f"--only-stems: {len(falta)} stems fora da coorte "
                             f"(ex.: {sorted(falta)[:3]})")
        antes = len(stems)
        stems = [s for s in stems if s in querido]
        log(f"  --only-stems: {len(stems)} de {antes}")
    if args.exclude_stems:
        excluded = set(args.exclude_stems.split(","))
        before = len(stems)
        stems = [s for s in stems if s not in excluded]
        log(f"  --exclude-stems: removidas {before - len(stems)} de {before}")
    log(f"  {len(stems)} imagens")
    if args.limit:
        stems = stems[:args.limit]
        log(f"  --limit: {len(stems)}")

    label_path, _, _ = LABELS[args.dataset]
    meta = pd.read_csv(ROOT / label_path, dtype={"stem": str})
    stem_map = build_stem_map(args.image_dir, meta)
    missing = [s for s in stems if s not in stem_map]
    if missing:
        raise SystemExit(f"{len(missing)} stems sem imagem (ex.: {missing[:5]}); "
                         f"extrair com buraco daria bloco desalinhado (§5.18b)")

    from CharmFeatures import CharmFeatures
    names_raw = list(CharmFeatures(verbosity=0).get_feature_names())
    n_features = len(names_raw)
    log(f"  {n_features} features por tile x {GRID*GRID} tiles de "
        f"{STANDARD//GRID}x{STANDARD//GRID} px")

    shards = [(index, stems[index:index + SHARD])
              for index in range(0, len(stems), SHARD)]
    log(f"{len(shards)} shards de ate {SHARD}; {args.workers} workers")

    started = time.time()
    for order, (offset, chunk) in enumerate(shards, 1):
        shard_path = shard_dir / f"shard_{offset:07d}.npz"
        if shard_path.exists():
            continue
        tasks = [(stem, str(stem_map[stem])) for stem in chunk]
        values = np.empty((len(chunk), GRID * GRID, n_features), dtype=np.float32)
        flags = np.zeros(len(chunk), dtype=bool)
        failures: list[tuple[str, str]] = []
        with mp.Pool(args.workers, initializer=_init_worker) as pool:
            for position, (stem, tiles, upscaled, error) in enumerate(
                    pool.imap(_extract_one, tasks, chunksize=2)):
                if tiles is None:
                    failures.append((stem, error))
                else:
                    values[position] = tiles
                    flags[position] = upscaled
        if failures:
            raise SystemExit(f"{len(failures)} falhas no shard {offset} "
                             f"(ex.: {failures[:3]}); consertar e repetir")
        temporary = shard_path.with_suffix(".npz.part")
        with open(temporary, "wb") as handle:
            np.savez(handle, values=values, stems=np.asarray(chunk), upscaled=flags)
        temporary.rename(shard_path)
        done = offset + len(chunk)
        rate = done / max(1e-9, time.time() - started)
        log(f"  shard {order}/{len(shards)} ok  ({rate:.1f} img/s, "
            f"restam ~{(len(stems) - done) / max(1e-9, rate) / 60:.0f} min)")

    log("montando os arrays finais")
    matrix = np.lib.format.open_memmap(
        out_dir / "features.npy", mode="w+", dtype=np.float32,
        shape=(len(stems), GRID * GRID, n_features))
    all_flags = np.zeros(len(stems), dtype=bool)
    cursor, order_ok = 0, []
    for offset, _ in shards:
        blob = np.load(shard_dir / f"shard_{offset:07d}.npz", allow_pickle=False)
        block = blob["values"]
        matrix[cursor:cursor + len(block)] = block
        all_flags[cursor:cursor + len(block)] = blob["upscaled"]
        order_ok.extend(str(s) for s in blob["stems"])
        cursor += len(block)
    matrix.flush()
    if order_ok != stems:
        raise SystemExit("a ordem dos shards nao reproduz cohort.stems -- abortando")

    np.save(out_dir / "stems.npy", np.asarray(stems))
    np.save(out_dir / "upscaled.npy", all_flags)
    (out_dir / "meta.json").write_text(json.dumps({
        "raw_names": names_raw, "standard_side": STANDARD, "grid": GRID,
        "tile_side": STANDARD // GRID,
        "grayscale": "PIL convert L (ITU-R 601-2)",
        "color_transform": "NAO disponivel na implementacao de referencia "
                           "(capi.cpp: 'N.B.: Color is not supported')",
        "library": "CharmFeatures (ctypes libcharm, NIH/NIA)",
        "protocol": "Shamir et al., ACM TAP 7(2), 2010: 600x600, 16 tiles 150x150",
    }, indent=1))

    log(f"gravado {out_dir}/features.npy  {matrix.shape}")
    log(f"  upscaled: {int(all_flags.sum())} de {len(all_flags)} "
        f"({100*all_flags.mean():.1f}%)")
    log(f"custo total: {(time.time() - started) / 60:.1f} min de relogio")


if __name__ == "__main__":
    main()
