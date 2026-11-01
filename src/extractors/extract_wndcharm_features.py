"""WND-CHARM: 2.895 features de uma hierarquia composta de transformadas.

Orlov, Shamir, Macura, Johnston, Eckley & Goldberg, *Pattern Recognition
Letters* 29(11):1684-1693, 2008. Implementação OFICIAL (`CharmFeatures`, ctypes
sobre `libcharm`, grupo do NIH/NIA) -- **não** é reimplementação nossa, e isso é
deliberado: o §7.11b registra o custo de chamar de reprodução algo que não era.

    15 algoritmos x 12 cadeias de transformada = 2.895 features

    algoritmos   Zernike (576) · Comb Moments (576) · Haralick (312) ·
                 Multiscale Histograms (288) · Chebyshev-Fourier (256) ·
                 Chebyshev (256) · Fractal (240) · Radon (144) · Tamura (72) ·
                 Pixel Intensity Statistics (60) · Otsu Object (34) ·
                 Inverse-Otsu Object (34) · Edge (28) · Gini (12) · Gabor (7)

    transformadas  cru · Fourier · Wavelet · Chebyshev · Edge ·
                   Fourier(Wavelet) · Fourier(Edge) · Wavelet(Edge) ·
                   Chebyshev(Fourier) · Wavelet(Fourier) · Fourier(Chebyshev) ·
                   Chebyshev(Wavelet)

**O BANCO É MONOCROMÁTICO. Isto é a ressalva central e tem de ir em qualquer
tabela.** A API recebe matriz 2D e não há uma única feature de cor nas 2.895
(conferido por busca em `color|colour|hsv|rgb|hue|satur`: zero). Do nosso lado,
99 das 130 colunas do `Descriptors` são cor -- 90 de histograma Lab e 9 de
estatísticas (§10.2) -- e as quatro views de grafo pesam aresta por `color_dist`
(§9.3). Sem um braço de controle com a cor removida do nosso lado, a comparação
confunde "mais famílias" com "tem cor", e a conclusão não é atribuível.

Duas escolhas de protocolo, declaradas porque nenhuma é neutra:

**Resolução 256 px de lado máximo.** É a resolução de produção do nosso Fourier,
e mais generosa que os 128 px do Descriptors/Hilbert/Gabor (§5.4b). Medido nesta
máquina, o custo single-thread é 6,7 / 19,9 / 77,2 CPU-h a 128 / 256 / 512 px
para a coorte de 99.816 -- a 256 px ficamos em ~metade das 43,2 CPU-h que o
§5.4b mede para o nosso banco inteiro, então a escolha não é de economia. As
contagens de coeficiente de Zernike e Chebyshev são ordens polinomiais e **não**
dependem da resolução, então subir resolução custa tempo e não dimensão.

**Luminância por ITU-R 601-2** (`PIL.Image.convert("L")`), uma passada, sem
tiling. O uso canônico do `wndchrm` é por tiles (`-t4`), o que muda a unidade de
predição -- aqui a unidade tem de ser a imagem, para casar com a nossa coorte.

**Lê os ORIGINAIS do NFS, nunca a cópia local.** O §9.0 registra que
`/tmp/prism_data/multitask_painting100k/images` está com o lado MENOR reduzido a
256, ou seja é derivada e não cópia; e o §5.18g mede 0,84pp só de recompressão
JPEG. Reamostrar uma imagem já reamostrada introduziria exatamente esse viés.

**A ordem das linhas vem de `cohort.stems`**, nunca de um glob nem do metadado.
O §5.18b custou −2,92pp por reconstruir stems do metadado: o conjunto era o
mesmo, a ORDEM não, e o consumidor casa por stem sem reclamar.

Dependência: `pip install CharmFeatures` (wheel pré-compilada, cpython-312).
Instalada fora do venv do projeto? aponte `PYTHONPATH` para o diretório alvo.

Uso:
    python src/extractors/extract_wndcharm_features.py \
        --dataset multitask \
        --image-dir data/multitask_painting100k/original/images \
        --max-side 256 --workers 28
"""
from __future__ import annotations

import os

# Antes de numpy/scipy: cada worker é single-thread, senão 28 workers x N threads
# de BLAS brigam entre si e a medição de custo deixa de valer (§1).
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

# Varreduras de museu passam do limite anti-bomba do PIL (§8).
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

SHARD = 2000
_WORKER: dict = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sanitise(name: str) -> str:
    """`Edge Features (Fourier (Wavelet ())) [3]` -> nome seguro de coluna.

    Os nomes crus têm espaço, parêntese e colchete. O parquet aceita, mas o
    `registry.select()` filtra por nome de coluna e o `Representation` carimba
    prefixo, então um nome regular vale mais. Os nomes EXATOS vão para um
    sidecar `.json`, na ordem das colunas -- mesma convenção do
    `hilbert_feature_names.npy`.
    """
    out = []
    for char in name:
        out.append(char if char.isalnum() else "_")
    collapsed = "_".join(part for part in "".join(out).split("_") if part)
    return f"wndcharm_{collapsed}"


def _init_worker(max_side: int) -> None:
    from CharmFeatures import CharmFeatures

    # O forking interno da biblioteca é ~4x mais rápido em processo ÚNICO, mas
    # dentro de um Pool ele multiplica processos (28 x N) e a máquina é
    # compartilhada (§1). Um nível de paralelismo só, o nosso.
    _WORKER["cf"] = CharmFeatures(verbosity=0, forking_executor=False,
                                  forking_gabor=False, forking_haralick=False)
    _WORKER["max_side"] = max_side


def _extract_one(task: tuple[str, str]) -> tuple[str, np.ndarray | None, str]:
    stem, path = task
    try:
        image = Image.open(path).convert("L")
        width, height = image.size
        scale = _WORKER["max_side"] / max(width, height)
        if scale < 1.0:
            image = image.resize((max(1, int(width * scale)),
                                  max(1, int(height * scale))), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
        values = np.asarray(_WORKER["cf"].get_features(array), dtype=np.float32)
        return stem, values, ""
    except Exception as error:                       # noqa: BLE001
        return stem, None, f"{type(error).__name__}: {error}"


def build_stem_map(image_dir: Path, meta: pd.DataFrame | None) -> dict[str, Path]:
    """stem -> caminho. Tenta por NOME DE ARQUIVO puro primeiro (rapido, e o
    que datasets de diretorio plano como o Multitask precisam -- la
    `parent.name` e sempre "images", nunca bate com `style`). So cai para
    parear por (estilo, arquivo) se houver colisao de verdade: o OilPainting
    tem 258 pares de nomes repetidos em estilos diferentes (§8), e so ali o
    parent.name (a subpasta de estilo) resolve a ambiguidade.
    """
    images = [p for p in image_dir.rglob("*")
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]

    found: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(images):
        if path.stem in found:
            duplicates.setdefault(path.stem, [found[path.stem]]).append(path)
        found[path.stem] = path

    if not duplicates:
        return found

    if meta is None or not {"stem", "style", "filename"} <= set(meta.columns):
        sample = list(duplicates.items())[:3]
        raise SystemExit(
            f"{len(duplicates)} stems ambiguos em {image_dir}, e nao ha metadado "
            f"(estilo,arquivo) pra desambiguar -- indexar por nome de arquivo aqui "
            f"gravaria coorte errada sem reclamar (§8). exemplos: {sample}")

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
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="so as N primeiras linhas da coorte; para teste")
    args = parser.parse_args()

    shard_dir = args.shard_dir or ROOT / "outputs" / f"{args.dataset}_wndcharm"
    output = args.output or ROOT / "results" / f"wndcharm_{args.dataset}_features.parquet"
    shard_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    from registry import LABELS, load_blocks

    log(f"coorte de producao de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset)
    stems = [str(s) for s in cohort.stems]
    log(f"  {len(stems)} imagens, {cohort.dimensions}d em {len(cohort.blocks)} blocos")
    if args.limit:
        stems = stems[:args.limit]
        log(f"  --limit: so as {len(stems)} primeiras")

    log(f"mapa stem -> caminho em {args.image_dir}")
    label_path, _, _ = LABELS[args.dataset]
    meta = pd.read_csv(ROOT / label_path, dtype={"stem": str})
    stem_map = build_stem_map(args.image_dir, meta)
    missing = [s for s in stems if s not in stem_map]
    if missing:
        raise SystemExit(
            f"{len(missing)} stems da coorte sem imagem em disco "
            f"(ex.: {missing[:5]}). Extrair com buraco produziria bloco "
            f"desalinhado que nada na saida denuncia (§5.18b).")
    log(f"  {len(stem_map)} arquivos indexados, 0 stems faltando")

    from CharmFeatures import CharmFeatures
    names_raw = list(CharmFeatures(verbosity=0).get_feature_names())
    columns = [sanitise(n) for n in names_raw]
    if len(set(columns)) != len(columns):
        raise SystemExit("saneamento de nomes colidiu -- corrigir `sanitise`")
    log(f"  {len(columns)} features declaradas pela biblioteca")

    shards = [(index, stems[index:index + SHARD])
              for index in range(0, len(stems), SHARD)]
    log(f"{len(shards)} shards de ate {SHARD}; {args.workers} workers; "
        f"lado maximo {args.max_side} px")

    started = time.time()
    for order, (offset, chunk) in enumerate(shards, 1):
        shard_path = shard_dir / f"shard_{offset:07d}.npz"
        if shard_path.exists():
            log(f"  shard {order}/{len(shards)} ja existe, pulando")
            continue
        tasks = [(stem, str(stem_map[stem])) for stem in chunk]
        values = np.empty((len(chunk), len(columns)), dtype=np.float32)
        failures: list[tuple[str, str]] = []
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.max_side,)) as pool:
            for position, (stem, vector, error) in enumerate(
                    pool.imap(_extract_one, tasks, chunksize=4)):
                if vector is None:
                    failures.append((stem, error))
                    values[position] = np.nan
                else:
                    values[position] = vector
        if failures:
            raise SystemExit(
                f"{len(failures)} falhas no shard {offset} (ex.: {failures[:3]}). "
                f"Gravar shard com NaN esconderia o problema; consertar e repetir.")
        # `np.savez_compressed` acrescenta `.npz` a quem nao termina nisso, e o
        # rename de um `.tmp` falharia (§8) -- gravar por handle.
        temporary = shard_path.with_suffix(".npz.part")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, values=values, stems=np.asarray(chunk))
        temporary.rename(shard_path)
        done = offset + len(chunk)
        rate = done / max(1e-9, time.time() - started)
        log(f"  shard {order}/{len(shards)} ok  ({rate:.1f} img/s, "
            f"restam ~{(len(stems) - done) / max(1e-9, rate) / 60:.0f} min)")

    log("montando o parquet a partir dos shards")
    parts, part_stems = [], []
    for offset, _ in shards:
        blob = np.load(shard_dir / f"shard_{offset:07d}.npz", allow_pickle=False)
        parts.append(blob["values"])
        part_stems.extend(str(s) for s in blob["stems"])
    matrix = np.vstack(parts)
    if part_stems != stems:
        raise SystemExit("a ordem dos shards nao reproduz cohort.stems -- abortando")

    frame = pd.DataFrame(matrix, columns=columns)
    frame.insert(0, "stem", stems)
    frame.to_parquet(output, index=False)

    sidecar = output.with_suffix(".names.json")
    sidecar.write_text(json.dumps(
        {"raw_names": names_raw, "columns": columns,
         "max_side": args.max_side, "grayscale": "PIL convert L (ITU-R 601-2)",
         "library": "CharmFeatures (ctypes libcharm, NIH/NIA)",
         "reference": "Orlov et al., Pattern Recognition Letters 29(11), 2008"},
        indent=1))

    finite = int(np.isfinite(matrix).all(axis=0).sum())
    constant = int((matrix.std(axis=0) == 0).sum())
    log(f"gravado {output}  {matrix.shape[0]} x {matrix.shape[1]}")
    log(f"  colunas totalmente finitas {finite}/{matrix.shape[1]}  "
        f"constantes na coorte {constant}")
    log(f"  nomes exatos em {sidecar}")
    log(f"custo total: {(time.time() - started) / 60:.1f} min de relogio")


if __name__ == "__main__":
    main()
