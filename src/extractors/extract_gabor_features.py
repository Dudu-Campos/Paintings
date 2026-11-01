"""Banco de filtros de Gabor (Manjunath & Ma, 1996): media + desvio padrao da
magnitude de resposta por (orientacao, frequencia). Descritor classico de
textura, mesma familia de LBP/Haralick ja presentes em Descriptors, mas
faltante (achado 27 do CLAUDE.md: "faltam implementar so Gabor, fractal e
Tamura"). 4 orientacoes x 4 frequencias x 2 estatisticas = 32d.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import multiprocessing as mp
import time, sys

from skimage import color as skcolor
from skimage.filters import gabor_kernel
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from slic_graph_features import REGIONS, crop_to_region, pool_image

# Valores de produção. Ficam como default e são sobrescritos por CLI para a
# análise de theta-robustez (§7.4): a pergunta não é se a transformação é
# robusta a tudo, é se o ponto de operação escolhido a priori é um pico
# isolado -- o que tornaria os resultados frágeis.
TARGET = 128
N_THETAS = 4
FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
THETAS = [i * np.pi / N_THETAS for i in range(N_THETAS)]
KERNELS = []


def configure(target: int, n_thetas: int, frequencies: list[float]) -> None:
    """Reconfigura o banco de filtros. Chamado antes de qualquer extração."""
    global TARGET, N_THETAS, FREQUENCIES, THETAS, KERNELS
    TARGET, N_THETAS, FREQUENCIES = target, n_thetas, list(frequencies)
    THETAS = [i * np.pi / N_THETAS for i in range(N_THETAS)]
    KERNELS = [gabor_kernel(f, theta=t).astype(np.complex64)
               for t in THETAS for f in FREQUENCIES]


configure(TARGET, N_THETAS, FREQUENCIES)


def feature_names() -> list[str]:
    names = []
    for theta_idx in range(len(THETAS)):
        for freq in FREQUENCIES:
            names.append(f"gabor_t{theta_idx}_f{freq:.1f}_mean")
            names.append(f"gabor_t{theta_idx}_f{freq:.1f}_std")
    return names


def gabor_features(gray: np.ndarray) -> np.ndarray:
    feats = np.empty(len(KERNELS) * 2, dtype=np.float32)
    for i, kernel in enumerate(KERNELS):
        response = fftconvolve(gray, kernel, mode="same")
        magnitude = np.abs(response)
        feats[2 * i] = magnitude.mean()
        feats[2 * i + 1] = magnitude.std()
    return feats


def extract(task: tuple[str, Path, str, int]):
    stem, path, region, pool_factor = task
    try:
        with Image.open(path) as opened:
            img = crop_to_region(opened.convert("RGB"), region)
            img = pool_image(img, pool_factor)
        w, h = img.size
        if w > TARGET or h > TARGET:
            scale = TARGET / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        gray = skcolor.rgb2gray(arr)
        feat = gabor_features(gray)
        if not np.isfinite(feat).all():
            raise ValueError("non-finite gabor feature")
        return stem, feat
    except Exception:
        return stem, None


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--meta", help="Metadata CSV with filename (omit with --manifest)")
    ap.add_argument("--manifest", help="Derived-image manifest with sample_id/image_relative_path")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--n-thetas", type=int, default=N_THETAS)
    ap.add_argument("--frequencies", type=float, nargs="+", default=FREQUENCIES)
    ap.add_argument("--region", choices=REGIONS, default="global")
    ap.add_argument("--pool-factor", type=int, default=1)
    args = ap.parse_args()
    # Antes de qualquer Pool: no Linux o fork herda estes globais, então
    # reconfigurar aqui basta para os workers. Se algum dia isto virar `spawn`,
    # a configuração tem de ir para um initializer -- e a falha seria SILENCIOSA
    # (os filhos reimportariam o módulo com o default de produção).
    configure(args.target, args.n_thetas, args.frequencies)

    OUT_FEATS = Path(args.output) / "gabor_features.npy"
    OUT_STEMS = Path(args.output) / "gabor_stems.npy"
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if bool(args.meta) == bool(args.manifest):
        ap.error("provide exactly one of --meta or --manifest")
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        required = {"sample_id", "image_relative_path"}
        missing = required - set(manifest.columns)
        if missing:
            ap.error(f"manifest missing columns: {sorted(missing)}")
        paths = [Path(args.image_dir) / path for path in manifest["image_relative_path"]]
        missing_paths = [path for path in paths if not path.is_file()]
        if missing_paths:
            ap.error(f"manifest references {len(missing_paths):,} missing images")
        tasks = [
            (str(sample_id), path, args.region, args.pool_factor)
            for sample_id, path in zip(manifest["sample_id"], paths, strict=True)
        ]
    else:
        meta = pd.read_csv(args.meta, dtype={"stem": str})
        _images = [p for p in Path(args.image_dir).rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")]
        # Indexar por (estilo, arquivo) quando o metadado tem as tres colunas,
        # e usar a coluna `stem` como esta. Indexar so pelo nome do arquivo
        # colapsa quadros homonimos de estilos diferentes -- no OilPainting sao
        # 258 pares DISTINTOS que dividem nome, e a versao antiga perdia 378
        # imagens e gravava um stem que nao casava com o das demais
        # modalidades. O ramo antigo fica para os datasets ja extraidos.
        if {"stem", "style", "filename"} <= set(meta.columns):
            by_pair = {(p.parent.name, p.name): p for p in _images}
            tasks = [(str(r.stem), by_pair[(str(r.style), str(r.filename))],
                      args.region, args.pool_factor)
                     for r in meta.itertuples(index=False)
                     if (str(r.style), str(r.filename)) in by_pair]
        else:
            stems_meta = set(Path(f).stem for f in meta["filename"])
            stem_to_path = {p.stem: p for p in _images}
            tasks = [(s, stem_to_path[s], args.region, args.pool_factor)
                     for s in stems_meta if s in stem_to_path]
    log(f"{len(tasks):,} imagens ({args.region}, pool={args.pool_factor}, "
        f"{len(feature_names())}d) para processar com {args.workers} workers")

    stems_out, feats_out = [], []
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, (stem, feat) in enumerate(pool.imap_unordered(extract, tasks, chunksize=50)):
            if feat is not None:
                stems_out.append(stem)
                feats_out.append(feat)
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                log(f"{i+1:,}/{len(tasks):,}  ({elapsed/60:.1f} min)")

    X = np.stack(feats_out, axis=0)
    stems_arr = np.array(stems_out)
    np.save(OUT_FEATS, X)
    np.save(OUT_STEMS, stems_arr)
    np.save(Path(args.output) / "gabor_feature_names.npy", np.asarray(feature_names()))
    log(f"Salvo: {OUT_FEATS}  shape={X.shape}")
    log(f"Total: {(time.time()-t0)/60:.1f} min")
