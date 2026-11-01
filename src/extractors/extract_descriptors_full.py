"""
Extrai descritores clássicos para todas as imagens do dataset (100k).
Descritores compactos (130 features):
  - Histograma Lab 30 bins × 3 canais = 90
  - Estatísticas Lab (mean, std, skewness × 3) = 9
  - LBP uniforme (radius=2, n_points=16) = 18
  - Haralick via GLCM (13)
Saída: plots_v3/desc_full_features.npy  (N, 130)
       plots_v3/desc_full_stems.npy     (N,)

``--feature-profile rich`` preserva exatamente essas 130 colunas e acrescenta
medidas de cor, textura e borda obtidas da *mesma* imagem.  O objetivo não é
trocar de representação, mas poder medir a curva de saturação dentro da
família ``Descriptors``.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import multiprocessing as mp
import time, sys

from skimage import color as skcolor
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import kurtosis, skew
from skimage.filters import sobel_h, sobel_v

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from slic_graph_features import REGIONS, crop_to_region, pool_image

TARGET     = 128


def _safe_moments(values: np.ndarray) -> list[float]:
    """Momentos/quantis finitos mesmo para imagens quase constantes."""
    flattened = np.asarray(values, dtype=np.float64).ravel()
    return [
        float(flattened.mean()),
        float(flattened.std()),
        float(np.nan_to_num(skew(flattened), nan=0.0)),
        float(np.nan_to_num(kurtosis(flattened), nan=0.0)),
        *[float(np.quantile(flattened, quantile)) for quantile in (0.10, 0.50, 0.90)],
    ]


def compact_feature_names() -> list[str]:
    names = [f"lab_hist_{channel}_{bin_index:02d}"
             for channel in ("L", "a", "b") for bin_index in range(30)]
    names += [f"lab_{channel}_{stat}"
              for channel in ("L", "a", "b") for stat in ("mean", "std", "skew")]
    names += [f"lbp_p16_r2_{index:02d}" for index in range(18)]
    names += [
        f"glcm_d1_{property_name}_{stat}"
        for property_name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
        for stat in ("mean", "std")
    ]
    names += ["glcm_d1_correlation_max"]
    assert len(names) == 130
    return names


def _compact_features(lab: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """The historical 130d descriptor, factored out to keep it byte-for-byte nested."""
    hist = []
    for channel in range(3):
        values, _ = np.histogram(lab[:, :, channel].ravel(), bins=30, density=True)
        hist.append(values)
    feature_hist = np.concatenate(hist).astype(np.float32)

    feature_stats = []
    for channel in range(3):
        values = lab[:, :, channel].ravel().astype(float)
        feature_stats.extend([values.mean(), values.std(), float(np.nan_to_num(skew(values), nan=0.0))])
    feature_stats = np.asarray(feature_stats, dtype=np.float32)

    gray_u8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, P=16, R=2, method="uniform")
    feature_lbp, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 18), density=True)

    glcm = graycomatrix(
        gray_u8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256, symmetric=True, normed=True,
    )
    feature_har = []
    for property_name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        values = graycoprops(glcm, property_name)[0]
        feature_har.extend([values.mean(), values.std()])
    feature_har.append(graycoprops(glcm, "correlation")[0].max())
    feature_har = np.asarray(feature_har[:13], dtype=np.float32)
    return np.concatenate((feature_hist, feature_stats, feature_lbp, feature_har)).astype(np.float32)


def rich_feature_names() -> list[str]:
    names = compact_feature_names()
    names += [
        f"{space}_{channel}_hist16_{bin_index:02d}"
        for space, channels in (("rgb", ("r", "g", "b")), ("hsv", ("h", "s", "v")), ("lab", ("L", "a", "b")))
        for channel in channels for bin_index in range(16)
    ]
    names += [
        f"{space}_{channel}_{stat}"
        for space, channels in (("rgb", ("r", "g", "b")), ("hsv", ("h", "s", "v")), ("lab", ("L", "a", "b")))
        for channel in channels
        for stat in ("mean", "std", "skew", "kurtosis", "q10", "q50", "q90")
    ]
    names += [
        f"{space}_{channel}_hist_entropy"
        for space, channels in (("rgb", ("r", "g", "b")), ("hsv", ("h", "s", "v")), ("lab", ("L", "a", "b")))
        for channel in channels
    ]
    names += [f"lbp_p{points}_r{radius}_{index:02d}"
              for points, radius in ((8, 1), (24, 3)) for index in range(points + 2)]
    names += [
        f"glcm_d{distance}_{property_name}_{stat}"
        for distance in (2, 4)
        for property_name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
        for stat in ("mean", "std")
    ]
    names += [f"gradient_orientation_{index:02d}" for index in range(12)]
    names += [f"gradient_magnitude_{stat}" for stat in ("mean", "std", "skew", "kurtosis", "q10", "q50", "q90")]
    return names


def _rich_features(rgb: np.ndarray, lab: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Expanded but still image-statistics-only descriptor.

    Every appended value is a deterministic statistic of the same 128px view:
    additional colour co-ordinates, texture radii/distances and edge layout.
    """
    values: list[float] = []
    hsv = skcolor.rgb2hsv(rgb)
    for space in (rgb, hsv, lab):
        for channel in range(3):
            histogram, _ = np.histogram(space[:, :, channel].ravel(), bins=16, density=True)
            values.extend(histogram.tolist())
    for space in (rgb, hsv, lab):
        for channel in range(3):
            values.extend(_safe_moments(space[:, :, channel]))
    for space in (rgb, hsv, lab):
        for channel in range(3):
            histogram, _ = np.histogram(space[:, :, channel].ravel(), bins=16, density=False)
            probability = histogram / max(histogram.sum(), 1)
            nonzero = probability[probability > 0]
            values.append(float(-np.sum(nonzero * np.log(nonzero)) / np.log(16)))

    gray_u8 = (gray * 255).astype(np.uint8)
    for points, radius in ((8, 1), (24, 3)):
        lbp = local_binary_pattern(gray_u8, P=points, R=radius, method="uniform")
        histogram, _ = np.histogram(lbp.ravel(), bins=points + 2, range=(0, points + 2), density=True)
        values.extend(histogram.tolist())
    for distance in (2, 4):
        glcm = graycomatrix(
            gray_u8, distances=[distance], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256, symmetric=True, normed=True,
        )
        for property_name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
            property_values = graycoprops(glcm, property_name)[0]
            values.extend([float(property_values.mean()), float(property_values.std())])

    vertical = sobel_v(gray)
    horizontal = sobel_h(gray)
    magnitude = np.hypot(horizontal, vertical)
    orientation = np.mod(np.arctan2(vertical, horizontal), np.pi)
    orientation_histogram, _ = np.histogram(
        orientation.ravel(), bins=12, range=(0, np.pi), weights=magnitude.ravel(), density=False,
    )
    values.extend((orientation_histogram / max(orientation_histogram.sum(), 1)).tolist())
    values.extend(_safe_moments(magnitude))
    result = np.asarray(values, dtype=np.float32)
    if len(result) != len(rich_feature_names()) - len(compact_feature_names()):
        raise RuntimeError("rich descriptor feature-name/value mismatch")
    return result


def feature_names(profile: str) -> list[str]:
    if profile == "compact":
        return compact_feature_names()
    if profile == "rich":
        return rich_feature_names()
    raise ValueError(f"unknown feature profile: {profile}")

def extract(task: tuple[str, Path, str, int, str]):
    stem, path, region, pool_factor, profile = task
    try:
        with Image.open(path) as opened:
            img = crop_to_region(opened.convert('RGB'), region)
            img = pool_image(img, pool_factor)
        # Redimensionar
        w, h = img.size
        if w > TARGET or h > TARGET:
            scale = TARGET / max(w, h)
            img = img.resize((max(1, int(w*scale)), max(1, int(h*scale))),
                             Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        lab = skcolor.rgb2lab(arr)
        gray = skcolor.rgb2gray(arr)

        compact = _compact_features(lab, gray)
        feat = compact if profile == "compact" else np.concatenate((compact, _rich_features(arr, lab, gray)))
        if not np.isfinite(feat).all():
            raise ValueError("non-finite descriptor feature")
        return stem, feat
    except Exception:
        return stem, None

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--meta',      help='Metadata CSV with filename (omit with --manifest)')
    ap.add_argument('--manifest',  help='Derived-image manifest with sample_id/image_relative_path')
    ap.add_argument('--output',    required=True)
    ap.add_argument('--workers',   type=int, default=32)
    ap.add_argument('--region', choices=REGIONS, default='global')
    ap.add_argument('--pool-factor', type=int, default=1)
    ap.add_argument('--feature-profile', choices=('compact', 'rich'), default='compact',
                    help='compact reproduz o vetor histórico de 130d; rich o contém e acrescenta medições.')
    # O §7.1 registra que os alvos de 128/256/1024 nunca tiveram justificativa e
    # que resolução NUNCA foi variada. Aqui ela vira parâmetro, com o default
    # preservando todo resultado anterior. Cuidado ao mexer: LBP (R=1 e R=3) e a
    # distância do GLCM estão em UNIDADES DE PIXEL, então mudar o alvo muda o que
    # é medido, não só o custo -- é a mesma pegadinha do `felz_scale` no §9.3.
    ap.add_argument('--target', type=int, default=128,
                    help='lado maior após o redimensionamento (default 128, o histórico)')
    args = ap.parse_args()
    globals()['TARGET'] = args.target

    OUT_FEATS = Path(args.output) / 'desc_full_features.npy'
    OUT_STEMS = Path(args.output) / 'desc_full_stems.npy'
    Path(args.output).mkdir(parents=True, exist_ok=True)
    N_WORKERS = args.workers

    if bool(args.meta) == bool(args.manifest):
        ap.error('provide exactly one of --meta or --manifest')
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        required = {'sample_id', 'image_relative_path'}
        missing = required - set(manifest.columns)
        if missing:
            ap.error(f'manifest missing columns: {sorted(missing)}')
        paths = [Path(args.image_dir) / path for path in manifest['image_relative_path']]
        missing_paths = [path for path in paths if not path.is_file()]
        if missing_paths:
            ap.error(f'manifest references {len(missing_paths):,} missing images')
        tasks = [
            (str(sample_id), path, args.region, args.pool_factor, args.feature_profile)
            for sample_id, path in zip(manifest['sample_id'], paths, strict=True)
        ]
    else:
        meta = pd.read_csv(args.meta, dtype={'stem': str})
        _images = [p for p in Path(args.image_dir).rglob('*')
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff')]
        # Indexar por (estilo, arquivo) quando o metadado tem as tres colunas,
        # e usar a coluna `stem` como esta. Indexar so pelo nome do arquivo
        # colapsa quadros homonimos de estilos diferentes -- no OilPainting sao
        # 258 pares DISTINTOS que dividem nome, e a versao antiga perdia 378
        # imagens e gravava um stem que nao casava com o das demais
        # modalidades. O ramo antigo fica para os datasets ja extraidos.
        if {'stem', 'style', 'filename'} <= set(meta.columns):
            by_pair = {(p.parent.name, p.name): p for p in _images}
            tasks = [(str(r.stem), by_pair[(str(r.style), str(r.filename))],
                      args.region, args.pool_factor, args.feature_profile)
                     for r in meta.itertuples(index=False)
                     if (str(r.style), str(r.filename)) in by_pair]
        else:
            stems_meta = set(Path(f).stem for f in meta['filename'])
            stem_to_path = {p.stem: p for p in _images}
            tasks = [(s, stem_to_path[s], args.region, args.pool_factor, args.feature_profile)
                     for s in stems_meta if s in stem_to_path]
    log(f'{len(tasks):,} imagens ({args.region}, pool={args.pool_factor}, profile={args.feature_profile}, {len(feature_names(args.feature_profile))}d) para processar com {N_WORKERS} workers')

    stems_out, feats_out = [], []
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for i, (stem, feat) in enumerate(pool.imap_unordered(extract, tasks, chunksize=50)):
            if feat is not None:
                stems_out.append(stem)
                feats_out.append(feat)
            if (i+1) % 5000 == 0:
                elapsed = time.time() - t0
                log(f'{i+1:,}/{len(tasks):,}  ({elapsed/60:.1f} min)')

    X = np.stack(feats_out, axis=0)
    stems_arr = np.array(stems_out)
    np.save(OUT_FEATS, X)
    np.save(OUT_STEMS, stems_arr)
    np.save(Path(args.output) / 'desc_feature_names.npy', np.asarray(feature_names(args.feature_profile)))
    log(f'Salvo: {OUT_FEATS}  shape={X.shape}')
    log(f'Salvo: {OUT_STEMS}  shape={stems_arr.shape}')
    log(f'Total: {(time.time()-t0)/60:.1f} min')
