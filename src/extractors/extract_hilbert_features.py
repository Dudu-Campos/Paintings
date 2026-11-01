"""
extract_hilbert_features.py — Features de série temporal via curva de Hilbert.

Pipeline por imagem:
  1. Resize 128×128 (bilinear)
  2. RGB → Lab
  3. Ordena pixels pela curva de Hilbert → sinal 1D de 16384 pts por canal
  4. 8 sub-bandas: média e std de cada (16 feat por canal)
  5. Média e std globais do sinal (2 feat por canal)
  6. Correlação entre pares de canais Lab (3 feat)
  → 57 features por imagem no perfil ``compact``.

O perfil ``rich`` contém o vetor histórico e acrescenta uma pirâmide de
sub-bandas, estatísticas de incrementos e autocorrelação. Ele continua sendo
uma descrição da mesma série induzida pela curva de Hilbert; não introduz um
novo modelo ou uma representação espacial alternativa.

Saída:
  {output}/hilbert_features.npy   (N, 57) float32
  {output}/hilbert_stems.npy      (N,) stems

Uso:
    python src/extractors/extract_hilbert_features.py \
        --image_dir raw/images \
        --meta data/artemis/wikiart_meta.csv \
        --output outputs/features \
        --workers 32
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from slic_graph_features import REGIONS, crop_to_region, pool_image

# Ver nota em extract_gabor_features.configure: default de produção,
# sobrescrito por CLI só para a varredura de theta-robustez.
N_SIZE = 128
N_BANDS = 8
RICH_BANDS = (4, 16, 32)
AUTOCORRELATION_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

_hilbert_rows = None
_hilbert_cols = None


def _build_hilbert_indices(n):
    """Índices (row, col) em ordem de Hilbert para grade n×n (n potência de 2)."""
    N = n * n
    d = np.arange(N, dtype=np.int64)
    x = np.zeros(N, dtype=np.int32)
    y = np.zeros(N, dtype=np.int32)
    s = 1
    while s < n:
        rx = ((d >> 1) & 1).astype(np.int32)
        ry = ((d & 1) ^ rx).astype(np.int32)
        mask0 = ry == 0
        flip = mask0 & (rx == 1)
        x[flip] = s - 1 - x[flip]
        y[flip] = s - 1 - y[flip]
        x[mask0], y[mask0] = y[mask0].copy(), x[mask0].copy()
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return y, x  # rows, cols


def _init_worker(rows, cols):
    global _hilbert_rows, _hilbert_cols
    _hilbert_rows = rows
    _hilbert_cols = cols


def _safe_skew(values: np.ndarray) -> float:
    centered = values - values.mean()
    scale = values.std()
    if scale <= 1e-12:
        return 0.0
    return float(np.mean(np.power(centered / scale, 3)))


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag >= len(values):
        return 0.0
    left, right = values[:-lag], values[lag:]
    denominator = float(left.std() * right.std())
    if denominator <= 1e-12:
        return 0.0
    return float(np.mean((left - left.mean()) * (right - right.mean())) / denominator)


def compact_feature_names() -> list[str]:
    names = []
    for channel in ("L", "a", "b"):
        names += [f"hilbert_{channel}_band_mean_{index:02d}" for index in range(N_BANDS)]
        names += [f"hilbert_{channel}_band_std_{index:02d}" for index in range(N_BANDS)]
        names += [f"hilbert_{channel}_{stat}" for stat in ("mean", "std")]
    names += [f"hilbert_{left}_{right}_corr" for left, right in (("L", "a"), ("L", "b"), ("a", "b"))]
    # 3 canais x (2*N_BANDS + 2) + 3 correlações. Era `== 57`, o valor de
    # produção com N_BANDS=8; derivar de N_BANDS mantém a guarda viva na
    # varredura de theta-robustez em vez de transformá-la num bloqueio.
    assert len(names) == 3 * (2 * N_BANDS + 2) + 3, (len(names), N_BANDS)
    return names


def rich_feature_names() -> list[str]:
    names = compact_feature_names()
    for channel in ("L", "a", "b"):
        for n_bands in RICH_BANDS:
            names += [f"hilbert_{channel}_band{n_bands}_mean_{index:02d}" for index in range(n_bands)]
            names += [f"hilbert_{channel}_band{n_bands}_std_{index:02d}" for index in range(n_bands)]
        names += [f"hilbert_{channel}_delta_{stat}" for stat in ("mean", "std", "skew", "q10", "q50", "q90", "positive_frac")]
        names += [f"hilbert_{channel}_autocorr_{lag}" for lag in AUTOCORRELATION_LAGS]
    return names


def feature_names(profile: str) -> list[str]:
    if profile == "compact":
        return compact_feature_names()
    if profile == "rich":
        return rich_feature_names()
    raise ValueError(f"unknown feature profile: {profile}")


def _compact_features(signal: np.ndarray) -> list[float]:
    band_size = signal.shape[0] // N_BANDS
    features = []
    for channel in range(3):
        values = signal[:, channel]
        for band in range(N_BANDS):
            features.append(float(values[band * band_size:(band + 1) * band_size].mean()))
        for band in range(N_BANDS):
            features.append(float(values[band * band_size:(band + 1) * band_size].std()))
        features.extend([float(values.mean()), float(values.std())])
    for left, right in ((0, 1), (0, 2), (1, 2)):
        left_values, right_values = signal[:, left], signal[:, right]
        denominator = float(left_values.std() * right_values.std())
        correlation = (
            float(np.mean((left_values - left_values.mean()) * (right_values - right_values.mean())) / denominator)
            if denominator > 1e-12 else 0.0
        )
        features.append(correlation)
    return features


def _rich_features(signal: np.ndarray) -> list[float]:
    features = []
    for channel in range(3):
        values = signal[:, channel]
        for n_bands in RICH_BANDS:
            band_size = len(values) // n_bands
            bands = [values[index * band_size:(index + 1) * band_size] for index in range(n_bands)]
            features.extend(float(band.mean()) for band in bands)
            features.extend(float(band.std()) for band in bands)
        delta = np.diff(values)
        features.extend([
            float(delta.mean()), float(delta.std()), _safe_skew(delta),
            *[float(np.quantile(delta, quantile)) for quantile in (0.10, 0.50, 0.90)],
            float(np.mean(delta > 0)),
        ])
        features.extend(_autocorrelation(values, lag) for lag in AUTOCORRELATION_LAGS)
    return features


def _extract_one(args):
    stem, img_path, region, pool_factor, profile = args
    global _hilbert_rows, _hilbert_cols
    try:
        with Image.open(img_path) as opened:
            img = crop_to_region(opened.convert('RGB'), region)
            img = pool_image(img, pool_factor)
        img = img.resize((N_SIZE, N_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        lab = color.rgb2lab(arr).astype(np.float32)  # (128, 128, 3)

        signal = lab[_hilbert_rows, _hilbert_cols, :]  # (16384, 3)

        compact = _compact_features(signal)
        feats = compact if profile == "compact" else compact + _rich_features(signal)
        features = np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if len(features) != len(feature_names(profile)):
            raise RuntimeError("Hilbert feature-name/value mismatch")
        return stem, features
    except Exception:
        return stem, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--meta',      help='Metadata CSV with filename (omit with --manifest)')
    parser.add_argument('--manifest',  help='Derived-image manifest with sample_id/image_relative_path')
    parser.add_argument('--output',    required=True)
    parser.add_argument('--workers',   type=int, default=32)
    parser.add_argument('--n-size',    type=int, default=N_SIZE)
    parser.add_argument('--n-bands',   type=int, default=N_BANDS)
    parser.add_argument('--region', choices=REGIONS, default='global')
    parser.add_argument('--pool-factor', type=int, default=1)
    parser.add_argument('--feature-profile', choices=('compact', 'rich'), default='compact',
                        help='compact reproduz o vetor histórico de 57d; rich o contém e acrescenta estatísticas temporais.')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if bool(args.meta) == bool(args.manifest):
        parser.error('provide exactly one of --meta or --manifest')
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        required = {'sample_id', 'image_relative_path'}
        missing = required - set(manifest.columns)
        if missing:
            parser.error(f'manifest missing columns: {sorted(missing)}')
        tasks = [
            (str(sample_id), str(image_dir / relative_path), args.region, args.pool_factor, args.feature_profile)
            for sample_id, relative_path in manifest[['sample_id', 'image_relative_path']].itertuples(index=False)
        ]
        if any(not Path(path).is_file() for _, path, _, _ in tasks):
            parser.error('manifest references missing images')
    else:
        meta = pd.read_csv(args.meta, dtype={'stem': str})
        _images = [p for p in image_dir.rglob('*')
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff')]
        # Indexar por (estilo, arquivo) quando o metadado tem as tres colunas,
        # e usar a coluna `stem` como esta. Indexar so pelo nome do arquivo
        # colapsa quadros homonimos de estilos diferentes -- no OilPainting sao
        # 258 pares DISTINTOS que dividem nome, e a versao antiga perdia 378
        # imagens e gravava um stem que nao casava com o das demais
        # modalidades. O ramo antigo fica para os datasets ja extraidos.
        if {'stem', 'style', 'filename'} <= set(meta.columns):
            # Mesmo fallback ja aplicado em extract_cnn_full.py: o casamento por
            # (estilo, arquivo) exige pastas por estilo, e o Multitask tem
            # diretorio plano (images/102257.jpg) com `style` no metadado --
            # p.parent.name vira "images", nada casa, e o script grava ZERO
            # imagens sem erro nenhum. O fallback por nome so vale onde o nome
            # e inequivoco no dataset inteiro; onde ha homonimos (OilPainting:
            # 258 pares distintos com o mesmo nome) o par continua unica via.
            by_pair = {(p.parent.name, p.name): str(p) for p in _images}
            by_name, ambiguous = {}, set()
            for _p in _images:
                if _p.name in by_name:
                    ambiguous.add(_p.name)
                by_name[_p.name] = str(_p)
            for _name in ambiguous:
                by_name.pop(_name, None)
            if ambiguous:
                print(f'{len(ambiguous):,} nomes repetidos: so resolvem por (estilo, arquivo)',
                      flush=True)

            def _resolve(style: str, filename: str):
                return by_pair.get((style, filename)) or by_name.get(filename)

            tasks = [(str(r.stem), _resolve(str(r.style), str(r.filename)),
                      args.region, args.pool_factor, args.feature_profile)
                     for r in meta.itertuples(index=False)
                     if _resolve(str(r.style), str(r.filename))]
        else:
            stem_to_path = {p.stem: str(p) for p in _images}
            meta_stems = set(Path(str(r)).stem for r in meta['filename'])
            tasks = [(s, stem_to_path[s], args.region, args.pool_factor, args.feature_profile)
                     for s in meta_stems if s in stem_to_path]

    # A curva tem de ser reconstruída para o N_SIZE pedido, e os globais têm de
    # ser reatribuídos ANTES do Pool: `_extract_one` lê N_SIZE/N_BANDS do módulo
    # e o fork herda o valor do pai. Reconstruir a curva sem reatribuir N_SIZE
    # daria índices de um tamanho e resize de outro -- IndexError, mas só depois
    # de horas de extração em outro dataset.
    # Só via globals(): criar os nomes locais aqui tornaria N_SIZE local em
    # TODA a função, e o `default=N_SIZE` do argparse acima passaria a ler uma
    # variável ainda não atribuída.
    globals()["N_SIZE"], globals()["N_BANDS"] = args.n_size, args.n_bands
    rows, cols = _build_hilbert_indices(args.n_size)

    t0 = time.time()
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    log(f"{len(tasks)} imagens ({args.region}, profile={args.feature_profile}, {len(feature_names(args.feature_profile))}d), {args.workers} workers")

    stems_out, feats_out = [], []
    with Pool(args.workers, initializer=_init_worker, initargs=(rows, cols)) as pool:
        for i, (stem, feat) in enumerate(
            pool.imap_unordered(_extract_one, tasks, chunksize=64)
        ):
            if feat is not None:
                stems_out.append(stem)
                feats_out.append(feat)
            if (i + 1) % 5000 == 0:
                log(f"{i+1}/{len(tasks)}  ({(time.time()-t0)/60:.1f} min)")

    feats_arr = np.array(feats_out, dtype=np.float32)
    stems_arr = np.array(stems_out)

    np.save(out / 'hilbert_features.npy', feats_arr)
    np.save(out / 'hilbert_stems.npy', stems_arr)
    np.save(out / 'hilbert_feature_names.npy', np.asarray(feature_names(args.feature_profile)))
    log(f"Salvo: {out}/hilbert_features.npy  shape={feats_arr.shape}")
    log(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
