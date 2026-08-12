"""
Extrai descritores clássicos para todas as imagens do dataset (100k).
Descritores (130 features):
  - Histograma Lab 30 bins × 3 canais = 90
  - Estatísticas Lab (mean, std, skewness × 3) = 9
  - LBP uniforme (radius=2, n_points=16) = 18
  - Haralick via GLCM (13)
Saída: plots_v3/desc_full_features.npy  (N, 130)
       plots_v3/desc_full_stems.npy     (N,)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import multiprocessing as mp
import time, sys

from skimage import color as skcolor
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from slic_graph_features import REGIONS, crop_to_region

TARGET     = 128

def extract(task: tuple[str, Path, str]):
    stem, path, region = task
    try:
        with Image.open(path) as opened:
            img = crop_to_region(opened.convert('RGB'), region)
        # Redimensionar
        w, h = img.size
        if w > TARGET or h > TARGET:
            scale = TARGET / max(w, h)
            img = img.resize((max(1, int(w*scale)), max(1, int(h*scale))),
                             Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        lab = skcolor.rgb2lab(arr)
        gray = skcolor.rgb2gray(arr)

        # 1. Histograma Lab (90)
        hist = []
        for c in range(3):
            h, _ = np.histogram(lab[:,:,c].ravel(), bins=30, density=True)
            hist.append(h)
        feat_hist = np.concatenate(hist).astype(np.float32)

        # 2. Estatísticas (9)
        feat_stats = []
        for c in range(3):
            ch = lab[:,:,c].ravel().astype(float)
            feat_stats.extend([ch.mean(), ch.std(), float(skew(ch))])
        feat_stats = np.array(feat_stats, dtype=np.float32)

        # 3. LBP (18)
        gray_u8 = (gray * 255).astype(np.uint8)
        lbp = local_binary_pattern(gray_u8, P=16, R=2, method='uniform')
        feat_lbp, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 18), density=True)
        feat_lbp = feat_lbp.astype(np.float32)

        # 4. Haralick/GLCM (13)
        g8 = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(g8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        props = ['contrast','dissimilarity','homogeneity','energy',
                 'correlation','ASM']
        feat_har = []
        for p in props:
            vals = graycoprops(glcm, p)[0]
            feat_har.extend([vals.mean(), vals.std()])
        # energy mean já incluído; adicionar mais 1 para fechar 13
        feat_har.append(graycoprops(glcm, 'correlation')[0].max())
        feat_har = np.array(feat_har[:13], dtype=np.float32)

        feat = np.concatenate([feat_hist, feat_stats, feat_lbp, feat_har])
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
    args = ap.parse_args()

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
            (str(sample_id), path, args.region)
            for sample_id, path in zip(manifest['sample_id'], paths, strict=True)
        ]
    else:
        meta = pd.read_csv(args.meta)
        stems_meta = set(Path(f).stem for f in meta['filename'])
        stem_to_path = {p.stem: p for p in Path(args.image_dir).rglob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')}
        tasks = [(s, stem_to_path[s], args.region) for s in stems_meta if s in stem_to_path]
    log(f'{len(tasks):,} imagens ({args.region}) para processar com {N_WORKERS} workers')

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
    log(f'Salvo: {OUT_FEATS}  shape={X.shape}')
    log(f'Salvo: {OUT_STEMS}  shape={stems_arr.shape}')
    log(f'Total: {(time.time()-t0)/60:.1f} min')
