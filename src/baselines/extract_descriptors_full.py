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

IMAGE_DIR  = Path('./raw/images')
META_PATH  = Path('./meta/meta.csv')
OUT_FEATS  = Path('./plots_v3/desc_full_features.npy')
OUT_STEMS  = Path('./plots_v3/desc_full_stems.npy')
TARGET     = 128
N_WORKERS  = 32

def extract(path: Path):
    try:
        img = Image.open(path).convert('RGB')
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
        return path.stem, feat
    except Exception:
        return path.stem, None

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

if __name__ == '__main__':
    meta = pd.read_csv(META_PATH)
    stems_meta = set(Path(f).stem for f in meta['filename'])
    paths = [p for p in IMAGE_DIR.glob('*.jpg') if p.stem in stems_meta]
    log(f'{len(paths):,} imagens para processar com {N_WORKERS} workers')

    stems_out, feats_out = [], []
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for i, (stem, feat) in enumerate(pool.imap_unordered(extract, paths, chunksize=50)):
            if feat is not None:
                stems_out.append(stem)
                feats_out.append(feat)
            if (i+1) % 5000 == 0:
                elapsed = time.time() - t0
                log(f'{i+1:,}/{len(paths):,}  ({elapsed/60:.1f} min)')

    X = np.stack(feats_out, axis=0)
    stems_arr = np.array(stems_out)
    np.save(OUT_FEATS, X)
    np.save(OUT_STEMS, stems_arr)
    log(f'Salvo: {OUT_FEATS}  shape={X.shape}')
    log(f'Salvo: {OUT_STEMS}  shape={stems_arr.shape}')
    log(f'Total: {(time.time()-t0)/60:.1f} min')