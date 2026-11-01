"""
compare_baselines_filtered.py — Compara RAG/kNN vs CNN vs descritores clássicos
no mesmo subconjunto de imagens: as 3.648 imagens filtradas (n_nodes <= 100k,
estilos com >= 30 amostras, balanceado min=31/max=100).

Protocolo uniforme para todos os métodos:
  - LDA (priors uniformes) e XGBoost (sample_weight balanceado)
  - BAcc (balanced accuracy) como métrica principal
  - 5-fold CV estratificado, random_state=42

Uso:
    python compare_baselines_filtered.py \
        --graph_parquet  ./plots_v3/graph_features.parquet \
        --knn_parquet    ./plots_v3/knn_features.parquet \
        --image_dir      ./raw/images \
        --output         ./plots_v3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Para feature sets de alta dimensão (CNN 2048-d), reduz para 200 componentes
# antes do XGBoost — evita custo O(n_features) desnecessário (ResNet50 GAP
# tem correlação alta entre canais, PCA captura >95% da variância em ~100-150 PCs)
XGB_PCA_THRESHOLD = 200
XGB_PCA_COMPONENTS = 200

sys.path.insert(0, str(Path(__file__).parent))
from baseline_descriptors import extract_descriptors
from baseline_cnn import extract_cnn_features

GRAPH_FEATURE_COLS = [
    'n_nodes', 'n_edges', 'density',
    'area_gini', 'area_entropy', 'area_max_frac', 'area_mean', 'area_std',
    'color_dist_mean', 'color_dist_std', 'color_dist_max',
    'spatial_dist_mean', 'spatial_dist_std',
    'degree_mean', 'degree_std',
]

STRUCT_FEATURE_COLS = [
    'degree_max', 'hub_degree_ratio', 'n_leaf_frac',
    'degree_gini', 'area_second_max_frac', 'area_cv',
]

SPATIAL_FEATURE_COLS = ['spatial_density_cv', 'spatial_density_max_block_frac']

ALL_STRUCT_COLS = STRUCT_FEATURE_COLS + SPATIAL_FEATURE_COLS  # 8 features

COLOR_FEATURE_COLS = [
    'L_mean_w', 'L_std_w',
    'a_mean_w', 'a_std_w',
    'b_mean_w', 'b_std_w',
    'chroma_mean_w', 'chroma_std_w',
    'warm_frac',
]

_PER_WIN = GRAPH_FEATURE_COLS + STRUCT_FEATURE_COLS + SPATIAL_FEATURE_COLS  # 23
PYRAMID_FEATURE_COLS = [
    f'q{r}{c}_{f}'
    for r in range(2) for c in range(2)
    for f in _PER_WIN
]  # 92 features

MAX_NODES = 100_000
MIN_COUNT = 30

# Parâmetros XGBoost fixos — mesmos do ablation, sem busca para manter
# o tempo razoável na comparação com CNN (2048 features × 3.6k amostras)
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    objective='multi:softmax',
    eval_metric='mlogloss',
    n_jobs=1,
    random_state=42,
)


def load_filtered_graph(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = df[df['style'] != 'unknown']
    df = df[df['n_nodes'] <= MAX_NODES]
    counts = df['style'].value_counts()
    df = df[df['style'].isin(counts[counts >= MIN_COUNT].index)]
    return df.reset_index(drop=True)


def impute(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(float)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X


def run_lda_bacc(X: np.ndarray, y: np.ndarray, label: str) -> tuple[float, float]:
    X = impute(X)
    styles = np.unique(y)
    n_styles = len(styles)
    n_comps = min(n_styles - 1, X.shape[1])
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(n_components=n_comps,
                                           priors=np.ones(n_styles) / n_styles)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv,
                              scoring='balanced_accuracy', n_jobs=1)
    mean, std = float(scores.mean()), float(scores.std())
    print(f'    LDA  [{label}]  BAcc={mean:.4f} ± {std:.4f}')
    return mean, std


def run_xgb_bacc(X: np.ndarray, y: np.ndarray, label: str) -> tuple[float, float]:
    """
    5-fold CV com XGBoost e sample_weight balanceado por fold.
    Para feature sets com dim > XGB_PCA_THRESHOLD, aplica PCA antes do
    XGBoost — evita o custo desnecessário em embeddings correlacionados
    (ex: ResNet50 GAP de 2048-d).
    """
    X = impute(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    use_pca = X.shape[1] > XGB_PCA_THRESHOLD
    if use_pca:
        n_comp = min(XGB_PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
        print(f'    XGB  [{label}]  PCA {X.shape[1]}→{n_comp} antes do XGBoost')

    for train_idx, val_idx in cv.split(X, y_enc):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        if use_pca:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            pca = PCA(n_components=n_comp, random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_val = pca.transform(X_val)

        sw = compute_sample_weight('balanced', y_tr)
        clf = XGBClassifier(**XGB_PARAMS, num_class=n_classes)
        clf.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        pred = clf.predict(X_val)
        fold_scores.append(balanced_accuracy_score(y_val, pred))

    mean, std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    print(f'    XGB  [{label}]  BAcc={mean:.4f} ± {std:.4f}')
    return mean, std


def extract_descriptors_batch(paths: list[str]) -> tuple[np.ndarray, list[int]]:
    feats, valid = [], []
    for i, p in enumerate(paths):
        f = extract_descriptors(p)
        if f is not None:
            feats.append(f)
            valid.append(i)
        if (i + 1) % 500 == 0:
            print(f'  descritores: {i+1:>5}/{len(paths):,}...', end='\r')
    print()
    return np.stack(feats), valid


def align_to_valid(X_full: np.ndarray, valid_idx: list[int],
                   X_ref: np.ndarray, y_ref: np.ndarray):
    """
    Retorna (X_full[valid], X_ref[valid], y_ref[valid]) alinhados pelo
    subconjunto de índices válidos. Usado para combinar feature sets quando
    um deles pode ter falhas de leitura de imagem.
    """
    idx = np.array(valid_idx)
    return X_full, X_ref[idx], y_ref[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_parquet',   default='./plots_v3/graph_features.parquet')
    parser.add_argument('--knn_parquet',     default='./plots_v3/knn_features.parquet')
    parser.add_argument('--pyramid_parquet', default='./plots_v3/graph_features_pyramid.parquet')
    parser.add_argument('--image_dir',     default='./raw/images')
    parser.add_argument('--output',        default='./plots_v3')
    parser.add_argument('--device',        default='cuda')
    parser.add_argument('--batch_size',    type=int, default=128)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Dataset filtrado (âncora de stems/labels) ─────────────────────────────
    print('Carregando dataset filtrado...')
    gf = load_filtered_graph(args.graph_parquet)
    stems  = gf['stem'].astype(str).tolist()
    y_full = np.array(gf['style'].tolist())
    image_dir = Path(args.image_dir)
    paths = [str(image_dir / f'{s}.jpg') for s in stems]
    missing = sum(1 for p in paths if not Path(p).exists())
    if missing:
        print(f'AVISO: {missing} imagens não encontradas em {image_dir}')

    # Features completas: 15 básicas + 8 estruturais + 9 cor = 32
    struct_lookup  = gf.set_index('stem')[ALL_STRUCT_COLS + COLOR_FEATURE_COLS]
    ALL_GRAPH_COLS = GRAPH_FEATURE_COLS + ALL_STRUCT_COLS + COLOR_FEATURE_COLS
    X_rag_struct   = gf[ALL_GRAPH_COLS].to_numpy().astype(float)

    # Pirâmide 2×2 (92 features)
    X_pyramid = None
    if Path(args.pyramid_parquet).exists():
        pf = load_filtered_graph(args.pyramid_parquet)
        pf = pf.set_index('stem').reindex(stems).reset_index()
        X_pyramid = pf[PYRAMID_FEATURE_COLS].to_numpy().astype(float)

    # kNN — join das colunas estruturais+cor do RAG (mesmas regiões de cor)
    X_knn_struct = None
    if Path(args.knn_parquet).exists():
        kf     = load_filtered_graph(args.knn_parquet)
        kf     = kf.set_index('stem').reindex(stems).reset_index()
        kf_aug = kf.set_index('stem').join(struct_lookup, how='left').reset_index()
        X_knn_struct = kf_aug[ALL_GRAPH_COLS].to_numpy().astype(float)

    # ── Descritores (com cache) ───────────────────────────────────────────────
    desc_cache     = out / 'desc_features_filtered.npy'
    desc_idx_cache = out / 'desc_valid_idx_filtered.npy'
    if desc_cache.exists():
        print('\n=== Descritores clássicos (cache) ===')
        X_desc  = np.load(str(desc_cache))
        v_desc  = np.load(str(desc_idx_cache)).tolist()
    else:
        print('\n=== Descritores clássicos (extraindo...) ===')
        X_desc, v_desc = extract_descriptors_batch(paths)
        np.save(str(desc_cache), X_desc)
        np.save(str(desc_idx_cache), np.array(v_desc))
    y_desc = y_full[v_desc]

    # ── CNN ResNet50 (com cache) ──────────────────────────────────────────────
    cnn_cache     = out / 'cnn_resnet50_features_filtered.npy'
    cnn_idx_cache = out / 'cnn_resnet50_valid_idx_filtered.npy'
    if cnn_cache.exists():
        print('\n=== CNN ResNet50 (cache) ===')
        X_cnn  = np.load(str(cnn_cache))
        v_cnn  = np.load(str(cnn_idx_cache)).tolist()
    else:
        print('\n=== CNN ResNet50 (extraindo...) ===')
        import torch
        device = args.device if torch.cuda.is_available() else 'cpu'
        print(f'  device: {device}  batch_size: {args.batch_size}')
        X_cnn, v_cnn = extract_cnn_features(
            paths, model_name='resnet50', device=device,
            batch_size=args.batch_size)
        np.save(str(cnn_cache), X_cnn)
        np.save(str(cnn_idx_cache), np.array(v_cnn))
    y_cnn = y_full[v_cnn]

    # RAG+estrutural+desc alinhado ao subconjunto válido dos descritores
    _, X_rag_for_combo, y_combo = align_to_valid(X_desc, v_desc, X_rag_struct, y_full)
    X_rag_desc = np.hstack([X_rag_for_combo, X_desc])

    # ── Avaliação ─────────────────────────────────────────────────────────────
    scenarios = [
        ('RAG+struct+cor (32)',     X_rag_struct, y_full),
        ('kNN+struct+cor (32)',     X_knn_struct, y_full),
        ('Descritores (130)',       X_desc,       y_desc),
        ('CNN ResNet50 (2048)',     X_cnn,        y_cnn),
        ('RAG+cor+Desc (162)',      X_rag_desc,   y_combo),
    ]

    results = {}
    for name, X, y in scenarios:
        if X is None:
            continue
        n_feat = X.shape[1]
        n_samp = len(y)
        print(f'\n=== {name}  ({n_feat} features, {n_samp:,} amostras) ===')
        lda_mean, lda_std = run_lda_bacc(X, y, name)
        xgb_mean, xgb_std = run_xgb_bacc(X, y, name)
        results[name] = {
            'n_features': n_feat, 'n_samples': n_samp,
            'lda_bacc': lda_mean, 'lda_std': lda_std,
            'xgb_bacc': xgb_mean, 'xgb_std': xgb_std,
        }

    # ── Tabela final ──────────────────────────────────────────────────────────
    hdr = f'{"Método":<22} {"Feat":>5} {"n":>5}   {"LDA BAcc":>10}   {"XGB BAcc":>10}'
    sep = '─' * len(hdr)
    print(f'\n{sep}\n{hdr}\n{sep}')
    for name, r in results.items():
        print(f'{name:<22} {r["n_features"]:>5} {r["n_samples"]:>5}   '
              f'{r["lda_bacc"]:>6.4f}±{r["lda_std"]:.4f}   '
              f'{r["xgb_bacc"]:>6.4f}±{r["xgb_std"]:.4f}')
    print(sep)
    print('BAcc = balanced accuracy, priors uniformes (LDA) / sample_weight (XGB)')
    print('Mesmo conjunto de imagens para todos os métodos. 5-fold CV estratificado.')

    rows = [{'method': name, **r} for name, r in results.items()]
    pd.DataFrame(rows).to_csv(out / 'comparison_baselines_filtered.csv', index=False)
    print(f'\nSalvo em {out / "comparison_baselines_filtered.csv"}')


if __name__ == '__main__':
    main()