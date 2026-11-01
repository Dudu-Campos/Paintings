"""
plot_confusion_matrices.py — Matrizes de confusão lado a lado para os modelos atuais.

4 painéis:
  - RAG+estrutural (23 feat)  — LDA  (melhor classifier deste conjunto)
  - kNN+estrutural (23 feat)  — LDA
  - Descritores (130 feat)    — XGB  (melhor classifier deste conjunto)
  - CNN ResNet50 (2048 feat)  — XGB

Normalização por linha (recall por classe).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

# ── Constantes ────────────────────────────────────────────────────────────────

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
ALL_STRUCT_COLS = STRUCT_FEATURE_COLS + SPATIAL_FEATURE_COLS
ALL_FEAT_COLS   = GRAPH_FEATURE_COLS + ALL_STRUCT_COLS

MAX_NODES = 100_000
MIN_COUNT = 30

XGB_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.5, reg_lambda=2.0,
    objective='multi:softmax', eval_metric='mlogloss',
    n_jobs=1, random_state=42,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_filtered(parquet_path):
    df = pd.read_parquet(parquet_path)
    df = df[df['style'] != 'unknown']
    df = df[df['n_nodes'] <= MAX_NODES]
    counts = df['style'].value_counts()
    df = df[df['style'].isin(counts[counts >= MIN_COUNT].index)]
    return df.reset_index(drop=True)


def impute(X):
    X = X.copy().astype(float)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        if m.any():
            X[m, j] = np.nanmedian(X[:, j])
    return X


def cv_confusion(X, y_str, clf_type='lda'):
    X = impute(X)
    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_true, all_pred = [], []

    for tr, val in cv.split(X, y):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr],  y[val]

        if clf_type == 'lda':
            n_comps = min(n_classes - 1, X_tr.shape[1])
            pipe = Pipeline([
                ('sc',  StandardScaler()),
                ('lda', LinearDiscriminantAnalysis(
                    n_components=n_comps,
                    priors=np.ones(n_classes) / n_classes)),
            ])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_val)

        else:  # xgb
            if X_tr.shape[1] > 200:
                sc    = StandardScaler()
                X_tr  = sc.fit_transform(X_tr)
                X_val = sc.transform(X_val)
                pca   = PCA(n_components=200, random_state=42)
                X_tr  = pca.fit_transform(X_tr)
                X_val = pca.transform(X_val)
            sw  = compute_sample_weight('balanced', y_tr)
            clf = XGBClassifier(**XGB_PARAMS, num_class=n_classes)
            clf.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
            pred = clf.predict(X_val)

        all_true.extend(y_val)
        all_pred.extend(pred)

    cm = confusion_matrix(all_true, all_pred, labels=list(range(n_classes)))
    return cm, le.classes_


def plot_cm(ax, cm, labels, title, bacc):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')

    n = len(labels)
    short = [l[:16] for l in labels]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=5.5)
    ax.set_yticklabels(short, fontsize=5.5)
    ax.set_xlabel('Predito', fontsize=8)
    ax.set_ylabel('Verdadeiro', fontsize=8)
    ax.set_title(f'{title}\nBAcc = {bacc:.3f}', fontsize=9, fontweight='bold')

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1,
                                    fill=False, edgecolor='tomato',
                                    linewidth=0.5, alpha=0.7))
    return im


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out = Path('./plots_v3')

    print('Carregando parquets...')
    gf = load_filtered('./plots_v3/graph_features.parquet')
    stems  = gf['stem'].astype(str).tolist()
    y_full = np.array(gf['style'].tolist())

    X_rag_struct = gf[ALL_FEAT_COLS].to_numpy().astype(float)

    struct_lookup = gf.set_index('stem')[ALL_STRUCT_COLS]
    kf    = load_filtered('./plots_v3/knn_features.parquet')
    kf    = kf.set_index('stem').reindex(stems).reset_index()
    kf_aug = kf.set_index('stem').join(struct_lookup, how='left').reset_index()
    X_knn_struct = kf_aug[ALL_FEAT_COLS].to_numpy().astype(float)

    X_desc = np.load('./plots_v3/desc_features_filtered.npy')
    v_desc = np.load('./plots_v3/desc_valid_idx_filtered.npy').tolist()
    y_desc = y_full[v_desc]

    X_cnn = np.load('./plots_v3/cnn_resnet50_features_filtered.npy')
    v_cnn = np.load('./plots_v3/cnn_resnet50_valid_idx_filtered.npy').tolist()
    y_cnn = y_full[v_cnn]

    configs = [
        ('RAG+estrutural  (LDA, 23 feat)', X_rag_struct, y_full, 'lda', 0.188),
        ('kNN+estrutural  (LDA, 23 feat)', X_knn_struct, y_full, 'lda', 0.189),
        ('Descritores clássicos (XGB, 130 feat)', X_desc, y_desc, 'xgb', 0.268),
        ('CNN ResNet50  (XGB, 2048 feat)', X_cnn,        y_cnn,  'xgb', 0.366),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.ravel()

    for ax, (title, X, y, clf, bacc) in zip(axes, configs):
        print(f'  CV {title.split("(")[0].strip()}...')
        cm, labels = cv_confusion(X, y, clf_type=clf)
        im = plot_cm(ax, cm, labels, title, bacc)

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.55, pad=0.02)
    cbar.set_label('Recall (fração de predições corretas por classe)', fontsize=9)

    fig.suptitle('Matrizes de Confusão — 5-fold CV Estratificado\n'
                 'Normalização por linha  |  48 estilos  |  3.648 imagens',
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 0.92, 0.99])

    out_path = out / 'confusion_matrices_comparison.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'\nSalvo em {out_path}')


if __name__ == '__main__':
    main()