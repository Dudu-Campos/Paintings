"""
plot_confusion_analysis.py — Confusion matrix + per-class accuracy para
RAG+kNN(64) e CNN PCA(256), para diagnosticar desbalanceamento de classe.

Uso: python plot_confusion_analysis.py
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

sys.path.insert(0, '.')
from lda_graph_features import (
    GRAPH_FEATURE_COLS, STRUCT_FEATURE_COLS, SPATIAL_FEATURE_COLS, COLOR_FEATURE_COLS,
)

ALL_32 = GRAPH_FEATURE_COLS + STRUCT_FEATURE_COLS + SPATIAL_FEATURE_COLS + COLOR_FEATURE_COLS
OUT = Path('./plots_v3')
BALANCED_MIN = 31

# ── Carregar features ─────────────────────────────────────────────────────────
print('Carregando features...')
rag = pd.read_parquet('./plots_v3/full_rag_features.parquet')
knn = pd.read_parquet('./plots_v3/full_knn_features.parquet')
rag = rag[rag['style'] != 'unknown'].copy()
rag['stem'] = rag['stem'].astype(str)
knn['stem'] = knn['stem'].astype(str)
vc = rag['style'].value_counts()
rag = rag[rag['style'].isin(vc[vc >= BALANCED_MIN].index)].reset_index(drop=True)
df = rag.merge(knn[['stem'] + ALL_32], on='stem', how='inner', suffixes=('', '_knn'))

X_cnn_raw = np.load('./plots_v3/cnn_full_features.npy')
cnn_stems  = np.load('./plots_v3/cnn_full_stems.npy', allow_pickle=True).astype(str)
cnn_df     = pd.DataFrame({'stem': cnn_stems, 'cnn_row': np.arange(len(cnn_stems))})
df = df.merge(cnn_df, on='stem', how='inner').reset_index(drop=True)

y_str  = df['style'].to_numpy()
le     = LabelEncoder(); y = le.fit_transform(y_str)
X_graph = np.hstack([df[ALL_32].to_numpy(), df[[f'{c}_knn' for c in ALL_32]].to_numpy()]).astype(np.float32)
X_cnn   = X_cnn_raw[df['cnn_row'].to_numpy()].astype(np.float32)
# NaN impute
for X in [X_graph, X_cnn]:
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        if m.any(): X[m, j] = np.nanmedian(X[:, j])

print(f'Dataset: {len(df):,} imgs, {len(le.classes_)} estilos')
print(f'Contagens: min={vc[vc >= BALANCED_MIN].min()}  max={vc[vc >= BALANCED_MIN].max()}')

# ── Split único 80/20 estratificado ──────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, val = next(sss.split(X_graph, y))

# ── Ajustar modelos ───────────────────────────────────────────────────────────
n_s = len(le.classes_)

print('Treinando LDA (RAG+kNN)...')
lda_g = Pipeline([('sc', StandardScaler()),
                  ('lda', LinearDiscriminantAnalysis(
                      n_components=min(n_s-1, X_graph.shape[1]),
                      priors=np.ones(n_s)/n_s))])
lda_g.fit(X_graph[tr], y[tr])
pred_lda_g = lda_g.predict(X_graph[val])

print('Treinando LDA (CNN PCA256)...')
pca = PCA(n_components=256, random_state=42)
X_cnn_pca = pca.fit_transform(X_cnn)
lda_c = Pipeline([('sc', StandardScaler()),
                  ('lda', LinearDiscriminantAnalysis(
                      n_components=min(n_s-1, X_cnn_pca.shape[1]),
                      priors=np.ones(n_s)/n_s))])
lda_c.fit(X_cnn_pca[tr], y[tr])
pred_lda_c = lda_c.predict(X_cnn_pca[val])

bacc_g = balanced_accuracy_score(y[val], pred_lda_g)
bacc_c = balanced_accuracy_score(y[val], pred_lda_c)
print(f'BAcc  RAG+kNN LDA: {bacc_g:.4f}   CNN LDA: {bacc_c:.4f}')

# ── Per-class accuracy ────────────────────────────────────────────────────────
style_names  = le.classes_
n_val        = np.bincount(y[val], minlength=n_s)
correct_g    = np.zeros(n_s); correct_c = np.zeros(n_s)
for true, pg, pc in zip(y[val], pred_lda_g, pred_lda_c):
    correct_g[true] += (pg == true)
    correct_c[true] += (pc == true)
acc_g = np.where(n_val > 0, correct_g / np.maximum(n_val, 1), np.nan)
acc_c = np.where(n_val > 0, correct_c / np.maximum(n_val, 1), np.nan)

# Ordenar por tamanho de classe
n_train = np.bincount(y[tr], minlength=n_s)
order   = np.argsort(n_train)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, acc, title in [(axes[0], acc_g[order], f'RAG+kNN (64 feat)  BAcc={bacc_g:.3f}'),
                        (axes[1], acc_c[order], f'CNN PCA256         BAcc={bacc_c:.3f}')]:
    colors = ['#d73027' if a < 0.1 else '#fdae61' if a < 0.3 else '#4575b4' for a in acc]
    ax.barh(range(n_s), acc, color=colors)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([f'{style_names[i]} ({n_train[i]})' for i in order], fontsize=6)
    ax.set_xlabel('Recall (per-class accuracy)')
    ax.set_title(title)
    ax.axvline(1/n_s, color='gray', linestyle='--', alpha=0.5, label='chance')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7)

plt.tight_layout()
p = OUT / 'per_class_accuracy.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
print(f'Salvo: {p}')
plt.close()

# ── Confusion matrices (normalizadas por linha = recall) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
for ax, pred, title in [(axes[0], pred_lda_g, f'RAG+kNN LDA  BAcc={bacc_g:.3f}'),
                         (axes[1], pred_lda_c, f'CNN PCA256 LDA  BAcc={bacc_c:.3f}')]:
    cm = confusion_matrix(y[val], pred, labels=np.arange(n_s), normalize='true')
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Predito'); ax.set_ylabel('Real')
    if n_s <= 50:
        ax.set_xticks(range(n_s)); ax.set_xticklabels(style_names, rotation=90, fontsize=5)
        ax.set_yticks(range(n_s)); ax.set_yticklabels(style_names, fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.03)

plt.tight_layout()
p = OUT / 'confusion_matrices.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
print(f'Salvo: {p}')
plt.close()

# ── Estatísticas de desbalanceamento ─────────────────────────────────────────
print(f'\n=== Diagnóstico de desbalanceamento ===')
print(f'Classes com recall=0 (RAG+kNN): {(acc_g == 0).sum()}')
print(f'Classes com recall=0 (CNN):      {(acc_c == 0).sum()}')
print(f'Correlação recall vs n_train (RAG+kNN): {np.corrcoef(n_train, acc_g)[0,1]:.3f}')
print(f'Correlação recall vs n_train (CNN):      {np.corrcoef(n_train, acc_c)[0,1]:.3f}')
print(f'\nTop-10 classes com maior recall (RAG+kNN):')
for i in np.argsort(acc_g)[-10:][::-1]:
    print(f'  {style_names[i]:<35} recall={acc_g[i]:.2f}  n_train={n_train[i]}')
print(f'\nTop-10 classes com menor recall (RAG+kNN):')
for i in np.argsort(acc_g)[:10]:
    print(f'  {style_names[i]:<35} recall={acc_g[i]:.2f}  n_train={n_train[i]}')