"""
run_cnn_lgbm.py — CNN ResNet50 (2048→PCA256) + LightGBM, 5-fold CV.
Complementa os resultados de compare_all_lightgbm.py já calculados.
"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

sys.path.insert(0, '.')
from lda_graph_features import (
    GRAPH_FEATURE_COLS, STRUCT_FEATURE_COLS, SPATIAL_FEATURE_COLS, COLOR_FEATURE_COLS,
)

ALL_32 = GRAPH_FEATURE_COLS + STRUCT_FEATURE_COLS + SPATIAL_FEATURE_COLS + COLOR_FEATURE_COLS
BALANCED_MIN = 31

LGBM_P = dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
              class_weight='balanced', n_jobs=16, random_state=42, verbose=-1)

def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def impute(X):
    X = X.copy().astype(np.float32)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        if m.any(): X[m, j] = np.nanmedian(X[:, j])
    return X

def lgbm_score(X, y):
    X = impute(X)
    le = LabelEncoder(); ye = le.fit_transform(y)
    scores = []
    for tr, val in StratifiedKFold(5, shuffle=True, random_state=42).split(X, ye):
        clf = LGBMClassifier(**LGBM_P, num_class=len(le.classes_), objective='multiclass')
        clf.fit(X[tr], ye[tr])
        scores.append(balanced_accuracy_score(ye[val], clf.predict(X[val])))
    return float(np.mean(scores)), float(np.std(scores))

def lda_score(X, y):
    X = impute(X); n_s = len(np.unique(y))
    pipe = Pipeline([('sc', StandardScaler()),
                     ('lda', LinearDiscriminantAnalysis(
                         n_components=min(n_s-1, X.shape[1]),
                         priors=np.ones(n_s)/n_s))])
    s = cross_val_score(pipe, X, y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        scoring='balanced_accuracy', n_jobs=4)
    return float(s.mean()), float(s.std())

log('Carregando features...')
rag = pd.read_parquet('./plots_v3/full_rag_features.parquet')
rag = rag[rag['style'] != 'unknown'].copy()
rag['stem'] = rag['stem'].astype(str)
vc = rag['style'].value_counts()
rag = rag[rag['style'].isin(vc[vc >= BALANCED_MIN].index)].reset_index(drop=True)

X_cnn_raw = np.load('./plots_v3/cnn_full_features.npy')
cnn_stems  = np.load('./plots_v3/cnn_full_stems.npy', allow_pickle=True).astype(str)
cnn_df     = pd.DataFrame({'stem': cnn_stems, 'cnn_row': np.arange(len(cnn_stems))})
df = rag[['stem','style']].merge(cnn_df, on='stem', how='inner')
log(f'Dataset: {len(df):,} imagens, {df["style"].nunique()} estilos')

y = df['style'].to_numpy()
X_cnn = X_cnn_raw[df['cnn_row'].to_numpy()]

log('Aplicando PCA(256) nas features CNN...')
pca = PCA(n_components=256, random_state=42)
X_cnn_pca = pca.fit_transform(X_cnn.astype(np.float32))
var_explained = pca.explained_variance_ratio_.sum()
log(f'PCA(256): variância explicada = {var_explained:.3f}')

log(f'Calculando CNN ResNet50 PCA(256)...')
lm, ls = lda_score(X_cnn_pca, y)
xm, xs = lgbm_score(X_cnn_pca, y)
log(f'CNN ResNet50 PCA(256)         256  {lm:.4f}±{ls:.4f}  {xm:.4f}±{xs:.4f}')

# Também testar RAG+kNN+CNN combinado
knn = pd.read_parquet('./plots_v3/full_knn_features.parquet')
knn['stem'] = knn['stem'].astype(str)
df2 = df.merge(rag[['stem'] + ALL_32], on='stem', how='inner')
df2 = df2.merge(knn[['stem'] + ALL_32], on='stem', how='inner', suffixes=('','_knn'))
X_rag  = df2[ALL_32].to_numpy().astype(float)
X_knn  = df2[[f'{c}_knn' for c in ALL_32]].to_numpy().astype(float)
X_cnn2 = X_cnn_raw[df2['cnn_row'].to_numpy()]
X_cnn2_pca = pca.transform(X_cnn2.astype(np.float32))
X_all  = np.hstack([X_rag, X_knn, X_cnn2_pca])  # 64 + 256 = 320
y2 = df2['style'].to_numpy()

log(f'Calculando RAG+kNN+CNN(320)...')
lm2, ls2 = lda_score(X_all, y2)
xm2, xs2 = lgbm_score(X_all, y2)
log(f'RAG+kNN+CNN PCA(256)          320  {lm2:.4f}±{ls2:.4f}  {xm2:.4f}±{xs2:.4f}')

# Sumário final
print('\n=== RESULTADOS FINAIS (LightGBM, 100k imgs, 104 estilos) ===')
print(f'{"Método":<35} {"feat":>5}  {"LDA BAcc":>12}  {"LGBM BAcc":>12}')
print('─' * 68)
print(f'{"RAG global (32)":<35} {"32":>5}  {"0.1535±0.0080":>12}  {"0.1307±0.0049":>12}')
print(f'{"kNN global (32)":<35} {"32":>5}  {"0.1347±0.0064":>12}  {"0.1185±0.0046":>12}')
print(f'{"RAG+kNN global (64)":<35} {"64":>5}  {"0.1709±0.0067":>12}  {"0.1372±0.0053":>12}')
print(f'{"CNN ResNet50 PCA(256)":<35} {"256":>5}  {lm:.4f}±{ls:.4f}  {xm:.4f}±{xs:.4f}')
print(f'{"RAG+kNN+CNN PCA(256) (320)":<35} {"320":>5}  {lm2:.4f}±{ls2:.4f}  {xm2:.4f}±{xs2:.4f}')
print('─' * 68)

Path('./plots_v3/lightgbm_final_results.txt').write_text(
    f'RAG(32):          LDA={0.1535:.4f} LGBM={0.1307:.4f}\n'
    f'kNN(32):          LDA={0.1347:.4f} LGBM={0.1185:.4f}\n'
    f'RAG+kNN(64):      LDA={0.1709:.4f} LGBM={0.1372:.4f}\n'
    f'CNN PCA(256):     LDA={lm:.4f} LGBM={xm:.4f}\n'
    f'RAG+kNN+CNN(320): LDA={lm2:.4f} LGBM={xm2:.4f}\n'
    f'Dataset: 100k imgs, 104 estilos, sem cap\n'
)
log('Salvo: plots_v3/lightgbm_final_results.txt')