"""
xgboost_ablation.py — Roda o ablation completo (curva, RAG, k-NN, combinado)
usando XGBoost em vez de LDA.

Reaproveita os parquets já salvos pelos scripts anteriores:
  - graph_features.parquet     (de lda_graph_features.py)   → RAG básico
  - knn_features.parquet       (de lda_knn_features.py)     → k-NN básico
  - combined_features.parquet  (de lda_combined_features.py) → RAG+kNN

Se algum parquet não existir, o cenário correspondente é pulado.

XGBoost lida naturalmente com features redundantes/correlacionadas e
interações não-lineares — diferente do LDA, que assume covariância
compartilhada e relações lineares. Isso ajuda a diferenciar se um
platô de acurácia é real (sem mais sinal nas features) ou um artefato
da limitação do classificador linear.

Uso:
    python xgboost_ablation.py \
        --graph_parquet    ./plots/graph_features.parquet \
        --knn_parquet      ./plots/knn_features.parquet \
        --combined_parquet ./plots/combined_features.parquet \
        --output ./plots \
        --min_count 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, train_test_split,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight


GRAPH_FEATURE_COLS = [
    'n_nodes', 'n_edges', 'density',
    'area_gini', 'area_entropy', 'area_max_frac', 'area_mean', 'area_std',
    'color_dist_mean', 'color_dist_std', 'color_dist_max',
    'spatial_dist_mean', 'spatial_dist_std',
    'degree_mean', 'degree_std',
]

TOPO_FEATURE_COLS = [
    'modularity', 'n_communities',
    'clustering_coef_mean', 'clustering_coef_std',
    'assortativity_degree',
    'avg_path_length_approx', 'diameter_approx',
    'spectral_gap',
]


def filter_min_count(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    df = df[df['style'] != 'unknown'].copy()
    counts = df['style'].value_counts()
    valid  = counts[counts >= min_count].index
    return df[df['style'].isin(valid)]


PARAM_GRID = {
    'n_estimators':     [20, 50, 100],
    'max_depth':        [1, 2, 3],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.6, 0.8],
    'colsample_bytree':  [0.5, 0.7],
    'min_child_weight': [3, 5, 10],
    'reg_alpha':         [0, 0.5, 1.0],
    'reg_lambda':        [1.0, 2.0, 5.0],
}


def run_xgb_with_search(X: np.ndarray, y: np.ndarray, label: str,
                         test_size: float = 0.2,
                         n_iter_grid: int = 30,
                         random_state: int = 42) -> dict:
    """
    Divide treino/teste, busca hiperparâmetros via RandomizedSearchCV (CV
    no treino), e reporta acurácia (simples e balanceada) + F1-macro,
    no treino e no teste hold-out.

    Usa sample_weight balanceado por classe — sem isso, com classes
    desbalanceadas (ex: Impressionism muito mais frequente que outros
    estilos), o classificador pode colapsar para sempre prever a classe
    majoritária e ainda parecer "razoável" em acurácia simples.

    test_size=0.2 com estratificação por estilo.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, stratify=y_enc,
        random_state=random_state)

    sw_train = compute_sample_weight('balanced', y_train)

    base_clf = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=random_state,
    )

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    from sklearn.model_selection import RandomizedSearchCV
    search = RandomizedSearchCV(
        base_clf, PARAM_GRID, n_iter=n_iter_grid,
        cv=cv_inner, scoring='balanced_accuracy',
        random_state=random_state, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train, sample_weight=sw_train)

    best_clf = search.best_estimator_
    pred_train = best_clf.predict(X_train)
    pred_test  = best_clf.predict(X_test)

    acc_train   = accuracy_score(y_train, pred_train)
    bacc_train  = balanced_accuracy_score(y_train, pred_train)
    f1_train    = f1_score(y_train, pred_train, average='macro')

    acc_test    = accuracy_score(y_test, pred_test)
    bacc_test   = balanced_accuracy_score(y_test, pred_test)
    f1_test     = f1_score(y_test, pred_test, average='macro')

    bacc_cv_train = search.best_score_

    print(f'[{label}] Melhores params: {search.best_params_}')
    print(f'[{label}] Treino — acc={acc_train:.3f}  bal_acc={bacc_train:.3f}  '
          f'f1_macro={f1_train:.3f}')
    print(f'[{label}] CV(treino, bal_acc): {bacc_cv_train:.3f}')
    print(f'[{label}] Teste  — acc={acc_test:.3f}  bal_acc={bacc_test:.3f}  '
          f'f1_macro={f1_test:.3f}')

    return {
        'best_params':   search.best_params_,
        'acc_train':     acc_train,
        'bacc_train':    bacc_train,
        'f1_train':      f1_train,
        'bacc_cv_train': bacc_cv_train,
        'acc_test':      acc_test,
        'bacc_test':     bacc_test,
        'f1_test':       f1_test,
        'model':         best_clf,
    }


def get_feature_importance_from_model(model: XGBClassifier,
                                       feature_names: list[str],
                                       top_n: int = 10) -> pd.DataFrame:
    imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    return imp.head(top_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_parquet',    default='')
    parser.add_argument('--knn_parquet',      default='')
    parser.add_argument('--combined_parquet', default='')
    parser.add_argument('--output',    required=True)
    parser.add_argument('--min_count', type=int, default=30)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_iter_grid', type=int, default=60,
                        help='Número de combinações de hiperparâmetros '
                             'testadas via RandomizedSearchCV')
    parser.add_argument('--acc_curve_lda', type=float, default=0.150,
                        help='Acurácia LDA da curva N(ε), para referência')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    importances = {}

    scenarios = []

    if args.graph_parquet and Path(args.graph_parquet).exists():
        df = filter_min_count(pd.read_parquet(args.graph_parquet), args.min_count)
        scenarios.append(('RAG básico', df, GRAPH_FEATURE_COLS))
        if all(c in df.columns for c in TOPO_FEATURE_COLS):
            scenarios.append(('RAG + topológicas', df,
                               GRAPH_FEATURE_COLS + TOPO_FEATURE_COLS))

    if args.knn_parquet and Path(args.knn_parquet).exists():
        df = filter_min_count(pd.read_parquet(args.knn_parquet), args.min_count)
        scenarios.append(('k-NN básico', df, GRAPH_FEATURE_COLS))

    if args.combined_parquet and Path(args.combined_parquet).exists():
        df = filter_min_count(pd.read_parquet(args.combined_parquet), args.min_count)
        combined_cols = [f'rag_{c}' for c in GRAPH_FEATURE_COLS] + \
                         [f'knn_{c}' for c in GRAPH_FEATURE_COLS]
        scenarios.append(('RAG + k-NN combinado', df, combined_cols))

    for label, df, cols in scenarios:
        print(f'\n[{label}] {len(df):,} imagens  {df["style"].nunique()} estilos  '
              f'{len(cols)} features')
        X = df[cols].values.astype(float)
        y = df['style'].values
        r = run_xgb_with_search(X, y, label,
                                 test_size=args.test_size,
                                 n_iter_grid=args.n_iter_grid)
        results[f'{label} — {len(cols)} features'] = r
        importances[label] = get_feature_importance_from_model(r['model'], cols)

    print('\n' + '='*108)
    print(f'{"Método":<38} {"Acc tr":>7} {"BAcc tr":>8} {"F1 tr":>7} '
          f'{"Acc te":>7} {"BAcc te":>8} {"F1 te":>7}')
    print('-'*108)
    print(f'{"Curva N(ε) (LDA, ref, sem balanceamento)":<38} {"—":>7} {"—":>8} '
          f'{"—":>7} {args.acc_curve_lda:>7.3f} {"—":>8} {"—":>7}')
    for name, r in results.items():
        print(f'{name:<38} {r["acc_train"]:>7.3f} {r["bacc_train"]:>8.3f} '
              f'{r["f1_train"]:>7.3f} {r["acc_test"]:>7.3f} '
              f'{r["bacc_test"]:>8.3f} {r["f1_test"]:>7.3f}')
    print('='*108)
    print('Acc = acurácia simples | BAcc = acurácia balanceada '
          '(pesa cada classe igualmente) | F1 = F1-macro')
    print('Se BAcc << Acc, o modelo pode estar se apoiando na classe '
          'majoritária para inflar a acurácia simples.')

    print('\n=== Melhores hiperparâmetros por cenário ===')
    for name, r in results.items():
        print(f'{name}: {r["best_params"]}')

    print('\n=== Top features por importância (XGBoost) ===')
    for scenario, imp_df in importances.items():
        print(f'\n{scenario}:')
        print(imp_df.to_string(index=False))

    summary_rows = [{
        'scenario':      name,
        'acc_train':     r['acc_train'],
        'bacc_train':    r['bacc_train'],
        'f1_train':      r['f1_train'],
        'bacc_cv_train': r['bacc_cv_train'],
        'acc_test':      r['acc_test'],
        'bacc_test':     r['bacc_test'],
        'f1_test':       r['f1_test'],
        'best_params':   str(r['best_params']),
    } for name, r in results.items()]
    pd.DataFrame(summary_rows).to_csv(
        out / 'xgboost_ablation_summary.csv', index=False)
    print(f'\nResumo salvo em: {out / "xgboost_ablation_summary.csv"}')


if __name__ == '__main__':
    main()
