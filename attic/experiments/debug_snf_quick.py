import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(v, '16')
import sys
sys.path.insert(0, 'experiments')
sys.path.insert(0, '.')
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from src.pipeline.registry import load_blocks
import evaluate_snf_similarity_graph as snf

CACHE = 'outputs/snf_debug_cache.npz'
if os.path.exists(CACHE):
    print('usando cache existente')
    d = np.load(CACHE)
    train_posterior, test_posterior = d['train_posterior'], d['test_posterior']
    y_train, y_test = d['y_train'], d['y_test']
    base_train, base_test = d['base_train'], d['base_test']
    n_classes = int(d['n_classes'])
else:
    rng = np.random.default_rng(42)
    cohort = load_blocks('wikiart', snf.CORE_BLOCKS, allow_external=False)
    sample_idx = rng.choice(len(cohort.stems), 10000, replace=False)
    y = LabelEncoder().fit_transform(cohort.labels[sample_idx])
    n_classes = len(np.unique(y))
    train_idx, test_idx = train_test_split(np.arange(10000), test_size=0.2, random_state=42, stratify=y)
    y_train, y_test = y[train_idx], y[test_idx]

    multiscale_desc_only = cohort.select('Multiscale', lambda c: '_Descriptors_' in f'_{c}_')[sample_idx]
    base_blocks = [cohort.blocks[name][sample_idx] for name in snf.CORE_BLOCKS if name != 'Multiscale']
    base = np.hstack(base_blocks + [multiscale_desc_only]).astype(np.float32)
    base_train, base_test = base[train_idx], base[test_idx]

    blocks_train = {name: cohort.blocks[name][sample_idx][train_idx].astype(np.float32) for name in snf.CORE_BLOCKS}
    blocks_test = {name: cohort.blocks[name][sample_idx][test_idx].astype(np.float32) for name in snf.CORE_BLOCKS}
    s_mats, fused_train = snf.snf_fuse(blocks_train)
    onehot = np.zeros((len(train_idx), n_classes))
    onehot[np.arange(len(train_idx)), y_train] = 1.0
    train_posterior = snf.spread_labels(fused_train, onehot)
    test_posterior = snf.inductive_test_posterior(blocks_train, blocks_test, train_posterior)

    np.savez(CACHE, train_posterior=train_posterior, test_posterior=test_posterior,
             y_train=y_train, y_test=y_test, base_train=base_train, base_test=base_test,
             n_classes=n_classes)
    print('cache salvo em', CACHE)

prior = np.bincount(y_train, minlength=n_classes).astype(np.float64)
prior /= prior.sum()

for smoothing in (0.001, 0.01, 0.05, 0.1):
    train_feature = snf.log_lift(train_posterior, prior, smoothing=smoothing).astype(np.float32)
    test_feature = snf.log_lift(test_posterior, prior, smoothing=smoothing).astype(np.float32)
    print(f'--- smoothing={smoothing} --- min={train_feature.min():.2f} max={train_feature.max():.2f} std={train_feature.std():.2f}')

    from evaluate_snf_similarity_graph import make_lda, make_mlp
    for clf_name, make in (('lda', make_lda), ('mlp', make_mlp)):
        xtr = np.hstack([base_train, train_feature])
        xte = np.hstack([base_test, test_feature])
        model = make(n_classes, xtr.shape[1]) if clf_name == 'lda' else make()
        model.fit(xtr, y_train)
        pred = model.predict(xte)
        bacc = balanced_accuracy_score(y_test, pred)
        print(f'  base+snf_graph {clf_name}: BAcc={bacc*100:.2f}%')

# baseline sem grafo, pra comparar
from evaluate_snf_similarity_graph import make_lda, make_mlp
print('--- base sozinha ---')
for clf_name, make in (('lda', make_lda), ('mlp', make_mlp)):
    model = make(n_classes, base_train.shape[1]) if clf_name == 'lda' else make()
    model.fit(base_train, y_train)
    pred = model.predict(base_test)
    bacc = balanced_accuracy_score(y_test, pred)
    print(f'  base {clf_name}: BAcc={bacc*100:.2f}%')
