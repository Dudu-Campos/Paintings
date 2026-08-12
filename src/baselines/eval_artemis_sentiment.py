"""
Avalia features de sentimento extraídas das captions do modelo ArtEmis
(fine-tuned no stage 2) usando LDA e XGBoost com cross-validation.

Compara com baseline naive COCO (LDA=0.058, XGB=0.055, chance=0.037).
Label: art_style (27 classes).
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json as _json

ROOT       = Path(__file__).parent.parent.parent
CAPTIONS   = ROOT / 'outputs' / 'affect_captions' / 'artemis_test_captions.jsonl'
NRC_PATH   = ROOT / '.venv/lib/python3.12/site-packages/nrclex/data/nrc_en.json'

NRC_CATS   = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
              'negative', 'positive', 'sadness', 'surprise', 'trust']
FEAT_NAMES = (['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
              + ['tb_polarity', 'tb_subjectivity']
              + [f'nrc_{c}' for c in NRC_CATS])

_NRC_LEXICON = _json.loads(NRC_PATH.read_text())


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def extract_features(text, vader):
    vs = vader.polarity_scores(text)
    vader_feats = [vs['neg'], vs['neu'], vs['pos'], vs['compound']]
    tb = TextBlob(text)
    tb_feats = [tb.sentiment.polarity, tb.sentiment.subjectivity]
    words  = text.lower().split()
    counts = {c: 0 for c in NRC_CATS}
    for w in words:
        for cat in _NRC_LEXICON.get(w, []):
            if cat in counts:
                counts[cat] += 1
    n = max(len(words), 1)
    nrc_feats = [counts[c] / n for c in NRC_CATS]
    return np.array(vader_feats + tb_feats + nrc_feats, dtype=np.float32)


def main():
    log(f'Carregando captions de {CAPTIONS}...')
    records = [json.loads(l) for l in open(CAPTIONS)]
    log(f'  {len(records):,} captions')

    vader = SentimentIntensityAnalyzer()
    feats, labels = [], []
    for r in records:
        feats.append(extract_features(r['caption'], vader))
        labels.append(r['split'])   # art_style = nome da pasta

    X = np.vstack(feats)
    y = LabelEncoder().fit_transform(labels)
    styles, counts = np.unique(labels, return_counts=True)
    log(f'  {len(styles)} estilos, {X.shape} features')
    log(f'  Chance: {1/len(styles):.4f}')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log('LDA 5-fold...')
    lda_scores = cross_val_score(
        LinearDiscriminantAnalysis(), X, y, cv=cv,
        scoring='balanced_accuracy', n_jobs=-1)
    log(f'  LDA BAcc = {lda_scores.mean():.4f} ± {lda_scores.std():.4f}')

    log('XGBoost 5-fold...')
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.model_selection import cross_validate

    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                        use_label_encoder=False, eval_metric='mlogloss',
                        n_jobs=-1, random_state=42, verbosity=0)

    def xgb_fit_score(estimator, X_tr, y_tr, X_te, y_te):
        sw = compute_sample_weight('balanced', y_tr)
        estimator.fit(X_tr, y_tr, sample_weight=sw)
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_te, estimator.predict(X_te))

    xgb_scores = []
    for tr_idx, te_idx in cv.split(X, y):
        xgb_clone = XGBClassifier(n_estimators=100, max_depth=3,
                                   use_label_encoder=False,
                                   eval_metric='mlogloss', n_jobs=-1,
                                   random_state=42, verbosity=0)
        sw = compute_sample_weight('balanced', y[tr_idx])
        xgb_clone.fit(X[tr_idx], y[tr_idx], sample_weight=sw)
        from sklearn.metrics import balanced_accuracy_score
        xgb_scores.append(balanced_accuracy_score(y[te_idx], xgb_clone.predict(X[te_idx])))

    xgb_scores = np.array(xgb_scores)
    log(f'  XGB BAcc = {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}')

    log('--- Resumo ---')
    log(f'  Chance:         {1/len(styles):.4f}')
    log(f'  LDA (ArtEmis):  {lda_scores.mean():.4f} ± {lda_scores.std():.4f}')
    log(f'  XGB (ArtEmis):  {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}')
    log(f'  LDA (COCO naive, 51 artistas): 0.0584')
    log(f'  XGB (COCO naive, 51 artistas): 0.0551')


if __name__ == '__main__':
    main()
