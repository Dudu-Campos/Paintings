"""
Extrai features de sentimento das captions ArtEmis (train + test) para o WikiArt completo.

Lê:  outputs/affect_captions/artemis_train_captions.jsonl
     outputs/affect_captions/artemis_test_captions.jsonl
Lê:  data/artemis/wikiart_meta.csv  (para obter style/artist/split)
Salva: outputs/wikiart_features/sentiment_features.parquet
       colunas: stem, style, artist, split + 16 features de sentimento
"""
import json
import sys
import time
from pathlib import Path

import json as _json
import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).parent.parent
_NRC_PATH = ROOT / '.venv/lib/python3.12/site-packages/nrclex/data/nrc_en.json'
_NRC_LEXICON: dict = _json.loads(_NRC_PATH.read_text())

NRC_CATS = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
            'negative', 'positive', 'sadness', 'surprise', 'trust']

FEAT_NAMES = (
    ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
     'tb_polarity', 'tb_subjectivity']
    + [f'nrc_{c}' for c in NRC_CATS]
)


def extract_feats(text: str, vader: SentimentIntensityAnalyzer) -> list:
    vs = vader.polarity_scores(text)
    tb = TextBlob(text)
    words = text.lower().split()
    counts = {c: 0 for c in NRC_CATS}
    for w in words:
        for cat in _NRC_LEXICON.get(w, []):
            if cat in counts:
                counts[cat] += 1
    n = max(len(words), 1)
    return (
        [vs['neg'], vs['neu'], vs['pos'], vs['compound'],
         tb.sentiment.polarity, tb.sentiment.subjectivity]
        + [counts[c] / n for c in NRC_CATS]
    )


def main():
    meta = pd.read_csv(ROOT / 'data/artemis/wikiart_meta.csv')
    meta_idx = {r['stem']: r for _, r in meta.iterrows()}

    vader = SentimentIntensityAnalyzer()
    rows = []
    t0 = time.time()

    for jsonl in [
        ROOT / 'outputs/affect_captions/artemis_train_captions.jsonl',
        ROOT / 'outputs/affect_captions/artemis_test_captions.jsonl',
    ]:
        print(f'Processando {jsonl.name}...', flush=True)
        for i, line in enumerate(open(jsonl)):
            r = json.loads(line)
            stem = Path(r['file_name']).stem
            m = meta_idx.get(stem, {})
            feats = extract_feats(r['caption'], vader)
            row = {
                'stem':   stem,
                'style':  m.get('style',  'unknown') if isinstance(m, dict) else m['style'],
                'artist': m.get('artist', 'unknown') if isinstance(m, dict) else m['artist'],
                'split':  m.get('split',  'unknown') if isinstance(m, dict) else m['split'],
            }
            for name, val in zip(FEAT_NAMES, feats):
                row[name] = val
            rows.append(row)
            if (i + 1) % 5000 == 0:
                print(f'  {i+1:,}  {time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / 'outputs/wikiart_features/sentiment_features.parquet'
    df.to_parquet(out, index=False)
    print(f'Salvo: {out}  shape={df.shape}')
    print(f'Split counts:\n{df["split"].value_counts()}')


if __name__ == '__main__':
    main()
