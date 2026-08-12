"""
Extrai features de sentimento das legendas geradas pelo caption_model.

16 features por pintura:
  VADER    (4): neg, neu, pos, compound
  TextBlob (2): polarity, subjectivity
  NRC     (10): anger, anticipation, disgust, fear, joy,
                negative, positive, sadness, surprise, trust

Saída: outputs/features/caption_sentiment_features.npy  (N, 16)
       outputs/features/caption_sentiment_stems.npy     (N,)
       outputs/features/caption_sentiment_labels.npy    (N,) — artista
"""
import json
import time
from pathlib import Path

import json as _json
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# carrega léxico NRC manualmente (evita bug de path detection do nrclex)
_NRC_PATH = Path(__file__).parent.parent.parent / '.venv/lib/python3.12/site-packages/nrclex/data/nrc_en.json'
_NRC_LEXICON: dict = _json.loads(_NRC_PATH.read_text())

ROOT      = Path(__file__).parent.parent.parent
CAPTIONS  = ROOT / 'outputs' / 'affect_captions' / 'painting_captions.jsonl'
OUT_DIR   = ROOT / 'outputs' / 'features'

NRC_CATS = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
            'negative', 'positive', 'sadness', 'surprise', 'trust']

FEAT_NAMES = (
    ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
    + ['tb_polarity', 'tb_subjectivity']
    + [f'nrc_{c}' for c in NRC_CATS]
)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def extract_features(text: str, vader: SentimentIntensityAnalyzer) -> np.ndarray:
    # VADER
    vs = vader.polarity_scores(text)
    vader_feats = [vs['neg'], vs['neu'], vs['pos'], vs['compound']]

    # TextBlob
    tb = TextBlob(text)
    tb_feats = [tb.sentiment.polarity, tb.sentiment.subjectivity]

    # NRC — conta ocorrências de cada emoção, normaliza pelo nº de palavras
    words   = text.lower().split()
    counts  = {c: 0 for c in NRC_CATS}
    for w in words:
        for cat in _NRC_LEXICON.get(w, []):
            if cat in counts:
                counts[cat] += 1
    n_words   = max(len(words), 1)
    nrc_feats = [counts[c] / n_words for c in NRC_CATS]

    return np.array(vader_feats + tb_feats + nrc_feats, dtype=np.float32)


def main():
    log(f'Carregando legendas de {CAPTIONS}...')
    records = [json.loads(l) for l in open(CAPTIONS)]
    log(f'{len(records):,} legendas')

    vader = SentimentIntensityAnalyzer()

    feats, stems, labels = [], [], []
    t0 = time.time()
    for i, r in enumerate(records):
        stem   = r['image_id']
        artist = '_'.join(stem.split('_')[:-1])  # "Vincent_van_Gogh_89" → "Vincent_van_Gogh"
        f = extract_features(r['caption'], vader)
        feats.append(f)
        stems.append(stem)
        labels.append(artist)
        if (i + 1) % 1000 == 0:
            log(f'  {i+1:,}/{len(records):,}  {time.time()-t0:.1f}s')

    feats  = np.vstack(feats)
    stems  = np.array(stems)
    labels = np.array(labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / 'caption_sentiment_features.npy', feats)
    np.save(OUT_DIR / 'caption_sentiment_stems.npy',    stems)
    np.save(OUT_DIR / 'caption_sentiment_labels.npy',   labels)
    log(f'Salvo: shape={feats.shape}, {len(set(labels))} artistas')
    log(f'Features: {FEAT_NAMES}')


if __name__ == '__main__':
    main()
