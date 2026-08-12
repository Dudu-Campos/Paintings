"""
Cria split estratificado do ArtEmis por estilo.

Split oficial: 80% treino / 20% teste (estratificado por estilo).
Dentro do treino, 30% do total (FINETUNE_RATIO) foi usado para fine-tune
do caption model — os outros 50% do treino são destinados a CNN e modelos
profundos. O fine-tuning do caption model foi feito com FINETUNE_RATIO e
não é refeito ao mudar TRAIN_RATIO.

Saída:
  data/artemis/finetune_stems.txt — stems usados no fine-tune do caption
  data/artemis/train_stems.txt   — todos os stems de treino (80%)
  data/artemis/test_stems.txt    — stems de teste limpos (20%)
  data/artemis/finetune_pairs.jsonl — pares para o stage 2 (inalterado)
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT        = Path(__file__).parent.parent.parent
ARTEMIS_ZIP = ROOT / 'raw' / 'archive(11).zip'
OUT_DIR     = ROOT / 'data' / 'artemis'
TRAIN_RATIO    = 0.80
FINETUNE_RATIO = 0.30  # subconjunto do treino usado no fine-tune do caption
SEED           = 42


def log(msg):
    import time
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    import zipfile
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log('Carregando ArtEmis...')
    with zipfile.ZipFile(ARTEMIS_ZIP) as z:
        with z.open('artemis_dataset_release_v0.csv') as f:
            reader = csv.DictReader(line.decode() for line in f)
            rows = list(reader)
    log(f'  {len(rows):,} utterances')

    # agrupa por estilo → pinturas únicas
    style_paintings = defaultdict(set)
    painting_utterances = defaultdict(list)
    painting_style = {}
    for r in rows:
        style    = r['art_style']
        painting = r['painting']
        style_paintings[style].add(painting)
        painting_utterances[painting].append(r['utterance'])
        painting_style[painting] = style

    log(f'  {len(painting_style):,} pinturas únicas, {len(style_paintings)} estilos')

    # split 80/20 por estilo; dentro do treino, primeiros 30% = fine-tune
    finetune_stems, train_stems, test_stems = set(), set(), set()
    for style, paintings in sorted(style_paintings.items()):
        paintings = sorted(paintings)
        random.shuffle(paintings)
        n_train = max(1, int(len(paintings) * TRAIN_RATIO))
        n_ft    = max(1, int(len(paintings) * FINETUNE_RATIO))
        finetune_stems.update(paintings[:n_ft])
        train_stems.update(paintings[:n_train])
        test_stems.update(paintings[n_train:])
        log(f'  {style}: {len(paintings)} → train {n_train} (fine-tune {n_ft}) / test {len(paintings)-n_train}')

    (OUT_DIR / 'finetune_stems.txt').write_text('\n'.join(sorted(finetune_stems)))
    (OUT_DIR / 'train_stems.txt').write_text('\n'.join(sorted(train_stems)))
    (OUT_DIR / 'test_stems.txt').write_text('\n'.join(sorted(test_stems)))

    # finetune_pairs inalterado (modelo já treinado)
    finetune_pairs = []
    for stem in finetune_stems:
        utterances = painting_utterances[stem]
        random.shuffle(utterances)
        finetune_pairs.append({
            'stem':      stem,
            'utterance': utterances[0],
            'art_style': painting_style[stem],
        })
    with open(OUT_DIR / 'finetune_pairs.jsonl', 'w') as f:
        for p in finetune_pairs:
            f.write(json.dumps(p) + '\n')

    log(f'Salvo em {OUT_DIR}:')
    log(f'  train_stems.txt    : {len(train_stems):,} pinturas (80%)')
    log(f'  finetune_stems.txt : {len(finetune_stems):,} pinturas (30% — subconjunto do treino)')
    log(f'  test_stems.txt     : {len(test_stems):,} pinturas (20%)')


if __name__ == '__main__':
    main()
