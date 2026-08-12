"""
Stage 2 fine-tune: adapta o caption_model (ViT+GPT-2 já treinado no COCO)
para descrições emocionais de pinturas usando pares do ArtEmis.

Entrada:
  data/artemis/finetune_pairs.jsonl  — {stem, utterance, art_style}
  raw/wikiart/<art_style>/<stem>.jpg — imagens do WikiArt Kaggle

Saída:
  outputs/caption_model_artemis/     — modelo fine-tunado

Split: 80% treino / 20% validação dentro dos 30% de fine-tune separados
pelo build_artemis_split.py. O conjunto de 70% restante (eval_stems.txt)
não é tocado aqui e serve para avaliação final limpa.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

ROOT             = Path(__file__).parent.parent.parent
PAIRS_JSONL      = ROOT / 'data' / 'artemis' / 'finetune_pairs.jsonl'
WIKIART_DIR      = ROOT / 'raw' / 'wikiart'
BASE_MODEL_DIR   = ROOT / 'outputs' / 'caption_model'
DEFAULT_OUT_DIR  = ROOT / 'outputs' / 'caption_model_artemis'
IGNORE_PAD_TOKEN = True


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_examples(pairs_path, wikiart_dir, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    pairs = [json.loads(l) for l in open(pairs_path)]

    examples = []
    n_missing = 0
    for p in pairs:
        style = p['art_style']
        img_path = wikiart_dir / style / f"{p['stem']}.jpg"
        if not img_path.exists():
            n_missing += 1
            continue
        examples.append({'image_path': str(img_path), 'caption': p['utterance']})

    log(f'  {len(examples):,} pares com imagem ({n_missing:,} sem imagem)')

    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_ratio))
    return examples[n_val:], examples[:n_val]


def make_tokenize_fn(tokenizer, max_target_length):
    def tokenize_fn(batch):
        labels = tokenizer(
            batch['caption'],
            padding='max_length',
            truncation=True,
            max_length=max_target_length,
        ).input_ids
        if IGNORE_PAD_TOKEN:
            pad_id = tokenizer.pad_token_id
            labels = [[(t if t != pad_id else -100) for t in seq] for seq in labels]
        return {'labels': labels}
    return tokenize_fn


def make_collator(image_processor):
    import torch
    from PIL import Image

    def collate(features):
        images = [Image.open(f['image_path']).convert('RGB') for f in features]
        pixel_values = image_processor(images=images, return_tensors='pt').pixel_values
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        return {'pixel_values': pixel_values, 'labels': labels}

    return collate


def make_compute_metrics(tokenizer):
    import evaluate
    import nltk
    for resource in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except (LookupError, OSError):
            nltk.download(resource, quiet=True)

    metric = evaluate.load('rouge')

    def postprocess(preds, labels):
        preds  = ['\n'.join(nltk.sent_tokenize(p.strip())) for p in preds]
        labels = ['\n'.join(nltk.sent_tokenize(l.strip())) for l in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds  = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dec_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        dec_preds, dec_labels = postprocess(dec_preds, dec_labels)
        result = metric.compute(predictions=dec_preds, references=dec_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        result['gen_len'] = float(np.mean([np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]))
        return result

    return compute_metrics


def main():
    ap = argparse.ArgumentParser(
        description='Stage 2: fine-tune do caption_model com pares ArtEmis (pinturas + descrições emocionais).')
    ap.add_argument('--pairs',         type=Path, default=PAIRS_JSONL)
    ap.add_argument('--wikiart_dir',   type=Path, default=WIKIART_DIR)
    ap.add_argument('--base_model',    type=Path, default=BASE_MODEL_DIR,
                    help='Modelo de partida (stage 1, treinado no COCO).')
    ap.add_argument('--output_dir',    type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--val_ratio',     type=float, default=0.2)
    ap.add_argument('--max_target_length', type=int, default=128)
    ap.add_argument('--epochs',        type=float, default=5.0)
    ap.add_argument('--train_batch',   type=int, default=16)
    ap.add_argument('--eval_batch',    type=int, default=16)
    ap.add_argument('--lr',            type=float, default=2e-5,
                    help='LR menor que stage 1 (5e-5) para domain adaptation.')
    ap.add_argument('--fp16',          action='store_true')
    ap.add_argument('--dataloader_workers', type=int, default=8)
    ap.add_argument('--seed',          type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
    )
    import datasets

    cuda_ok = torch.cuda.is_available()
    log(f'torch {torch.__version__} | cuda={cuda_ok} | gpus={torch.cuda.device_count()}')
    if cuda_ok:
        log(f'  GPU: {torch.cuda.get_device_name(0)}')

    log(f'Carregando pares de {args.pairs} ...')
    train_ex, val_ex = load_examples(args.pairs, args.wikiart_dir, args.val_ratio, args.seed)
    log(f'  treino={len(train_ex):,}  val={len(val_ex):,}')

    log(f'Carregando modelo base de {args.base_model} ...')
    model           = VisionEncoderDecoderModel.from_pretrained(str(args.base_model))
    image_processor = AutoImageProcessor.from_pretrained(str(args.base_model))
    tokenizer       = AutoTokenizer.from_pretrained(str(args.base_model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id           = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id           = tokenizer.pad_token_id

    train_ds = datasets.Dataset.from_list(train_ex)
    val_ds   = datasets.Dataset.from_list(val_ex)

    tokenize_fn = make_tokenize_fn(tokenizer, args.max_target_length)
    log('Tokenizando ...')
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['caption'])
    val_ds   = val_ds.map(tokenize_fn,   batched=True, remove_columns=['caption'])

    collator = make_collator(image_processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        predict_with_generate=True,
        generation_max_length=128,
        eval_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        report_to='none',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=image_processor,
        args=training_args,
        compute_metrics=make_compute_metrics(tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    log(f'Treinando {args.epochs} épocas, lr={args.lr} ...')
    trainer.train()

    log('Avaliação final ...')
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        log(f'  {k}: {v}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    with open(args.output_dir / 'eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f'Modelo salvo em {args.output_dir}')


if __name__ == '__main__':
    main()
