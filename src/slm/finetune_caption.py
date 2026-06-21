"""
Fine-tune e validação de um modelo de image captioning (ViT encoder + GPT-2
decoder, via VisionEncoderDecoderModel), usando as descrições geradas pelo SLM
em outputs/slm_descriptions/coco_descriptions.jsonl como captions-alvo.

Baseado em:
https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/

Entradas:
  - Descrições: JSONL com {image_id, file_name, split, description, model}.
  - Imagens COCO: <coco_dir>/images/<split>/<file_name> (via download_coco.py).

Junta cada descrição ao arquivo de imagem por (split, file_name), descartando
registros cuja imagem não exista no disco. Treina no split de treino, valida no
de validação (ROUGE) e salva modelo + métricas em --output_dir.

Observação: as descrições do SLM são parágrafos longos (~210 palavras), bem mais
longas que captions COCO. --max_target_length trunca os alvos (padrão 128).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent

DEFAULT_DESCRIPTIONS = ROOT / 'outputs' / 'slm_descriptions' / 'coco_descriptions.jsonl'
DEFAULT_COCO_DIR = ROOT / 'data' / 'coco'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'caption_model'

ENCODER_MODEL = 'google/vit-base-patch16-224-in21k'
DECODER_MODEL = 'gpt2'

IGNORE_PAD_TOKEN_FOR_LOSS = True


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_examples(descriptions_path, coco_dir, split, max_examples=0):
    """Lê o JSONL e junta cada descrição ao caminho da imagem no disco.

    Mantém apenas registros do split pedido cuja imagem exista.
    """
    images_dir = Path(coco_dir) / 'images' / split
    examples = []
    n_total = 0
    n_missing = 0
    with open(descriptions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get('split') != split:
                continue
            n_total += 1
            caption = (rec.get('description') or '').strip()
            if not caption:
                continue
            img_path = images_dir / rec['file_name']
            if not img_path.exists():
                n_missing += 1
                continue
            examples.append({'image_path': str(img_path), 'caption': caption})
            if max_examples and len(examples) >= max_examples:
                break
    log(f'  split={split}: {len(examples):,} usáveis '
        f'({n_total:,} no JSONL, {n_missing:,} sem imagem no disco)')
    return examples


def build_dataset(examples):
    import datasets
    return datasets.Dataset.from_list(examples)


def make_preprocess_fn(tokenizer, image_processor, max_target_length):
    from PIL import Image

    def tokenization_fn(captions):
        return tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=max_target_length,
        ).input_ids

    def feature_extraction_fn(image_paths):
        images = [Image.open(p).convert('RGB') for p in image_paths]
        return image_processor(images=images, return_tensors='np').pixel_values

    def preprocess_fn(examples):
        labels = tokenization_fn(examples['caption'])
        if IGNORE_PAD_TOKEN_FOR_LOSS:
            pad_id = tokenizer.pad_token_id
            labels = [
                [(t if t != pad_id else -100) for t in seq]
                for seq in labels
            ]
        return {
            'labels': labels,
            'pixel_values': feature_extraction_fn(examples['image_path']),
        }

    return preprocess_fn


def make_compute_metrics(tokenizer):
    import evaluate
    import nltk

    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, OSError):
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except (LookupError, OSError):
        nltk.download('punkt_tab', quiet=True)

    metric = evaluate.load('rouge')

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        preds = ['\n'.join(nltk.sent_tokenize(p)) for p in preds]
        labels = ['\n'.join(nltk.sent_tokenize(l)) for l in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result['gen_len'] = float(np.mean(prediction_lens))
        return result

    return compute_metrics


def main():
    ap = argparse.ArgumentParser(
        description='Fine-tune ViT+GPT2 (image captioning) com descrições do SLM.')
    ap.add_argument('--descriptions', type=Path, default=DEFAULT_DESCRIPTIONS,
                    help='JSONL de descrições (image_id, file_name, split, description).')
    ap.add_argument('--coco_dir', type=Path, default=DEFAULT_COCO_DIR,
                    help='Dir do COCO com images/<split>/ (do download_coco.py).')
    ap.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT,
                    help='Dir de saída do modelo e métricas.')
    ap.add_argument('--train_split', default='train2017')
    ap.add_argument('--val_split', default='val2017')
    ap.add_argument('--encoder_model', default=ENCODER_MODEL)
    ap.add_argument('--decoder_model', default=DECODER_MODEL)
    ap.add_argument('--max_target_length', type=int, default=128,
                    help='Tamanho máx. (tokens) do caption-alvo; trunca acima disso.')
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--train_batch_size', type=int, default=4)
    ap.add_argument('--eval_batch_size', type=int, default=4)
    ap.add_argument('--learning_rate', type=float, default=5e-5)
    ap.add_argument('--generation_max_length', type=int, default=64,
                    help='max_length usado na geração durante a avaliação.')
    ap.add_argument('--max_train', type=int, default=0,
                    help='Limita o nº de exemplos de treino (0 = todos).')
    ap.add_argument('--max_val', type=int, default=0,
                    help='Limita o nº de exemplos de validação (0 = todos).')
    ap.add_argument('--map_workers', type=int, default=1,
                    help='Processos para o pré-processamento (datasets.map).')
    ap.add_argument('--fp16', action='store_true', help='Treino em fp16 (requer GPU).')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
        default_data_collator,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'Device: {device}')

    log(f'Carregando exemplos de {args.descriptions} ...')
    train_examples = load_examples(args.descriptions, args.coco_dir,
                                   args.train_split, args.max_train)
    val_examples = load_examples(args.descriptions, args.coco_dir,
                                 args.val_split, args.max_val)
    if not train_examples:
        raise SystemExit(
            f'Nenhum exemplo de treino encontrado. Verifique se as imagens do split '
            f'"{args.train_split}" estão em {args.coco_dir}/images/{args.train_split}/ '
            f'(use src/slm/download_coco.py).')
    if not val_examples:
        raise SystemExit(
            f'Nenhum exemplo de validação encontrado para o split "{args.val_split}".')

    log(f'Inicializando modelo: encoder={args.encoder_model} decoder={args.decoder_model}')
    decoder_config = AutoConfig.from_pretrained(args.decoder_model)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder_model, args.decoder_model, decoder_config=decoder_config)
    image_processor = AutoImageProcessor.from_pretrained(args.encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = build_dataset(train_examples)
    val_ds = build_dataset(val_examples)

    preprocess_fn = make_preprocess_fn(tokenizer, image_processor, args.max_target_length)
    log('Pré-processando (tokenização + extração de features) ...')
    train_ds = train_ds.map(preprocess_fn, batched=True,
                            remove_columns=train_ds.column_names,
                            num_proc=args.map_workers or 1)
    val_ds = val_ds.map(preprocess_fn, batched=True,
                        remove_columns=val_ds.column_names,
                        num_proc=args.map_workers or 1)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        eval_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        output_dir=str(args.output_dir),
        report_to='none',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=image_processor,
        args=training_args,
        compute_metrics=make_compute_metrics(tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )

    log(f'Treinando: {len(train_ds):,} treino / {len(val_ds):,} validação, '
        f'{args.epochs} épocas ...')
    trainer.train()

    log('Avaliação final no split de validação ...')
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        log(f'  {k}: {v}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    with open(args.output_dir / 'eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f'Modelo e métricas salvos em {args.output_dir}')


if __name__ == '__main__':
    main()
