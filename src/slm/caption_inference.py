"""
Roda um modelo de captioning fine-tuned (VisionEncoderDecoderModel, ex.: o de
finetune_caption.py em outputs/caption_model) sobre um conjunto de imagens e
salva as legendas geradas em JSONL. Genérico: serve para qualquer dataset.

Entrada: um manifest CSV com colunas (image_id, file_name, split, path) — ex.:
o manifest do download_coco.py ou do download_artemis.py — ou um diretório de
imagens (--image_dir).
Saída:   cada linha {"image_id", "file_name", "split", "caption", "model"}.

Resumível: pula image_id já presente no JSONL de saída.
"""
import argparse
import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

DEFAULT_MODEL_DIR = ROOT / 'outputs' / 'caption_model'

EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _coerce_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def load_items(args):
    if args.manifest:
        with open(args.manifest, newline='') as f:
            rows = list(csv.DictReader(f))
        if args.split:
            rows = [r for r in rows if r.get('split') == args.split]
        return [
            {'image_id': _coerce_id(r['image_id']), 'file_name': r['file_name'],
             'split': r.get('split', ''), 'path': r['path']}
            for r in rows
        ]
    paths = sorted(p for p in Path(args.image_dir).rglob('*') if p.suffix.lower() in EXTS)
    return [
        {'image_id': p.stem, 'file_name': p.name, 'split': args.split or '', 'path': str(p)}
        for p in paths
    ]


def load_done(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['image_id'])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    ap = argparse.ArgumentParser(
        description='Inferência de captioning com modelo fine-tuned sobre um manifest/diretório (qualquer dataset).')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--manifest', type=Path,
                     help='manifest.csv com colunas image_id, file_name, split, path.')
    src.add_argument('--image_dir', type=Path, help='Diretório com imagens (alternativa ao manifest).')
    ap.add_argument('--model_dir', type=Path, default=DEFAULT_MODEL_DIR,
                    help=f'Diretório do modelo treinado (padrão: {DEFAULT_MODEL_DIR}).')
    ap.add_argument('--output', type=Path, required=True,
                    help='JSONL de saída das legendas geradas.')
    ap.add_argument('--split', help='Filtra o manifest por split/art_style.')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_length', type=int, default=256,
                    help='Comprimento máx. da legenda gerada (tokens).')
    ap.add_argument('--num_beams', type=int, default=1,
                    help='Beams na geração (1 = greedy, mais rápido).')
    ap.add_argument('--fp16', action='store_true', help='Inferência em fp16 (GPU).')
    ap.add_argument('--limit', type=int, default=0, help='Processa no máximo N imagens (0 = todas).')
    ap.add_argument('--allow_cpu', action='store_true',
                    help='Permite rodar em CPU. Por padrão, aborta se a GPU não estiver disponível.')
    args = ap.parse_args()

    if args.image_dir:
        args.manifest = None

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    cuda_ok = torch.cuda.is_available()
    log(f'torch {torch.__version__} | cuda.is_available()={cuda_ok} | n_gpus={torch.cuda.device_count()}')
    if not cuda_ok and not args.allow_cpu:
        raise SystemExit('CUDA indisponível. Use --allow_cpu para rodar em CPU mesmo assim.')
    device = 'cuda' if cuda_ok else 'cpu'

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_items(args)
    done = load_done(out_path)
    items = [it for it in items if it['image_id'] not in done]
    if args.limit:
        items = items[:args.limit]
    if not items:
        log('Nada a processar (tudo já feito ou nenhuma imagem encontrada).')
        return
    log(f'{len(items):,} imagens a processar  ({len(done):,} já feitas)')

    log(f'Carregando modelo de {args.model_dir} ...')
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if args.fp16 and cuda_ok:
        model = model.half()
    log(f'Modelo carregado em {device}.')

    t0 = time.time()
    processed = 0
    with open(out_path, 'a') as out_f:
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            images, kept = [], []
            for it in batch:
                try:
                    images.append(Image.open(it['path']).convert('RGB'))
                except (FileNotFoundError, OSError) as e:
                    log(f'  pulando {it["path"]}: {e}')
                    continue
                kept.append(it)
            if not kept:
                continue

            pixel_values = image_processor(images=images, return_tensors='pt').pixel_values.to(device)
            if args.fp16 and cuda_ok:
                pixel_values = pixel_values.half()

            with torch.no_grad():
                out_ids = model.generate(
                    pixel_values,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                )
            captions = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            for it, caption in zip(kept, captions):
                out_f.write(json.dumps({
                    'image_id': it['image_id'],
                    'file_name': it['file_name'],
                    'split': it['split'],
                    'caption': caption.strip(),
                    'model': str(args.model_dir),
                }) + '\n')
            out_f.flush()

            processed += len(kept)
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1e-6)
            eta = (len(items) - processed) / max(rate, 1e-6)
            log(f'  {processed:,}/{len(items):,}  {elapsed/60:.1f} min  '
                f'{rate:.1f} img/s  ETA {eta/60:.0f} min')

    log(f'Concluído: {out_path}')


if __name__ == '__main__':
    main()
