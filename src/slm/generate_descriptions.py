"""
Gera descrições de imagens do COCO com um SLM multimodal via vLLM em batch.

Entrada: manifest do download_coco.py (image_id, file_name, split, path) ou um
diretório de imagens (--image_dir).
Saída:   outputs/slm_descriptions/coco_descriptions.jsonl
         cada linha: {"image_id", "file_name", "split", "description", "model"}

Resumível: pula image_id já presente no JSONL de saída.
"""
import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

DEFAULT_MODEL = 'unsloth/Qwen2.5-VL-7B-Instruct'
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

PROMPT = (
    'Describe this image in a rich, detailed paragraph. Explore the feelings '
    'and emotions it evokes, the objects present, the lighting, the '
    'morphological aspects of the shapes and figures, and the intensity of '
    'the mood it conveys.'
)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _coerce_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def load_items(args):
    if args.manifest:
        with open(args.manifest) as f:
            rows = list(csv.DictReader(f))
        if args.split:
            rows = [r for r in rows if r['split'] == args.split]
        return [
            {'image_id': _coerce_id(r['image_id']), 'file_name': r['file_name'],
             'split': r['split'], 'path': r['path']}
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


async def run(args):
    from PIL import Image

    from vllm_service import VLLMMultimodalService

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

    log(f'Carregando {args.model} no vLLM ...')
    service = VLLMMultimodalService(
        model_path=args.model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
        quantization=args.quantization,
    )
    await service.init()
    log('Modelo carregado.')

    t0 = time.time()
    processed = 0
    with open(out_path, 'a') as out_f:
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            texts, images, kept = [], [], []
            for it in batch:
                try:
                    images.append(Image.open(it['path']).convert('RGB'))
                except (FileNotFoundError, OSError) as e:
                    log(f'  pulando {it["path"]}: {e}')
                    continue
                texts.append(PROMPT)
                kept.append(it)

            if not kept:
                continue

            outputs = await service.generate_batch(texts, images)
            for it, desc in zip(kept, outputs):
                out_f.write(json.dumps({
                    'image_id': it['image_id'],
                    'file_name': it['file_name'],
                    'split': it['split'],
                    'description': desc.strip(),
                    'model': args.model,
                }) + '\n')
            out_f.flush()

            processed += len(kept)
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1e-6)
            eta = (len(items) - processed) / max(rate, 1e-6)
            log(f'  {processed:,}/{len(items):,}  {elapsed/60:.1f} min  '
                f'{rate:.1f} img/s  ETA {eta/60:.0f} min')

    log(f'Concluído: {out_path}')


def main():
    ap = argparse.ArgumentParser(description='Gera descrições de imagens com SLM via vLLM.')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--manifest', type=Path, help='manifest.csv do download_coco.py')
    src.add_argument('--image_dir', type=Path, help='Diretório com imagens (alternativa ao manifest).')
    ap.add_argument('--split', help='Filtra o manifest por split (ex: val2017).')
    ap.add_argument('--model', default=DEFAULT_MODEL, help=f'Modelo HF (padrão: {DEFAULT_MODEL}).')
    ap.add_argument('--output', type=Path,
                    default=Path('outputs/slm_descriptions/coco_descriptions.jsonl'))
    ap.add_argument('--batch_size', type=int, default=8,
                    help='Imagens por batch / max_num_seqs do vLLM.')
    ap.add_argument('--max_tokens', type=int, default=256)
    ap.add_argument('--max_model_len', type=int, default=4096)
    ap.add_argument('--gpu_memory_utilization', type=float, default=0.90)
    ap.add_argument('--quantization', default=None,
                    help='Quantização do vLLM (ex: bitsandbytes para os repos -bnb-4bit da Unsloth).')
    ap.add_argument('--limit', type=int, default=0, help='Processa no máximo N imagens (0 = todas).')
    args = ap.parse_args()

    asyncio.run(run(args))


if __name__ == '__main__':
    main()
