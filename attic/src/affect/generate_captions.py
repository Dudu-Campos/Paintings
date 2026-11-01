"""
Passo 1: Gera descrições sentimentais de pinturas com Qwen2-VL-7B.

Entrada: diretório com pinturas (ArtGAN + Best Artworks, sem WikiArt).
Saída:   outputs/affect_captions/painting_captions.jsonl
         cada linha: {"stem": "...", "caption": "..."}
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

ROOT      = Path(__file__).parent.parent.parent
OUT_JSONL = ROOT / 'outputs' / 'affect_captions' / 'painting_captions.jsonl'
MODEL_ID  = 'Qwen/Qwen2-VL-7B-Instruct'
DEVICE    = 'cuda'
BATCH     = 4

PROMPT = (
    "Describe the emotional atmosphere and aesthetic feeling of this image "
    "in 2-3 sentences. Do not name specific objects or people. Focus on: "
    "how the colors make you feel, the overall mood and emotional tone, "
    "and the visual qualities that create this impression."
)
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dirs', nargs='+', required=True,
                    help='Diretórios com imagens de pinturas (ex: raw/artgan raw/best_artworks)')
    args = ap.parse_args()

    log(f'Carregando {MODEL_ID}...')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda',
    )
    model.eval()
    log('Modelo carregado.')

    paths = []
    for d in args.image_dirs:
        paths.extend(p for p in Path(d).rglob('*') if p.suffix.lower() in EXTS)
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f'Nenhuma imagem nos diretórios: {args.image_dirs}')
    log(f'{len(paths):,} pinturas')

    already = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                already.add(json.loads(line)['stem'])
        log(f'  {len(already):,} já processadas — continuando')

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = len(already)

    with open(OUT_JSONL, 'a') as out_f:
        for i in range(0, len(paths), BATCH):
            batch_paths = [p for p in paths[i:i + BATCH] if p.stem not in already]
            if not batch_paths:
                continue

            messages_batch = []
            images_batch   = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                images_batch.append(img)
                messages_batch.append([{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': img},
                        {'type': 'text',  'text': PROMPT},
                    ],
                }])

            texts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_batch
            ]
            inputs = processor(
                text=texts,
                images=images_batch,
                padding=True,
                return_tensors='pt',
            ).to(DEVICE)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,
                )

            # Decodifica só os tokens novos
            input_len = inputs['input_ids'].shape[1]
            for p, ids in zip(batch_paths, output_ids):
                caption = processor.decode(
                    ids[input_len:], skip_special_tokens=True
                ).strip()
                out_f.write(json.dumps({'stem': p.stem, 'caption': caption}) + '\n')
            out_f.flush()

            done += len(batch_paths)
            if done % 200 == 0 or done == len(paths):
                elapsed = time.time() - t0
                rate    = done / max(elapsed, 1)
                eta     = (len(paths) - done) / max(rate, 1e-6)
                log(f'  {done:,}/{len(paths):,}  {elapsed/60:.1f} min  ETA {eta/60:.0f} min')

    log(f'Concluído: {OUT_JSONL}')


if __name__ == '__main__':
    main()
