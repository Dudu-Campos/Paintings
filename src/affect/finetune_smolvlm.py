"""
Passo 2: Fine-tune SmolVLM-256M nos pares (foto, descrição sentimental).

Entrada:  outputs/affect_captions/photo_captions.jsonl
          raw/coco_val2017/*.jpg
Saída:    outputs/affect_model/  (checkpoint final)

LoRA no vision encoder + language model head.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq

ROOT       = Path(__file__).parent.parent.parent
COCO_DIR   = ROOT / 'raw' / 'coco_val2017'
CAPTIONS   = ROOT / 'outputs' / 'affect_captions' / 'photo_captions.jsonl'
OUT_DIR    = ROOT / 'outputs' / 'affect_model'
MODEL_ID   = 'HuggingFaceTB/SmolVLM-256M-Instruct'
DEVICE     = 'cuda'

LR         = 2e-4
EPOCHS     = 3
BATCH      = 8
MAX_LEN    = 256
LORA_R     = 16
LORA_ALPHA = 32


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class CaptionDataset(Dataset):
    def __init__(self, records, img_dir):
        self.records = records
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(self.img_dir / f"{r['stem']}.jpg").convert('RGB')
        return img, r['caption']


def collate(batch, processor):
    images, captions = zip(*batch)
    messages = [[{
        'role': 'user',
        'content': [{'type': 'image'}, {'type': 'text', 'text':
            'Describe the emotional atmosphere and aesthetic feeling.'}],
    }, {
        'role': 'assistant',
        'content': [{'type': 'text', 'text': cap}],
    }] for cap in captions]

    texts = [
        processor.apply_chat_template(m, tokenize=False)
        for m in messages
    ]
    enc = processor(
        text=texts, images=list(images),
        padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors='pt',
    )
    # Labels: -100 nos tokens de input, loss só na resposta
    labels = enc['input_ids'].clone()
    for i, text in enumerate(texts):
        # Encontrar onde começa a resposta do assistant
        assistant_token = processor.tokenizer.encode('<|im_start|>assistant',
                                                      add_special_tokens=False)
        inp = enc['input_ids'][i].tolist()
        # Mascarar tudo antes da resposta
        for j in range(len(inp) - len(assistant_token)):
            if inp[j:j + len(assistant_token)] == assistant_token:
                labels[i, :j + len(assistant_token)] = -100
                break
        else:
            labels[i] = -100  # fallback: ignorar se não achou
    enc['labels'] = labels
    return enc


def main():
    records = []
    with open(CAPTIONS) as f:
        for line in f:
            records.append(json.loads(line))
    log(f'{len(records):,} pares carregados')

    log(f'Carregando {MODEL_ID}...')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16)

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        bias='none', task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(DEVICE)

    dataset = CaptionDataset(records, COCO_DIR)
    loader  = DataLoader(
        dataset, batch_size=BATCH, shuffle=True, num_workers=4,
        collate_fn=lambda b: collate(b, processor),
    )

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))

    log('Treinando...')
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss  = model(**batch).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total_loss += loss.item()
            if step % 50 == 0:
                log(f'  ep {ep} step {step}/{len(loader)}  loss={total_loss/step:.4f}  ({(time.time()-t0)/60:.1f} min)')

        log(f'Época {ep}  loss_médio={total_loss/len(loader):.4f}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    log(f'Modelo salvo: {OUT_DIR}')


if __name__ == '__main__':
    main()
