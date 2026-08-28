"""Etapa 04 — estudante (decoder-only, prefixo visual) + acompanhamento.

Treina do zero um decoder-only compacto (não-LLM) que recebe a imagem como
prefixo (padrão ClipCap: encoder de imagem congelado -> MLP -> k tokens de
prefixo) e gera a legenda autoregressivamente. A cada validação, além da loss,
computa ROUGE (ROUGE-L) contra as legendas do professor no split de validação
e salva a curva.

Início/fim e "caber em 512":
  - sequência = [PREFIX_LEN prefixo] + <bos> + texto + <eos>, pad até o batch;
  - alvo inclui prever <eos>; a loss MASCARA <pad> (senão o modelo emite pad);
  - alvos truncados a MAX_CAPTION_TOKENS -> distribuição de comprimento limitada,
    bem abaixo de MAX_SEQ_LEN=512.

Entrada:
    artifacts/image_index.csv        (etapa 01, para caminhos de imagem)
    artifacts/teacher_captions.csv   (etapa 02, alvos)
    artifacts/tokenizer/             (etapa 03)

Saída:
    artifacts/student_ckpt/          melhor checkpoint (por ROUGE-L de val)
    artifacts/train_log.csv          step, epoch, train_loss, val_loss, rougeL

Uso:
    python 04_student.py --epochs 15 --batch-size 256 --bf16
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from rouge_score import rouge_scorer
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BATCH_SIZE,
    BOS,
    D_FF,
    D_MODEL,
    DROPOUT,
    EOS,
    EPOCHS,
    IMAGE_INDEX_CSV,
    LABEL_SMOOTHING,
    LR,
    MAX_CAPTION_TOKENS,
    N_HEADS,
    N_LAYERS,
    OUT_DIR,
    PAD,
    PREFIX_LEN,
    SEED,
    STUDENT_CKPT_DIR,
    TEACHER_CAPTIONS_CSV,
    TOKENIZER_DIR,
    TRAIN_LOG_CSV,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    log,
)

CLIP_NAME = "openai/clip-vit-base-patch32"   # ViT-B/32 -> 512-d image feat
CLIP_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# CLIP frozen image features (cached to disk so cada época não re-encoda)
# --------------------------------------------------------------------------
def build_clip_cache(paths: list[str], cache_path: Path) -> np.ndarray:
    """Encoda cada imagem com CLIP (congelado) uma vez e salva um .npy."""
    if cache_path.is_file():
        feats = np.load(cache_path)
        if len(feats) == len(paths):
            log(f"Cache de features CLIP reaproveitado: {feats.shape}")
            return feats
    log(f"Encodando {len(paths)} imagens com {CLIP_NAME} (congelado)…")
    processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)
    clip = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
    feats = np.zeros((len(paths), CLIP_DIM), dtype=np.float32)
    batch = 256
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
            emb = clip.get_image_features(pixel_values=px)
            # transformers>=5 retorna BaseModelOutputWithPooling; o vetor
            # 512-d está em .pooler_output (versões antigas devolviam o tensor).
            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output
            feats[i:i + batch] = emb.float().cpu().numpy()
            if (i // batch) % 4 == 0:
                log(f"  CLIP {min(i + batch, len(paths))}/{len(paths)}")
    del clip
    torch.cuda.empty_cache()
    np.save(cache_path, feats)
    return feats


# --------------------------------------------------------------------------
# Dataset: cada item = (feature CLIP, input_ids, labels)
# --------------------------------------------------------------------------
class PrefixCaptionDataset(Dataset):
    def __init__(self, feats, captions, tokenizer, bos_id, eos_id, pad_id,
                 max_tok):
        self.feats = feats
        self.captions = captions
        self.tok = tokenizer
        self.bos, self.eos, self.pad = bos_id, eos_id, pad_id
        self.max_tok = max_tok

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        text = str(self.captions[idx]).replace(EOS, "").strip()
        ids = self.tok.encode(text).ids[: self.max_tok - 1]
        # sequência de texto: <bos> tokens <eos>
        seq = [self.bos] + ids + [self.eos]
        feat = torch.tensor(self.feats[idx], dtype=torch.float32)
        return feat, torch.tensor(seq, dtype=torch.long)


def collate(batch, pad_id):
    feats = torch.stack([b[0] for b in batch])
    seqs = [b[1] for b in batch]
    maxlen = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
    return feats, ids


# --------------------------------------------------------------------------
# Modelo: prefixo visual (ClipCap) + decoder-only causal
# --------------------------------------------------------------------------
class VisualPrefixDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout,
                 prefix_len, clip_dim, max_seq_len, pad_id):
        super().__init__()
        self.prefix_len = prefix_len
        self.pad_id = pad_id
        self.d_model = d_model

        # projeção do prefixo visual: clip_dim -> prefix_len * d_model
        self.prefix_mlp = nn.Sequential(
            nn.Linear(clip_dim, d_model * prefix_len // 2),
            nn.GELU(),
            nn.Linear(d_model * prefix_len // 2, d_model * prefix_len),
        )
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # decoder-only: stack de camadas de self-attention com máscara causal
        # (TransformerEncoderLayer + src_mask causal = bloco de decoder-only).
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, activation="gelu",
                norm_first=True)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight       # weight tying
        self.max_seq_len = max_seq_len
        self.emb_scale = math.sqrt(d_model)

        # Init GPT-style: sem isso, o weight tying (lm_head = emb) produz
        # logits com std enorme (~23) e a cross-entropy explode p/ centenas.
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[pad_id].zero_()

    def _prefix(self, feats):
        p = self.prefix_mlp(feats)
        return p.view(feats.size(0), self.prefix_len, self.d_model)

    def forward(self, feats, input_ids):
        """input_ids: [B, T] (texto). Retorna logits [B, T, vocab] alinhados a
        prever o próximo token de texto (o prefixo não é previsto)."""
        b, t = input_ids.shape
        prefix = self._prefix(feats)                       # [B, P, D]
        tok = self.token_emb(input_ids) * self.emb_scale   # [B, T, D]
        x = torch.cat([prefix, tok], dim=1)                # [B, P+T, D]
        seqlen = x.size(1)
        pos = torch.arange(seqlen, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        # máscara causal sobre P+T
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device),
                          diagonal=1).bool()
        for blk in self.blocks:
            x = blk(x, src_mask=mask)
        x = self.norm(x)
        logits = self.lm_head(x)                            # [B, P+T, vocab]
        return logits[:, self.prefix_len:, :]              # só posições de texto

    @torch.inference_mode()
    def generate(self, feats, bos_id, eos_id, max_len, no_repeat=3):
        self.eval()
        b = feats.size(0)
        prefix = self._prefix(feats)
        # seqs cresce em passo com TODAS as linhas (retangular): sequências já
        # finalizadas recebem <pad>, mas o corte real fica registrado em `end`.
        seqs = [[bos_id] for _ in range(b)]
        done = [False] * b
        # nunca-terminadas: comprimento final = bos + max_len tokens appendados
        end = [max_len + 1] * b
        for t in range(max_len):
            ids = torch.tensor(seqs, device=feats.device)
            tok = self.token_emb(ids) * self.emb_scale
            x = torch.cat([prefix, tok], dim=1)
            pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
            m = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device),
                           diagonal=1).bool()
            for blk in self.blocks:
                x = blk(x, src_mask=m)
            logits = self.lm_head(self.norm(x))[:, -1, :]
            # bloqueio de n-gramas repetidos (só nas sequências ainda ativas)
            for i in range(b):
                if not done[i] and no_repeat and len(seqs[i]) >= no_repeat:
                    prev = tuple(seqs[i][-(no_repeat - 1):])
                    for j in range(no_repeat - 1, len(seqs[i])):
                        if tuple(seqs[i][j - no_repeat + 1:j]) == prev:
                            logits[i, seqs[i][j]] -= 1e9
            nxt = logits.argmax(-1).tolist()
            for i in range(b):
                if done[i]:
                    seqs[i].append(self.pad_id)     # mantém retangular
                    continue
                seqs[i].append(nxt[i])
                if nxt[i] == eos_id:
                    done[i] = True
                    end[i] = len(seqs[i])           # inclui o <eos>
            if all(done):
                break
        # recorta cada sequência no seu <eos> (remove o pad de preenchimento)
        return [seqs[i][:end[i]] for i in range(b)]


def decode_clean(tok, ids, bos_id, eos_id):
    out = [i for i in ids if i not in (bos_id, eos_id)]
    return tok.decode(out).strip()


def evaluate(model, loader, tok, bos_id, eos_id, pad_id, scorer, max_tok):
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss, n = 0.0, 0
    rouges = []
    with torch.inference_mode():
        for feats, ids in loader:
            feats, ids = feats.to(DEVICE), ids.to(DEVICE)
            logits = model(feats, ids)
            # prever token t+1 a partir de t (dentro do texto)
            pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
            gold = ids[:, 1:].reshape(-1)
            loss = ce(pred, gold)
            total_loss += loss.item() * feats.size(0)
            n += feats.size(0)
            gen = model.generate(feats, bos_id, eos_id, max_tok)
            for g, ref_ids in zip(gen, ids.tolist()):
                hyp = decode_clean(tok, g, bos_id, eos_id)
                ref = decode_clean(tok, ref_ids, bos_id, eos_id)
                if ref:
                    rouges.append(scorer.score(ref, hyp)["rougeL"].fmeasure)
    return total_loss / max(1, n), float(np.mean(rouges)) if rouges else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1, help="épocas")
    parser.add_argument("--out", type=Path, default=STUDENT_CKPT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    bos_id, eos_id, pad_id = (tok.token_to_id(BOS), tok.token_to_id(EOS),
                              tok.token_to_id(PAD))
    vocab_size = tok.get_vocab_size()
    log(f"tokenizer: vocab={vocab_size} bos={bos_id} eos={eos_id} pad={pad_id}")

    caps = pd.read_csv(TEACHER_CAPTIONS_CSV)
    if args.limit:
        caps = caps.head(args.limit)
    # split usa a coluna 'split' herdada do image_index (train/val/test)
    train_df = caps[caps["split"] == "train"].reset_index(drop=True)
    val_df = caps[caps["split"] == "val"].reset_index(drop=True)
    if len(val_df) < 8:                       # fallback se val ficou pequeno
        val_df = caps.sample(frac=0.1, random_state=SEED).reset_index(drop=True)
        train_df = caps.drop(val_df.index).reset_index(drop=True)
    log(f"train={len(train_df)} val={len(val_df)}")

    # features CLIP (congelado), cacheadas em disco por split
    tr_feats = build_clip_cache(list(train_df["path"]),
                                OUT_DIR / "clip_feats_train.npy")
    va_feats = build_clip_cache(list(val_df["path"]),
                                OUT_DIR / "clip_feats_val.npy")

    tr_ds = PrefixCaptionDataset(tr_feats, list(train_df["caption"]), tok,
                                 bos_id, eos_id, pad_id, MAX_CAPTION_TOKENS)
    va_ds = PrefixCaptionDataset(va_feats, list(val_df["caption"]), tok,
                                 bos_id, eos_id, pad_id, MAX_CAPTION_TOKENS)
    coll = lambda b: collate(b, pad_id)  # noqa: E731
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=coll, num_workers=4, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=coll, num_workers=2)

    model = VisualPrefixDecoder(
        vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT, PREFIX_LEN,
        CLIP_DIM, 512, pad_id).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"parâmetros treináveis: {n_params / 1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(tr_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * WARMUP_RATIO)

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    ce = nn.CrossEntropyLoss(ignore_index=pad_id,
                             label_smoothing=LABEL_SMOOTHING)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    use_bf16 = args.bf16 and DEVICE == "cuda"

    log_rows = []
    best_rouge = -1.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for feats, ids in tr_loader:
            feats, ids = feats.to(DEVICE), ids.to(DEVICE)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.zero_grad()
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if use_bf16 else torch.enable_grad())
            with ctx:
                logits = model(feats, ids)
                pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
                gold = ids[:, 1:].reshape(-1)
                loss = ce(pred, gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
            step += 1
        train_loss = ep_loss / max(1, nb)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss, rougeL = evaluate(model, va_loader, tok, bos_id, eos_id,
                                        pad_id, scorer, MAX_CAPTION_TOKENS)
            log(f"epoch {epoch:3d} | lr {args.lr * lr_at(step):.2e} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"ROUGE-L {rougeL:.4f}")
            log_rows.append({"epoch": epoch, "step": step,
                             "train_loss": round(train_loss, 4),
                             "val_loss": round(val_loss, 4),
                             "rougeL": round(rougeL, 4)})
            pd.DataFrame(log_rows).to_csv(TRAIN_LOG_CSV, index=False)
            if rougeL > best_rouge:
                best_rouge = rougeL
                torch.save({"model": model.state_dict(),
                            "config": {"vocab_size": vocab_size,
                                       "d_model": D_MODEL, "n_layers": N_LAYERS,
                                       "prefix_len": PREFIX_LEN},
                            "epoch": epoch, "rougeL": rougeL},
                           args.out / "best.pt")
                # amostra qualitativa
                feats, ids = next(iter(va_loader))
                gen = model.generate(feats[:3].to(DEVICE), bos_id, eos_id,
                                     MAX_CAPTION_TOKENS)
                for k in range(min(3, len(gen))):
                    log(f"    amostra: {decode_clean(tok, gen[k], bos_id, eos_id)}")
        else:
            log(f"epoch {epoch:3d} | train_loss {train_loss:.4f}")

    log(f"Treino concluído. Melhor ROUGE-L={best_rouge:.4f} -> "
        f"{args.out / 'best.pt'}; curva -> {TRAIN_LOG_CSV}")


if __name__ == "__main__":
    main()
