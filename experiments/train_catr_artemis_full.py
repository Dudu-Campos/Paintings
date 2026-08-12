"""Full clean CATR -> ArtEmis adaptation with grouped multi-caption training.

Each WikiArt image is encoded once per batch, while every eligible human
ArtEmis utterance for that painting supervises the language decoder.  The
visual CATR stack remains frozen.  Short, independent captions retain explicit
SEP supervision.  CATR pseudo-caption replay decreases across epochs to reduce
catastrophic forgetting without reintroducing Qwen supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAPER = Path(__file__).resolve().parent
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

import audit_catr_artemis_pilots as audit  # noqa: E402
import pilot_finetune_catr_artemis_multicaption as multicap  # noqa: E402
import pilot_finetune_catr_artemis_rich as common  # noqa: E402


ROOT = common.ROOT
DEFAULT_OUTPUT = ROOT / "outputs/catr_artemis_full_clean"


def eligible_human_captions(rows: list[dict], style: str) -> list[dict]:
    unique = {}
    for row in rows:
        text = common.clean_sentence(row["utterance"])
        count = len(common.words(text))
        normalized = common.normalize_text(text)
        if not normalized or count < 5 or count > 50:
            continue
        if common.style_is_named(text, style):
            continue
        unique.setdefault(
            normalized,
            {"text": text, "emotion": row["emotion"]},
        )
    return list(unique.values())


def make_paintings(stems, grouped, catr) -> list[dict]:
    result = []
    for stem in sorted(stems):
        rows = grouped.get(stem)
        if not rows:
            continue
        style = rows[0]["art_style"]
        objective = catr.get((style, stem))
        humans = eligible_human_captions(rows, style)
        if objective is None or not humans or not Path(objective["path"]).is_file():
            continue
        result.append(
            {
                "stem": stem,
                "style": style,
                "path": objective["path"],
                "catr_caption": common.clean_sentence(objective["caption"]),
                "human_captions": humans,
            }
        )
    return result


def stratified_train_validation(rows: list[dict], val_ratio: float, seed: int):
    groups = defaultdict(list)
    for row in rows:
        groups[row["style"]].append(row)
    train, validation = [], []
    for style, items in sorted(groups.items()):
        rng = random.Random(f"{seed}:{style}")
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio))
        validation.extend(items[:n_val])
        train.extend(items[n_val:])
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    return train, validation


def balanced_limit(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(rows):
        return rows
    return common.balanced_sample(rows, limit, seed)


class PaintingCaptionDataset(Dataset):
    def __init__(
        self,
        paintings: list[dict],
        tokenizer,
        max_tokens: int,
        mode: str,
        replay_repeats: int = 0,
    ):
        if mode not in {"train", "human", "objective"}:
            raise ValueError(f"Unknown dataset mode: {mode}")
        self.paintings = paintings
        self.mode = mode
        self.replay_repeats = replay_repeats
        self.max_tokens = max_tokens
        self.human_tokens = []
        self.objective_tokens = []
        for painting in paintings:
            human_texts = [item["text"] for item in painting["human_captions"]]
            human = tokenizer(
                human_texts,
                max_length=max_tokens,
                padding="max_length",
                truncation=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            objective = tokenizer(
                painting["catr_caption"],
                max_length=max_tokens,
                padding="max_length",
                truncation=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            self.human_tokens.append(torch.tensor(human, dtype=torch.long))
            self.objective_tokens.append(torch.tensor(objective, dtype=torch.long))

    def __len__(self):
        return len(self.paintings)

    def __getitem__(self, index):
        painting = self.paintings[index]
        with Image.open(painting["path"]) as image:
            pixels = common.val_transform(image.convert("RGB"))
        if self.mode == "objective":
            captions = self.objective_tokens[index].unsqueeze(0)
        elif self.mode == "human":
            captions = self.human_tokens[index]
        else:
            captions = self.human_tokens[index]
            if self.replay_repeats:
                replay = self.objective_tokens[index].unsqueeze(0).repeat(
                    self.replay_repeats, 1
                )
                captions = torch.cat([captions, replay], dim=0)
        return pixels, captions, painting


def grouped_collate(batch):
    images, caption_groups, paintings = zip(*batch)
    samples = common.nested_tensor_from_tensor_list(list(images))
    owners = []
    for image_index, captions in enumerate(caption_groups):
        owners.extend([image_index] * len(captions))
    return (
        samples,
        torch.cat(caption_groups, dim=0),
        torch.tensor(owners, dtype=torch.long),
        list(paintings),
    )


def grouped_loss(model, samples, tokens, owners, criterion):
    features, positions = model.backbone(samples)
    source, source_mask = features[-1].decompose()
    projected = model.input_proj(source)
    source_sequence = projected.flatten(2).permute(2, 0, 1)
    position_sequence = positions[-1].flatten(2).permute(2, 0, 1)
    source_mask = source_mask.flatten(1)
    memory = model.transformer.encoder(
        source_sequence,
        src_key_padding_mask=source_mask,
        pos=position_sequence,
    )

    memory = memory.index_select(1, owners)
    position_sequence = position_sequence.index_select(1, owners)
    source_mask = source_mask.index_select(0, owners)
    decoder_input = tokens[:, :-1]
    target_mask = decoder_input.eq(0)
    embedded = model.transformer.embeddings(decoder_input).permute(1, 0, 2)
    length = decoder_input.shape[1]
    query_position = (
        model.transformer.embeddings.position_embeddings.weight[:length]
        .unsqueeze(1)
        .repeat(1, decoder_input.shape[0], 1)
    )
    hidden = model.transformer.decoder(
        embedded,
        memory,
        memory_key_padding_mask=source_mask,
        tgt_key_padding_mask=target_mask,
        pos=position_sequence,
        query_pos=query_position,
        tgt_mask=common.generate_square_subsequent_mask(length).to(tokens.device),
    )
    logits = model.mlp(hidden.permute(1, 0, 2))
    return criterion(logits.permute(0, 2, 1), tokens[:, 1:])


def make_loader(dataset, batch_size, workers, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=True,
        collate_fn=grouped_collate,
        persistent_workers=False,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_captions = 0
    for samples, tokens, owners, _ in loader:
        samples = samples.to(device)
        tokens = tokens.to(device, non_blocking=True)
        owners = owners.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = grouped_loss(model, samples, tokens, owners, criterion)
        count = len(tokens)
        total_loss += float(loss) * count
        total_captions += count
    return total_loss / max(1, total_captions)


def atomic_torch_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_checkpoint(path, model, optimizer, scheduler, epoch, history, args):
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "arguments": vars(args),
        },
        path,
    )


def parse_replay(value: str, epochs: int) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if len(result) != epochs or any(item < 0 for item in result):
        raise ValueError("--replay-repeats needs one non-negative integer per epoch")
    return result


def train_full(model, train_rows, val_rows, tokenizer, args, device):
    replay = parse_replay(args.replay_repeats, args.epochs)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, round(total_steps * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return max(0.05, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    start_epoch = 1
    best_score = math.inf
    last_path = args.output_dir / "last.pth"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state.get("history", [])
        start_epoch = int(state["epoch"]) + 1
        if history:
            best_score = min(item["selection_score"] for item in history)
        common.log(f"resuming at epoch {start_epoch}")

    human_val = PaintingCaptionDataset(
        val_rows, tokenizer, args.max_tokens, mode="human"
    )
    objective_val = PaintingCaptionDataset(
        val_rows, tokenizer, args.max_tokens, mode="objective"
    )
    human_loader = make_loader(
        human_val, args.batch_size, args.workers, False, args.seed
    )
    objective_loader = make_loader(
        objective_val, args.batch_size, args.workers, False, args.seed
    )
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        training = PaintingCaptionDataset(
            train_rows,
            tokenizer,
            args.max_tokens,
            mode="train",
            replay_repeats=replay[epoch - 1],
        )
        loader = make_loader(
            training, args.batch_size, args.workers, True, args.seed + epoch
        )
        model.train()
        common.keep_frozen_modules_in_eval(model)
        running_loss = 0.0
        running_captions = 0
        epoch_started = time.perf_counter()
        for step, (samples, tokens, owners, _) in enumerate(loader, 1):
            samples = samples.to(device)
            tokens = tokens.to(device, non_blocking=True)
            owners = owners.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = grouped_loss(model, samples, tokens, owners, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 0.1)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP may skip an optimizer step after detecting an overflow.
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            count = len(tokens)
            running_loss += float(loss.detach()) * count
            running_captions += count
            if step == 1 or step % args.log_every == 0 or step == len(loader):
                rate = running_captions / max(1e-6, time.perf_counter() - epoch_started)
                common.log(
                    f"epoch={epoch}/{args.epochs} replay={replay[epoch-1]} "
                    f"step={step}/{len(loader)} captions={running_captions:,} "
                    f"loss={running_loss/running_captions:.4f} rate={rate:.1f} captions/s"
                )
        human_loss = evaluate(model, human_loader, criterion, device)
        objective_loss = evaluate(model, objective_loader, criterion, device)
        score = human_loss + args.objective_loss_weight * objective_loss
        item = {
            "epoch": epoch,
            "replay_repeats": replay[epoch - 1],
            "train_loss": running_loss / running_captions,
            "human_validation_loss": human_loss,
            "objective_validation_loss": objective_loss,
            "selection_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(item)
        common.log("validation " + json.dumps(item))
        model_only = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": item,
            "base_checkpoint": str(args.checkpoint),
            "qwen_used": False,
        }
        atomic_torch_save(model_only, args.output_dir / f"epoch_{epoch}.pth")
        if score < best_score:
            best_score = score
            atomic_torch_save(model_only, args.output_dir / "best.pth")
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, history, args)
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    return history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=common.DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=common.DEFAULT_TOKENIZER)
    parser.add_argument("--captions", type=Path, default=common.DEFAULT_CAPTIONS)
    parser.add_argument("--artemis", type=Path, default=common.DEFAULT_ARTEMIS)
    parser.add_argument("--finetune-stems", type=Path, default=common.DEFAULT_FINETUNE)
    parser.add_argument("--test-stems", type=Path, default=common.DEFAULT_TEST)
    parser.add_argument("--old-test", type=Path, default=common.DEFAULT_OLD_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--replay-repeats", default="3,2,1")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=4, help="Paintings per batch")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--objective-loss-weight", type=float, default=0.15)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--limit-train-paintings", type=int)
    parser.add_argument("--limit-val-paintings", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 16 <= args.max_tokens <= 128:
        raise ValueError("--max-tokens must be between 16 and 128")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    common.log("loading all eligible ArtEmis utterances and clean CATR captions")
    grouped = common.load_artemis(args.artemis)
    catr = common.load_catr(args.captions)
    finetune_stems = set(args.finetune_stems.read_text().splitlines())
    test_stems = set(args.test_stems.read_text().splitlines())
    if finetune_stems & test_stems:
        raise RuntimeError("Fine-tune/test stem overlap")
    paintings = make_paintings(finetune_stems, grouped, catr)
    train_rows, val_rows = stratified_train_validation(
        paintings, args.val_ratio, args.seed
    )
    train_rows = balanced_limit(
        train_rows, args.limit_train_paintings, args.seed + 10
    )
    val_rows = balanced_limit(val_rows, args.limit_val_paintings, args.seed + 11)
    test_rows = common.balanced_sample(
        common.make_examples(test_stems, grouped, catr), args.test_size, args.seed + 12
    )
    train_stem_set = {item["stem"] for item in train_rows}
    val_stem_set = {item["stem"] for item in val_rows}
    test_stem_set = {item["stem"] for item in test_rows}
    if train_stem_set & val_stem_set or (train_stem_set | val_stem_set) & test_stem_set:
        raise RuntimeError("Prepared split overlap")
    human_train = sum(len(item["human_captions"]) for item in train_rows)
    human_val = sum(len(item["human_captions"]) for item in val_rows)
    common.log(
        f"paintings train={len(train_rows):,} val={len(val_rows):,} "
        f"held_out_test={len(test_rows):,}; human captions train={human_train:,} "
        f"val={human_val:,}"
    )
    common.write_jsonl(args.output_dir / "train_manifest.jsonl", train_rows)
    common.write_jsonl(args.output_dir / "validation_manifest.jsonl", val_rows)
    common.write_jsonl(args.output_dir / "test_manifest.jsonl", test_rows)

    tokenizer = common.BertTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    model = common.build_model(args.checkpoint, device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    common.log(f"trainable language parameters={trainable:,}/{total:,}")
    history = train_full(model, train_rows, val_rows, tokenizer, args, device)

    best = torch.load(args.output_dir / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    generated = multicap.generate_multicaption(
        model,
        test_rows,
        tokenizer,
        device,
        batch_size=16,
        max_length=48,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    common.write_jsonl(args.output_dir / "test_generations.jsonl", generated)
    old_map = common.load_old_emotional(args.old_test)
    old = [
        {**row, "caption": old_map[(row["style"], row["stem"])]}
        for row in test_rows
        if (row["style"], row["stem"]) in old_map
    ]
    baseline = [{**row, "caption": row["catr_caption"]} for row in test_rows]
    metrics = [
        audit.metrics("CATR COCO baseline", baseline),
        audit.metrics("CATR + full clean ArtEmis", generated),
    ]
    if old:
        metrics.append(audit.metrics("Old emotional model", old))
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as target:
        fields = sorted({key for item in metrics for key in item})
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "qwen_used": False,
        "base": "public CATR v3, human MS COCO supervision",
        "train_paintings": len(train_rows),
        "validation_paintings": len(val_rows),
        "human_train_captions": human_train,
        "human_validation_captions": human_val,
        "held_out_test_paintings": len(test_rows),
        "history": history,
        "selected_epoch": int(best["epoch"]),
        "metrics": metrics,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    common.log("final metrics:\n" + json.dumps(metrics, indent=2))
    common.log(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()
