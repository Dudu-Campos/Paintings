"""Small, leakage-conscious CATR -> rich ArtEmis caption pilot.

The public CATR v3 checkpoint was trained on human MS COCO captions.  This
pilot keeps its visual representation frozen and adapts only the language
decoder to structured targets composed of:

  1. CATR's objective COCO-style sentence;
  2. two distinct human ArtEmis observations, preferably with different
     emotion annotations.

Train/validation paintings come only from ``finetune_stems.txt``.  The final
comparison comes only from the disjoint official ``test_stems.txt``.  Explicit
mentions of the target art-style label are excluded from supervision.

All outputs use a new directory; the released CATR and previous emotional
caption model are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
CATR_ROOT = ROOT / "outputs/external/catr"
if str(CATR_ROOT) not in sys.path:
    sys.path.insert(0, str(CATR_ROOT))

from configuration import Config  # noqa: E402
from datasets.coco import val_transform  # noqa: E402
from models import caption as caption_module  # noqa: E402
from models.transformer import generate_square_subsequent_mask  # noqa: E402
from models.utils import NestedTensor, nested_tensor_from_tensor_list  # noqa: E402
from transformers import BertTokenizer  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "outputs/catr_model/catr_v3_coco.pth"
DEFAULT_TOKENIZER = ROOT / "outputs/catr_model/bert-base-uncased-tokenizer"
DEFAULT_CAPTIONS = ROOT / "outputs/affect_captions/catr_coco_full.jsonl"
DEFAULT_ARTEMIS = ROOT / "raw/archive(11).zip"
DEFAULT_FINETUNE = ROOT / "data/artemis/finetune_stems.txt"
DEFAULT_TEST = ROOT / "data/artemis/test_stems.txt"
DEFAULT_OLD_TEST = ROOT / "outputs/affect_captions/artemis_test_captions.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/catr_artemis_rich_pilot"
ARTEMIS_MEMBER = "artemis_dataset_release_v0.csv"
SEED = 42


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item.strip()]


def clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def style_is_named(text: str, style: str) -> bool:
    normalized = f" {normalize_text(text)} "
    phrase = normalize_text(style.replace("_", " "))
    aliases = {phrase}
    generic = {
        "action", "art", "color", "early", "field", "high", "late",
        "modern", "new", "northern", "painting", "school",
    }
    for token in phrase.split():
        if len(token) >= 5 and token not in generic:
            aliases.add(token)
            if token.endswith("ism"):
                aliases.add(token[:-3] + "ist")
    return any(f" {alias} " in normalized for alias in aliases if alias)


def lexical_jaccard(left: str, right: str) -> float:
    a, b = set(words(left)), set(words(right))
    return len(a & b) / max(1, len(a | b))


def choose_human_pair(rows: list[dict], style: str) -> tuple[dict, dict] | None:
    unique: dict[str, dict] = {}
    for row in rows:
        utterance = clean_sentence(row["utterance"])
        normalized = normalize_text(utterance)
        count = len(words(utterance))
        if not normalized or count < 8 or count > 40:
            continue
        if style_is_named(utterance, style):
            continue
        unique.setdefault(normalized, {**row, "utterance": utterance})
    candidates = list(unique.values())
    if len(candidates) < 2:
        return None

    best: tuple[float, dict, dict] | None = None
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            diversity = 1.0 - lexical_jaccard(left["utterance"], right["utterance"])
            different_emotion = float(left["emotion"] != right["emotion"])
            length = min(60, len(words(left["utterance"])) + len(words(right["utterance"]))) / 60
            score = 3.0 * different_emotion + 2.0 * diversity + length
            candidate = (score, left, right)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best[1], best[2]


def load_artemis(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        with archive.open(ARTEMIS_MEMBER) as source:
            reader = csv.DictReader(line.decode("utf-8") for line in source)
            for row in reader:
                grouped[row["painting"]].append(row)
    return grouped


def load_catr(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            result[(row["style"], row["stem"])] = row
    return result


def make_examples(
    stems: set[str], grouped: dict[str, list[dict]], catr: dict[tuple[str, str], dict]
) -> list[dict]:
    result = []
    for stem in sorted(stems):
        rows = grouped.get(stem)
        if not rows:
            continue
        style = rows[0]["art_style"]
        objective = catr.get((style, stem))
        pair = choose_human_pair(rows, style)
        if objective is None or pair is None or not Path(objective["path"]).is_file():
            continue
        left, right = pair
        objective_caption = clean_sentence(objective["caption"])
        target = " ".join(
            [objective_caption, left["utterance"], right["utterance"]]
        )
        result.append(
            {
                "stem": stem,
                "style": style,
                "path": objective["path"],
                "catr_caption": objective_caption,
                "human_1": left["utterance"],
                "emotion_1": left["emotion"],
                "human_2": right["utterance"],
                "emotion_2": right["emotion"],
                "target": target,
            }
        )
    return result


def balanced_sample(examples: list[dict], count: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in examples:
        groups[row["style"]].append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    result = []
    styles = sorted(groups)
    while len(result) < count:
        progressed = False
        for style in styles:
            if groups[style] and len(result) < count:
                result.append(groups[style].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(result)
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


class RichCaptionDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer: BertTokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        row = self.examples[index]
        with Image.open(row["path"]) as image:
            pixels = val_transform(image.convert("RGB"))
        # CATR uses 128 decoder inputs and predicts 128 shifted targets.
        encoded = self.tokenizer(
            row["target"],
            max_length=129,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        return pixels, torch.tensor(encoded["input_ids"], dtype=torch.long), row


def collate(batch):
    pixels, tokens, rows = zip(*batch)
    samples = nested_tensor_from_tensor_list(list(pixels))
    return samples, torch.stack(tokens), list(rows)


def build_model(checkpoint: Path, device: torch.device) -> nn.Module:
    config = Config()
    model, _ = caption_module.build_model(config)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.transformer.embeddings, model.transformer.decoder, model.mlp):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return model.to(device)


def keep_frozen_modules_in_eval(model: nn.Module) -> None:
    model.backbone.eval()
    model.input_proj.eval()
    model.transformer.encoder.eval()


def loss_for_batch(
    model: nn.Module,
    samples: NestedTensor,
    tokens: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    outputs = model(samples, tokens[:, :-1], tokens[:, :-1].eq(0))
    return criterion(outputs.permute(0, 2, 1), tokens[:, 1:])


@torch.no_grad()
def validation_loss(model, loader, criterion, device, amp: bool) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for samples, tokens, _ in loader:
        samples = samples.to(device)
        tokens = tokens.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loss = loss_for_batch(model, samples, tokens, criterion)
        total += float(loss)
        batches += 1
    return total / max(1, batches)


def train(
    model,
    train_loader,
    val_loader,
    device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    amp: bool,
) -> list[dict]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best_loss = math.inf
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        keep_frozen_modules_in_eval(model)
        running = 0.0
        for step, (samples, tokens, _) in enumerate(train_loader, 1):
            samples = samples.to(device)
            tokens = tokens.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = loss_for_batch(model, samples, tokens, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 0.1)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            if step == 1 or step % 25 == 0 or step == len(train_loader):
                log(
                    f"epoch={epoch}/{epochs} step={step}/{len(train_loader)} "
                    f"train_loss={running / step:.4f}"
                )
        val = validation_loss(model, val_loader, criterion, device, amp)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "validation_loss": val,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        log(f"epoch={epoch} validation_loss={val:.4f}")
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "validation_loss": val,
            "pilot": True,
            "base": str(DEFAULT_CHECKPOINT),
        }
        torch.save(checkpoint, output_dir / "last.pth")
        if val < best_loss:
            best_loss = val
            torch.save(checkpoint, output_dir / "best.pth")
    return history


@torch.inference_mode()
def generate_batch(
    model: nn.Module,
    images: list[torch.Tensor],
    tokenizer: BertTokenizer,
    max_length: int,
    device: torch.device,
) -> list[dict]:
    samples = nested_tensor_from_tensor_list([image.to(device) for image in images])
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
    batch_size = len(images)
    start_id = int(tokenizer.cls_token_id)
    end_id = int(tokenizer.sep_token_id)
    tokens = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    token_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)
    tokens[:, 0] = start_id
    token_mask[:, 0] = False
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    lengths = torch.full((batch_size,), max_length, dtype=torch.long, device=device)

    for position in range(max_length - 1):
        target = model.transformer.embeddings(tokens).permute(1, 0, 2)
        query_position = (
            model.transformer.embeddings.position_embeddings.weight[:max_length]
            .unsqueeze(1)
            .repeat(1, batch_size, 1)
        )
        hidden = model.transformer.decoder(
            target,
            memory,
            memory_key_padding_mask=source_mask,
            tgt_key_padding_mask=token_mask,
            pos=position_sequence,
            query_pos=query_position,
            tgt_mask=generate_square_subsequent_mask(max_length).to(device),
        )
        predicted = torch.argmax(model.mlp(hidden[position]), dim=-1)
        newly_finished = (~finished) & predicted.eq(end_id)
        lengths[newly_finished] = position + 2
        predicted = torch.where(finished, torch.full_like(predicted, end_id), predicted)
        tokens[:, position + 1] = predicted
        token_mask[:, position + 1] = False
        finished |= predicted.eq(end_id)
        if bool(finished.all()):
            break

    result = []
    for sequence, ended, length in zip(
        tokens.detach().cpu().tolist(), finished.cpu().tolist(), lengths.cpu().tolist()
    ):
        result.append(
            {
                "caption": tokenizer.decode(sequence, skip_special_tokens=True).strip().capitalize(),
                "terminated_sep": bool(ended),
                "generated_tokens": int(length),
            }
        )
    return result


def generate(model, examples, tokenizer, device, batch_size, max_length):
    model.eval()
    rows = []
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        images = []
        for row in batch:
            with Image.open(row["path"]) as image:
                images.append(val_transform(image.convert("RGB")))
        predictions = generate_batch(model, images, tokenizer, max_length, device)
        for source, prediction in zip(batch, predictions):
            rows.append({**source, **prediction})
        log(f"generation={min(offset + batch_size, len(examples))}/{len(examples)}")
    return rows


def load_old_emotional(path: Path) -> dict[tuple[str, str], str]:
    result = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            result[(row["split"], row["image_id"])] = row["caption"]
    return result


def caption_metrics(name: str, rows: list[dict], caption_key: str) -> dict:
    captions = [row[caption_key] for row in rows]
    sentence_lists = [sentences(text) for text in captions]
    normalized_sentences = [[normalize_text(item) for item in items] for items in sentence_lists]
    repeated = [len(items) != len(set(items)) for items in normalized_sentences]
    consecutive = [
        any(a == b for a, b in zip(items, items[1:])) for items in normalized_sentences
    ]
    normalized_captions = [normalize_text(text) for text in captions]
    style_mentions = [style_is_named(row[caption_key], row["style"]) for row in rows]
    result = {
        "condition": name,
        "images": len(rows),
        "mean_words": float(np.mean([len(words(text)) for text in captions])),
        "median_words": float(np.median([len(words(text)) for text in captions])),
        "mean_sentences": float(np.mean([len(items) for items in sentence_lists])),
        "images_with_repeated_sentence_rate": float(np.mean(repeated)),
        "images_with_consecutive_repeat_rate": float(np.mean(consecutive)),
        "unique_caption_rate": len(set(normalized_captions)) / max(1, len(captions)),
        "explicit_style_mention_rate": float(np.mean(style_mentions)),
    }
    if caption_key == "caption" and rows and "terminated_sep" in rows[0]:
        result["sep_termination_rate"] = float(
            np.mean([row["terminated_sep"] for row in rows])
        )
        result["hit_length_limit_rate"] = float(
            np.mean([not row["terminated_sep"] for row in rows])
        )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--artemis", type=Path, default=DEFAULT_ARTEMIS)
    parser.add_argument("--finetune-stems", type=Path, default=DEFAULT_FINETUNE)
    parser.add_argument("--test-stems", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--old-test", type=Path, default=DEFAULT_OLD_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-generation-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot expects the available CUDA GPU")
    if args.train_size + args.val_size < 2:
        raise ValueError("train + validation sample is too small")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    amp = True

    log("loading ArtEmis human responses")
    grouped = load_artemis(args.artemis)
    log("loading cached clean CATR captions")
    catr = load_catr(args.captions)
    finetune_stems = set(args.finetune_stems.read_text().splitlines())
    test_stems = set(args.test_stems.read_text().splitlines())
    if finetune_stems & test_stems:
        raise RuntimeError("Official fine-tune/test stem sets overlap")

    candidates = make_examples(finetune_stems, grouped, catr)
    selected = balanced_sample(candidates, args.train_size + args.val_size, args.seed)
    train_rows = selected[: args.train_size]
    val_rows = selected[args.train_size :]
    test_candidates = make_examples(test_stems, grouped, catr)
    test_rows = balanced_sample(test_candidates, args.test_size, args.seed + 1)
    if len(train_rows) != args.train_size or len(val_rows) != args.val_size:
        raise RuntimeError("Not enough fine-tune examples")
    if len(test_rows) != args.test_size:
        raise RuntimeError("Not enough held-out test examples")
    if {x["stem"] for x in selected} & {x["stem"] for x in test_rows}:
        raise RuntimeError("Pilot train/validation overlaps held-out test")

    write_jsonl(args.output_dir / "train_manifest.jsonl", train_rows)
    write_jsonl(args.output_dir / "validation_manifest.jsonl", val_rows)
    write_jsonl(args.output_dir / "test_manifest.jsonl", test_rows)
    log(
        f"samples train={len(train_rows)} validation={len(val_rows)} "
        f"held_out_test={len(test_rows)}"
    )

    tokenizer = BertTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    train_dataset = RichCaptionDataset(train_rows, tokenizer)
    val_dataset = RichCaptionDataset(val_rows, tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
    )

    model = build_model(args.checkpoint, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"model parameters trainable={trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")
    history = train(
        model,
        train_loader,
        val_loader,
        device,
        args.output_dir,
        args.epochs,
        args.lr,
        amp,
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    best = torch.load(args.output_dir / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    generated = generate(
        model,
        test_rows,
        tokenizer,
        device,
        args.batch_size,
        args.max_generation_length,
    )
    old = load_old_emotional(args.old_test)
    comparable_old = []
    for row in test_rows:
        caption = old.get((row["style"], row["stem"]))
        if caption:
            comparable_old.append({**row, "old_caption": caption})

    metrics = [
        caption_metrics("CATR COCO baseline", test_rows, "catr_caption"),
        caption_metrics("Rich human targets", test_rows, "target"),
        caption_metrics("CATR ArtEmis rich pilot", generated, "caption"),
    ]
    if comparable_old:
        metrics.append(
            caption_metrics("Old emotional model", comparable_old, "old_caption")
        )
    write_jsonl(args.output_dir / "test_generations.jsonl", generated)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as target:
        fieldnames = sorted({key for row in metrics for key in row})
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    examples = []
    for row in generated[:20]:
        examples.append(
            {
                "style": row["style"],
                "stem": row["stem"],
                "catr_before": row["catr_caption"],
                "human_target": row["target"],
                "pilot_after": row["caption"],
                "terminated_sep": row["terminated_sep"],
            }
        )
    write_jsonl(args.output_dir / "examples.jsonl", examples)

    run = {
        "seed": args.seed,
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": sha256(args.checkpoint),
        "base_supervision": "MS COCO 2017 human captions",
        "qwen_used": False,
        "train_size": len(train_rows),
        "validation_size": len(val_rows),
        "held_out_test_size": len(test_rows),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["validation_loss"]),
        "trained_modules": ["transformer.embeddings", "transformer.decoder", "mlp"],
        "frozen_modules": ["backbone", "input_proj", "transformer.encoder"],
        "target": "CATR objective caption + two distinct ArtEmis human observations",
        "style_label_filter": True,
        "metrics": metrics,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("metrics:\n" + json.dumps(metrics, indent=2))
    log(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()
