"""Pilot 2: adapt CATR with independent short captions, then sample diversity.

Unlike ``pilot_finetune_catr_artemis_rich.py``, this experiment does not ask a
COCO captioner to learn a multi-sentence target.  Each painting contributes
three conventional caption records (one original CATR caption and two distinct
human ArtEmis observations).  At inference, the unchanged cached CATR sentence
is followed by two reproducibly sampled emotional captions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import pilot_finetune_catr_artemis_rich as common


ROOT = common.ROOT
DEFAULT_OUTPUT = ROOT / "outputs/catr_artemis_multicaption_pilot"


def expand_independent(rows: list[dict], include_objective: bool) -> list[dict]:
    expanded = []
    for row in rows:
        targets = [
            ("human_emotional_1", row["human_1"]),
            ("human_emotional_2", row["human_2"]),
        ]
        if include_objective:
            targets.insert(0, ("objective_catr", row["catr_caption"]))
        for target_kind, target in targets:
            expanded.append({**row, "target": target, "target_kind": target_kind})
    return expanded


def build_conservative_model(checkpoint: Path, device: torch.device) -> nn.Module:
    model = common.build_model(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Preserve vocabulary embeddings and most of the pretrained language model.
    for layer in model.transformer.decoder.layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if model.transformer.decoder.norm is not None:
        for parameter in model.transformer.decoder.norm.parameters():
            parameter.requires_grad = True
    for parameter in model.mlp.parameters():
        parameter.requires_grad = True
    return model


def ban_repeated_ngram(
    logits: torch.Tensor, token_ids: torch.Tensor, position: int, ngram_size: int
) -> None:
    if position + 1 < ngram_size:
        return
    sequences = token_ids[:, : position + 1].detach().cpu().tolist()
    for row_index, sequence in enumerate(sequences):
        prefix = tuple(sequence[-(ngram_size - 1) :])
        for start in range(len(sequence) - ngram_size + 1):
            if tuple(sequence[start : start + ngram_size - 1]) == prefix:
                logits[row_index, sequence[start + ngram_size - 1]] = -torch.inf


def top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    sampled_sorted = torch.multinomial(probabilities, num_samples=1)
    return sorted_indices.gather(1, sampled_sorted).squeeze(1)


@torch.inference_mode()
def sample_batch(
    model: nn.Module,
    images: list[torch.Tensor],
    tokenizer,
    max_length: int,
    min_length: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[dict]:
    samples = common.nested_tensor_from_tensor_list([image.to(device) for image in images])
    features, positions = model.backbone(samples)
    source, source_mask = features[-1].decompose()
    projected = model.input_proj(source)
    source_sequence = projected.flatten(2).permute(2, 0, 1)
    position_sequence = positions[-1].flatten(2).permute(2, 0, 1)
    source_mask = source_mask.flatten(1)
    memory = model.transformer.encoder(
        source_sequence, src_key_padding_mask=source_mask, pos=position_sequence
    )

    batch_size = len(images)
    start_id = int(tokenizer.cls_token_id)
    end_id = int(tokenizer.sep_token_id)
    pad_id = int(tokenizer.pad_token_id)
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
            tgt_mask=common.generate_square_subsequent_mask(max_length).to(device),
        )
        logits = model.mlp(hidden[position]).float() / temperature
        logits[:, start_id] = -torch.inf
        logits[:, pad_id] = -torch.inf
        if position + 1 < min_length:
            logits[:, end_id] = -torch.inf
        ban_repeated_ngram(logits, tokens, position, ngram_size=3)
        predicted = top_p_sample(logits, top_p)
        newly_finished = (~finished) & predicted.eq(end_id)
        lengths[newly_finished] = position + 2
        predicted = torch.where(finished, torch.full_like(predicted, end_id), predicted)
        tokens[:, position + 1] = predicted
        token_mask[:, position + 1] = False
        finished |= predicted.eq(end_id)
        if bool(finished.all()):
            break

    output = []
    for sequence, ended, length in zip(
        tokens.cpu().tolist(), finished.cpu().tolist(), lengths.cpu().tolist()
    ):
        output.append(
            {
                "text": tokenizer.decode(sequence, skip_special_tokens=True).strip(),
                "terminated": bool(ended),
                "tokens": int(length),
            }
        )
    return output


def generate_multicaption(
    model,
    examples,
    tokenizer,
    device,
    batch_size,
    max_length,
    sample_count,
    seed,
):
    model.eval()
    rows = []
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        images = []
        for row in batch:
            with Image.open(row["path"]) as image:
                images.append(common.val_transform(image.convert("RGB")))
        sampled_runs = []
        for sample_index in range(sample_count):
            torch.manual_seed(seed + 10_000 * sample_index + offset)
            torch.cuda.manual_seed_all(seed + 10_000 * sample_index + offset)
            sampled_runs.append(
                sample_batch(
                    model,
                    images,
                    tokenizer,
                    max_length=max_length,
                    min_length=7,
                    temperature=0.8,
                    top_p=0.9,
                    device=device,
                )
            )
        for item_index, source in enumerate(batch):
            candidates = [run[item_index] for run in sampled_runs]
            accepted = []
            seen = {common.normalize_text(source["catr_caption"])}
            for candidate in candidates:
                text = common.clean_sentence(candidate["text"])
                normalized = common.normalize_text(text)
                if normalized and normalized not in seen:
                    accepted.append({**candidate, "text": text})
                    seen.add(normalized)
                if len(accepted) == 2:
                    break
            parts = [source["catr_caption"], *[item["text"] for item in accepted]]
            rows.append(
                {
                    **source,
                    "caption": " ".join(parts),
                    "sampled_captions": [item["text"] for item in accepted],
                    "accepted_samples": len(accepted),
                    "terminated_sep": all(item["terminated"] for item in accepted),
                    "generated_tokens": sum(item["tokens"] for item in accepted),
                }
            )
        common.log(f"generation={min(offset + batch_size, len(examples))}/{len(examples)}")
    return rows


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
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-generation-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--include-objective", action="store_true")
    parser.add_argument("--all-language", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    common.log("loading ArtEmis and cached CATR")
    grouped = common.load_artemis(args.artemis)
    catr = common.load_catr(args.captions)
    finetune_stems = set(args.finetune_stems.read_text().splitlines())
    test_stems = set(args.test_stems.read_text().splitlines())
    if finetune_stems & test_stems:
        raise RuntimeError("Fine-tune/test overlap")
    candidates = common.make_examples(finetune_stems, grouped, catr)
    selected = common.balanced_sample(
        candidates, args.train_size + args.val_size, args.seed
    )
    train_paintings = selected[: args.train_size]
    val_paintings = selected[args.train_size :]
    test_paintings = common.balanced_sample(
        common.make_examples(test_stems, grouped, catr), args.test_size, args.seed + 1
    )
    if len(train_paintings) != args.train_size or len(val_paintings) != args.val_size:
        raise RuntimeError("Insufficient fine-tune sample")
    if len(test_paintings) != args.test_size:
        raise RuntimeError("Insufficient held-out sample")
    if {x["stem"] for x in selected} & {x["stem"] for x in test_paintings}:
        raise RuntimeError("Train/test overlap")
    train_rows = expand_independent(train_paintings, args.include_objective)
    val_rows = expand_independent(val_paintings, args.include_objective)
    common.write_jsonl(args.output_dir / "train_paintings.jsonl", train_paintings)
    common.write_jsonl(args.output_dir / "validation_paintings.jsonl", val_paintings)
    common.write_jsonl(args.output_dir / "test_paintings.jsonl", test_paintings)
    common.write_jsonl(args.output_dir / "train_caption_records.jsonl", train_rows)
    common.log(
        f"paintings train={len(train_paintings)} val={len(val_paintings)} "
        f"test={len(test_paintings)}; caption records={len(train_rows)}"
    )

    tokenizer = common.BertTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    train_loader = DataLoader(
        common.RichCaptionDataset(train_rows, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=common.collate,
    )
    val_loader = DataLoader(
        common.RichCaptionDataset(val_rows, tokenizer),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=common.collate,
    )
    model = (
        common.build_model(args.checkpoint, device)
        if args.all_language
        else build_conservative_model(args.checkpoint, device)
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    common.log(f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")
    history = common.train(
        model,
        train_loader,
        val_loader,
        device,
        args.output_dir,
        args.epochs,
        args.lr,
        amp=True,
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    best = torch.load(args.output_dir / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    generated = generate_multicaption(
        model,
        test_paintings,
        tokenizer,
        device,
        args.batch_size,
        args.max_generation_length,
        args.sample_count,
        args.seed,
    )
    common.write_jsonl(args.output_dir / "test_generations.jsonl", generated)

    old = common.load_old_emotional(args.old_test)
    comparable_old = [
        {**row, "old_caption": old[(row["style"], row["stem"])]}
        for row in test_paintings
        if (row["style"], row["stem"]) in old
    ]
    metrics = [
        common.caption_metrics("CATR COCO baseline", test_paintings, "catr_caption"),
        common.caption_metrics("CATR + 2 sampled ArtEmis", generated, "caption"),
    ]
    if comparable_old:
        metrics.append(common.caption_metrics("Old emotional model", comparable_old, "old_caption"))
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as target:
        fieldnames = sorted({key for row in metrics for key in row})
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "approach": "independent short captions + diverse top-p sampling",
        "qwen_used": False,
        "base": "public CATR v3 trained on human MS COCO captions",
        "train_paintings": len(train_paintings),
        "train_caption_records": len(train_rows),
        "validation_paintings": len(val_paintings),
        "held_out_test_paintings": len(test_paintings),
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["validation_loss"]),
        "learning_rate": args.lr,
        "include_objective_in_finetune": args.include_objective,
        "train_scope": "all_language" if args.all_language else "last_two_decoder_layers",
        "metrics": metrics,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    common.log("metrics:\n" + json.dumps(metrics, indent=2))
    common.log(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()
