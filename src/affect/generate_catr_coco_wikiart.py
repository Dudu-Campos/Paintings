"""Generate resumable CATR captions for the fixed 12k WikiArt cohort.

The model is the public CATR v3 checkpoint trained on human MS COCO captions.
Unlike the local distilled captioner, this path never uses Qwen-generated
targets.  Encoder outputs are cached during autoregressive decoding so the
released model can be evaluated efficiently without changing its predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CATR_ROOT = ROOT / "outputs/external/catr"
if str(CATR_ROOT) not in sys.path:
    sys.path.insert(0, str(CATR_ROOT))

from configuration import Config  # noqa: E402
from datasets.coco import val_transform  # noqa: E402
from models import caption as caption_module  # noqa: E402
from models.transformer import generate_square_subsequent_mask  # noqa: E402
from models.utils import nested_tensor_from_tensor_list  # noqa: E402
from transformers import BertTokenizer  # noqa: E402


DEFAULT_SAMPLE = ROOT / "paper/qwen_prompt_ablation_sample.csv"
DEFAULT_OUTPUT = ROOT / "outputs/affect_captions/catr_coco_12k.jsonl"
DEFAULT_CHECKPOINT = ROOT / "outputs/catr_model/catr_v3_coco.pth"
DEFAULT_TOKENIZER = ROOT / "outputs/catr_model/bert-base-uncased-tokenizer"
MODEL_NAME = "CATR v3"
MODEL_REPOSITORY = "https://github.com/saahiluppal/catr"
MODEL_COMMIT = "fac82f9b4004b1dd39ccf89760b758ad19a2dbee"
CHECKPOINT_URL = (
    "https://github.com/saahiluppal/catr/releases/download/0.2/"
    "weight493084032.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed resumable output at {path}:{line_number}"
                ) from error
            image_id = str(row.get("image_id", row["stem"]))
            if image_id in completed:
                raise ValueError(f"Duplicate completed image_id: {image_id}")
            completed.add(image_id)
    return completed


def load_cohort(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    required = {"stem", "style", "path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Sample lacks columns: {sorted(missing)}")
    if "image_id" not in frame.columns:
        if frame["stem"].duplicated().any():
            raise ValueError(
                "Sample contains duplicate stems and lacks a unique image_id"
            )
        frame["image_id"] = frame["stem"]
    if frame["image_id"].duplicated().any():
        raise ValueError("Sample contains duplicate image_id values")
    missing_paths = [item for item in frame["path"] if not Path(item).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing image: {missing_paths[0]}")
    return frame.sort_values(["style", "image_id"]).reset_index(drop=True)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    config = Config()
    model, _ = caption_module.build_model(config)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    return model


def load_images(paths: list[str]) -> list[torch.Tensor]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(val_transform(image.convert("RGB")))
    return images


@torch.inference_mode()
def generate_batch(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    tokenizer: BertTokenizer,
    max_length: int,
    device: torch.device,
) -> list[str]:
    """Greedy decoding equivalent to the released CATR predict script.

    CATR's reference script recomputes the ResNet and Transformer encoder for
    every generated token.  Here the visual memory is computed once, while the
    decoder, vocabulary and greedy rule remain unchanged.
    """
    samples = nested_tensor_from_tensor_list(
        [image.to(device, non_blocking=True) for image in images]
    )
    features, positions = model.backbone(samples)
    source, source_mask = features[-1].decompose()
    projected = model.input_proj(source)
    batch_size, channels, height, width = projected.shape
    source_sequence = projected.flatten(2).permute(2, 0, 1)
    position_sequence = positions[-1].flatten(2).permute(2, 0, 1)
    source_mask = source_mask.flatten(1)
    memory = model.transformer.encoder(
        source_sequence,
        src_key_padding_mask=source_mask,
        pos=position_sequence,
    )

    start_id = int(tokenizer.cls_token_id)
    end_id = int(tokenizer.sep_token_id)
    tokens = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    token_mask = torch.ones(
        (batch_size, max_length), dtype=torch.bool, device=device
    )
    tokens[:, 0] = start_id
    token_mask[:, 0] = False
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

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
        logits = model.mlp(hidden[position])
        predicted = torch.argmax(logits, dim=-1)
        predicted = torch.where(
            finished, torch.full_like(predicted, end_id), predicted
        )
        tokens[:, position + 1] = predicted
        token_mask[:, position + 1] = False
        finished |= predicted.eq(end_id)
        if bool(finished.all()):
            break

    captions = []
    for sequence in tokens.detach().cpu().tolist():
        captions.append(
            tokenizer.decode(sequence, skip_special_tokens=True).strip().capitalize()
        )
    return captions


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    if not 2 <= args.max_length <= 128:
        raise ValueError("--max-length must be between 2 and 128")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    cohort = load_cohort(args.sample)
    completed = read_completed(args.output)
    pending = cohort[~cohort["image_id"].isin(completed)]
    if args.limit is not None:
        pending = pending.head(args.limit)
    checkpoint_hash = file_sha256(args.checkpoint)
    print(
        f"{MODEL_NAME}; device={device}; cohort={len(cohort):,}; "
        f"complete={len(completed):,}; pending={len(pending):,}; "
        f"checkpoint_sha256={checkpoint_hash}",
        flush=True,
    )
    if pending.empty:
        print("Nothing to do.", flush=True)
        return

    tokenizer = BertTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    model = load_model(args.checkpoint, device)
    started = time.perf_counter()
    generated = 0

    for offset in range(0, len(pending), args.batch_size):
        batch = pending.iloc[offset : offset + args.batch_size]
        batch_started = time.perf_counter()
        images = load_images(batch["path"].tolist())
        captions = generate_batch(
            model, images, tokenizer, args.max_length, device
        )
        elapsed_batch = time.perf_counter() - batch_started
        rows = []
        for item, generated_caption in zip(
            batch.itertuples(index=False), captions
        ):
            rows.append(
                {
                    "image_id": str(item.image_id),
                    "stem": str(item.stem),
                    "style": str(item.style),
                    "path": str(item.path),
                    "caption": generated_caption,
                    "model": MODEL_NAME,
                    "training_captions": "MS COCO human captions",
                    "repository": MODEL_REPOSITORY,
                    "repository_commit": MODEL_COMMIT,
                    "checkpoint_url": CHECKPOINT_URL,
                    "checkpoint_sha256": checkpoint_hash,
                    "decoding": "greedy",
                    "max_length": args.max_length,
                }
            )
        append_rows(args.output, rows)
        generated += len(rows)
        total_done = len(completed) + generated
        if (
            generated == len(rows)
            or generated % args.log_every < len(rows)
            or generated == len(pending)
        ):
            elapsed = time.perf_counter() - started
            rate = generated / elapsed
            remaining = len(pending) - generated
            eta_hours = remaining / rate / 3600 if rate else float("inf")
            print(
                f"{total_done:,}/{len(cohort):,}; "
                f"{rate:.2f} images/s; batch={elapsed_batch:.1f}s; "
                f"ETA={eta_hours:.2f}h",
                flush=True,
            )

    print(
        f"Finished {generated:,} new captions in "
        f"{(time.perf_counter() - started) / 3600:.2f}h.",
        flush=True,
    )


if __name__ == "__main__":
    main()
