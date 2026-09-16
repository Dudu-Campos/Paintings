"""Zero-shot CLIP style classification on WikiArt -- leakage probe.

CLIP is used elsewhere in this project purely as a frozen *visual* encoder
(``experiments/image-to-text/04_student.py``'s ClipCap-style prefix,
``src/baselines/extract_clip_features.py``'s dumped embeddings) -- never as a
style classifier. This script asks a different question: with **zero
training**, can CLIP's own text-image similarity already guess a WikiArt
painting's style far above chance, just by comparing the image embedding to a
one-line prompt per style ("a painting in the {style} style")? If it can,
that is not evidence of learned visual style perception -- CLIP never saw a
gradient on this task -- it is evidence that its pre-training corpus already
contains WikiArt (or close copies of it) captioned with the style name, i.e.
the same memorization confound probed on the VLM side by
``experiments/leak_probe_qwen.py`` (see its ``style`` field).

Two independent checkpoints, trained on two disjoint pre-training corpora,
are evaluated so the result is not an artifact of one particular CLIP release
or one particular training set (OpenAI's private WIT vs. DataComp-1B):

* ``clip-vit-b32``    -- HF ``transformers`` ``openai/clip-vit-base-patch32``,
  the exact checkpoint used as the frozen encoder in the image-to-text
  distillation pipeline (``experiments/image-to-text/04_student.py``).
* ``open-clip-vit-l14`` -- ``open_clip`` ``ViT-L-14`` (``datacomp_xl_s13b_b90k``
  pretrained, i.e. trained on DataComp-1B rather than OpenAI's original WIT
  corpus), the exact checkpoint used in
  ``src/baselines/extract_clip_features.py`` to dump WikiArt visual features
  for the rest of the paper's baselines.

Ground truth style is read the same way as
``experiments/image-to-text/05_evaluate.py::wikiart_index`` -- the top-level
directory of each file in WikiArt's own ``classes.csv``. The class list for
the zero-shot prompts is taken from whatever style folders are actually
present in the evaluated split (not a hard-coded 27/28 constant), so the
comparison is exact regardless of the WikiArt release in use.

Metrics mirror the project's downstream classification convention
(``results/catr_coco_standalone_results.csv``: balanced accuracy, plain
accuracy, macro-F1) so this can be placed directly next to the CATR/COCO
"clean" baseline (BAcc=0.18138) and the caption TF-IDF/SVM classifier.

Outputs (under --out-dir, default ``results/``):
    clip_zeroshot_style_results.csv       one row per backend: BAcc/acc/macro-F1
    clip_zeroshot_style_predictions.csv   per-image true vs. predicted style

Usage:
    python experiments/clip_zeroshot_style.py \\
        --wikiart-root ~/.cache/kagglehub/datasets/steubk/wikiart/versions/1 \\
        --subset test --limit 2700 --backends both
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
CATR_REFERENCE_BACC = 0.18138  # results/catr_coco_standalone_results.csv


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# Ground truth (same convention as experiments/image-to-text/05_evaluate.py)
# --------------------------------------------------------------------------
def wikiart_index(wikiart_root: Path, subset: str, limit: int, seed: int) -> pd.DataFrame:
    classes = pd.read_csv(wikiart_root / "classes.csv")
    classes = classes[classes["filename"].apply(lambda f: (wikiart_root / f).is_file())]
    if subset:
        classes = classes[classes["subset"].astype(str) == subset]
    classes = classes.copy()
    classes["style"] = classes["filename"].apply(lambda f: str(Path(f).parts[0]))
    if limit:
        per = max(1, limit // classes["style"].nunique())
        parts = [g.sample(min(len(g), per), random_state=seed)
                 for _, g in classes.groupby("style")]
        classes = pd.concat(parts, ignore_index=True)
    classes["path"] = classes["filename"].apply(lambda f: str(wikiart_root / f))
    return classes[["filename", "style", "path"]].reset_index(drop=True)


def style_prompt(style: str, template: str) -> str:
    return template.format(style=style.replace("_", " "))


# --------------------------------------------------------------------------
# Backends -- each exposes encode_text(prompts) and encode_images(paths),
# both returning L2-normalized (n, d) float32 arrays.
# --------------------------------------------------------------------------
class HFClipBackend:
    name = "clip-vit-b32"
    model_id = "openai/clip-vit-base-patch32"

    def __init__(self, device: str, batch_size: int):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(self.model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            feats = self.model.get_text_features(**inputs)
        # transformers>=5 returns BaseModelOutputWithPooling; older versions
        # returned the tensor directly (see 04_student.py::build_clip_cache).
        if not isinstance(feats, self.torch.Tensor):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def encode_images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        out = np.zeros((len(paths), self.model.config.projection_dim), dtype=np.float32)
        for i in range(0, len(paths), self.batch_size):
            chunk = paths[i:i + self.batch_size]
            images = [Image.open(p).convert("RGB") for p in chunk]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                feats = self.model.get_image_features(**inputs)
            if not isinstance(feats, self.torch.Tensor):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out[i:i + len(chunk)] = feats.float().cpu().numpy()
            if (i // self.batch_size) % 10 == 0:
                log(f"  [{self.name}] {min(i + self.batch_size, len(paths))}/{len(paths)}")
        return out


class OpenClipBackend:
    name = "open-clip-vit-l14"
    model_id = "ViT-L-14"
    pretrained = "datacomp_xl_s13b_b90k"

    def __init__(self, device: str, batch_size: int):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_id, pretrained=self.pretrained, device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_id)

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(prompts).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def encode_images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        dim = self.model.visual.output_dim
        out = np.zeros((len(paths), dim), dtype=np.float32)
        for i in range(0, len(paths), self.batch_size):
            chunk = paths[i:i + self.batch_size]
            batch = self.torch.stack(
                [self.preprocess(Image.open(p).convert("RGB")) for p in chunk]
            ).to(self.device)
            with self.torch.no_grad(), self.torch.amp.autocast(
                "cuda" if self.device == "cuda" else "cpu"
            ):
                feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out[i:i + len(chunk)] = feats.float().cpu().numpy()
            if (i // self.batch_size) % 10 == 0:
                log(f"  [{self.name}] {min(i + self.batch_size, len(paths))}/{len(paths)}")
        return out


BACKENDS = {"clip-vit-b32": HFClipBackend, "open-clip-vit-l14": OpenClipBackend}


# --------------------------------------------------------------------------
# Zero-shot prediction: argmax cosine similarity, no training whatsoever.
# --------------------------------------------------------------------------
def zero_shot_predict(backend, image_paths: list[str], classes: list[str], template: str) -> np.ndarray:
    text_feats = backend.encode_text([style_prompt(c, template) for c in classes])
    image_feats = backend.encode_images(image_paths)
    similarity = image_feats @ text_feats.T
    return np.array(classes)[similarity.argmax(axis=1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikiart-root", type=Path, required=True)
    parser.add_argument("--subset", default="test")
    parser.add_argument("--limit", type=int, default=0, help="0 = all images in the subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backends", choices=["clip-vit-b32", "open-clip-vit-l14", "both"],
                        default="both")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--template", default="a painting in the {style} style")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    idx = wikiart_index(args.wikiart_root, args.subset, args.limit, args.seed)
    classes = sorted(idx["style"].unique())
    log(f"WikiArt {args.subset}: {len(idx)} images, {len(classes)} styles, device={device}")

    backend_names = list(BACKENDS) if args.backends == "both" else [args.backends]
    results_rows = []
    predictions = idx[["filename", "style"]].copy()

    for name in backend_names:
        log(f"Backend: {name} ({BACKENDS[name].model_id})")
        backend = BACKENDS[name](device, args.batch_size)
        pred = zero_shot_predict(backend, list(idx["path"]), classes, args.template)
        predictions[f"pred_{name}"] = pred

        bacc = balanced_accuracy_score(idx["style"], pred)
        acc = accuracy_score(idx["style"], pred)
        macro_f1 = f1_score(idx["style"], pred, average="macro")
        results_rows.append({
            "condition": f"CLIP zero-shot ({name})",
            "n_images": len(idx), "n_styles": len(classes),
            "bacc": bacc, "accuracy": acc, "macro_f1": macro_f1,
        })
        log(f"  {name}: BAcc={bacc:.4f} acc={acc:.4f} macroF1={macro_f1:.4f} "
            f"(CATR/COCO clean reference: {CATR_REFERENCE_BACC:.4f})")

        del backend
        torch.cuda.empty_cache()

    results = pd.DataFrame(results_rows)
    results_path = args.out_dir / "clip_zeroshot_style_results.csv"
    predictions_path = args.out_dir / "clip_zeroshot_style_predictions.csv"
    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    log(f"Saved {results_path}, {predictions_path}")


if __name__ == "__main__":
    main()
