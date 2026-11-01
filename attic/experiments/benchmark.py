"""Configuration-driven launcher for data/model benchmark matrices.

Examples
--------
python experiments/benchmark.py list
python experiments/benchmark.py run --datasets wikiart pandora rijksmuseum \\
    --models engineered_mlp resnet50_scratch resnet50_frozen --dry-run
python experiments/benchmark.py run --datasets pandora --models engineered_mlp
python experiments/benchmark.py summarize
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "benchmarking"))
from config import dataset_specs, model_specs  # noqa: E402


def command_for(dataset, model, folds: int, workers: int) -> list[str]:
    # Checkpoints, OOF predictions and per-fold work files stay local.  The
    # compact table promoted to the manuscript is produced by ``summarize`` in
    # results/benchmarks/.
    output = ROOT / "outputs" / "benchmarks" / dataset.name / model.name
    if model.runner == "blocks":
        command = [
            sys.executable, str(ROOT / "experiments" / "evaluate_benchmark_blocks.py"),
            "--dataset", dataset.name, "--blocks", *model.blocks,
            "--classifier", str(model.classifier), "--folds", str(folds),
            "--epochs", str(model.epochs), "--patience", str(model.patience),
            "--output-dir", str(output),
        ]
        if model.allow_external:
            command.append("--allow-external")
        return command
    if model.runner == "resnet":
        return [
            sys.executable, str(ROOT / "src" / "baselines" / "train_resnet_wikiart.py"),
            "--image-dir", str(dataset.image_dir), "--meta", str(dataset.metadata),
            "--task", dataset.task, "--min-per-class", str(dataset.min_per_class),
            "--init", str(model.initialization), "--folds", str(folds),
            "--workers", str(workers), "--epochs", str(model.epochs),
            "--patience", str(model.patience), "--output", str(output),
        ]
    if model.runner == "command":
        if not model.command:
            raise ValueError(f"{model.name}: runner=command requires a command array")
        output.mkdir(parents=True, exist_ok=True)
        values = {
            "python": sys.executable,
            "root": str(ROOT),
            "dataset": dataset.name,
            "image_dir": str(dataset.image_dir),
            "metadata": str(dataset.metadata),
            "task": dataset.task,
            "min_per_class": str(dataset.min_per_class),
            "folds": str(folds),
            "workers": str(workers),
            "output": str(output),
        }
        return [part.format(**values) for part in model.command]
    raise ValueError(f"unknown runner: {model.runner}")


def list_specs() -> None:
    print("Datasets:")
    for item in dataset_specs().values():
        print(f"  {item.name:<14} {item.task:<10} {item.description}")
    print("\nModels:")
    for item in model_specs().values():
        print(f"  {item.name:<24} {item.runner:<7} {item.description}")


def run(args: argparse.Namespace) -> None:
    datasets, models = dataset_specs(), model_specs()
    for name in args.datasets:
        item = datasets[name]
        for path, label in ((item.image_dir, "image_dir"), (item.metadata, "metadata")):
            if not path.exists():
                raise FileNotFoundError(f"{name}: {label} missing: {path}")
    for dataset_name in args.datasets:
        for model_name in args.models:
            command = command_for(datasets[dataset_name], models[model_name], args.folds, args.workers)
            print("\n$ " + " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


def summarize() -> None:
    rows = []
    root = ROOT / "outputs" / "benchmarks"
    for path in sorted(root.glob("*/*/summary.csv")):
        for row in csv.DictReader(path.open()):
            rows.append({"model": path.parent.name, **row, "source": str(path.relative_to(ROOT))})
        # ResNet has a JSON summary, intentionally normalized here rather than
        # requiring its trainer to know benchmark-table conventions.
    for path in sorted(root.glob("*/*/summary.json")):
        data = json.loads(path.read_text())
        mean, std = data["mean"], data["std"]
        config_path = path.parent / "config.json"
        task = json.loads(config_path.read_text()).get("task", "") if config_path.exists() else ""
        rows.append({
            "dataset": path.parent.parent.name, "task": task,
            "condition": data["initialization"], "classifier": "ResNet-50",
            "model": path.parent.name, "folds": len(data["folds"]),
            "n_images": data["n_images"], "n_classes": data["n_classes"],
            "dimensions": "", "bacc_mean": mean["balanced_accuracy"],
            "bacc_sd_sample": std["balanced_accuracy"], "accuracy_mean": mean["accuracy"],
            "macro_f1_mean": mean["macro_f1"], "elapsed_seconds": data["elapsed_seconds"],
            "source": str(path.relative_to(ROOT)),
        })
    if not rows:
        raise FileNotFoundError("No benchmark summaries found. Run a configured model first.")
    fieldnames = sorted({key for row in rows for key in row})
    output = ROOT / "results" / "benchmarks" / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {output}")


def main() -> None:
    datasets, models = dataset_specs(), model_specs()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--datasets", nargs="+", default=list(datasets), choices=list(datasets))
    run_parser.add_argument("--models", nargs="+", default=list(models), choices=list(models))
    run_parser.add_argument("--folds", type=int, default=5)
    run_parser.add_argument("--workers", type=int, default=16)
    run_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "list":
        list_specs()
    elif args.command == "run":
        run(args)
    else:
        summarize()


if __name__ == "__main__":
    main()
