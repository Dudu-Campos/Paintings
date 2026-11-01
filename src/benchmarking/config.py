"""Read benchmark specifications without tying experiments to one dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "benchmarks"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_dir: Path
    metadata: Path
    task: str
    min_per_class: int
    description: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    runner: str
    description: str
    classifier: str | None = None
    blocks: tuple[str, ...] = ()
    allow_external: bool = False
    initialization: str | None = None
    command: tuple[str, ...] = ()
    epochs: int = 150
    patience: int = 15


def _read(filename: str) -> dict:
    with (CONFIG_DIR / filename).open("rb") as handle:
        return tomllib.load(handle)


def dataset_specs() -> dict[str, DatasetSpec]:
    table = _read("datasets.toml")["datasets"]
    return {
        name: DatasetSpec(
            name=name,
            image_dir=ROOT / item["image_dir"],
            metadata=ROOT / item["metadata"],
            task=item["task"],
            min_per_class=int(item.get("min_per_class", 50)),
            description=item.get("description", ""),
        )
        for name, item in table.items()
    }


def model_specs() -> dict[str, ModelSpec]:
    table = _read("models.toml")["models"]
    return {
        name: ModelSpec(
            name=name,
            runner=item["runner"],
            description=item.get("description", ""),
            classifier=item.get("classifier"),
            blocks=tuple(item.get("blocks", [])),
            allow_external=bool(item.get("allow_external", False)),
            initialization=item.get("initialization"),
            command=tuple(item.get("command", [])),
            epochs=int(item.get("epochs", 150)),
            patience=int(item.get("patience", 15)),
        )
        for name, item in table.items()
    }
