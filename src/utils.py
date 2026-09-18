import os
import random
import time
import json
import csv
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class Timer:
    def __init__(self):
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    @property
    def seconds(self) -> float:
        return time.time() - self.t0


def ensure_dir(path: str | Path):
    os.makedirs(path, exist_ok=True)


def save_json(path: str | Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(obj):
        obj = asdict(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv_row(path: str | Path, header: list[str], row: list[Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def file_size_mb(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return -1.0
    return path.stat().st_size / (1024 * 1024)