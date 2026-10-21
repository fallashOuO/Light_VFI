import csv
import json
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
) -> None:
    if not isinstance(seed, int):
        raise TypeError(
            f"seed must be an integer, got {type(seed).__name__}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 加強相同硬體與相同環境下的實驗重現性。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def count_params(
    model: torch.nn.Module,
) -> dict[str, int]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be an instance of torch.nn.Module."
        )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


class Timer:
    def __init__(self) -> None:
        self._start_time: float | None = None
        self._elapsed_time: float | None = None

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        self._elapsed_time = None
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        if self._start_time is None:
            raise RuntimeError(
                "Timer was not started correctly."
            )

        self._elapsed_time = (
            time.perf_counter() - self._start_time
        )

        # 不攔截 with 區塊內發生的例外。
        return False

    @property
    def seconds(self) -> float:
        if self._start_time is None:
            raise RuntimeError(
                "Timer has not been started."
            )

        if self._elapsed_time is not None:
            return self._elapsed_time

        return time.perf_counter() - self._start_time


def ensure_dir(
    path: str | Path,
) -> Path:
    directory = Path(path)

    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(
            f"Path exists but is not a directory: {directory}"
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return value.detach().cpu().tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def save_json(
    path: str | Path,
    obj: Any,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if is_dataclass(obj):
        obj = asdict(obj)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            obj,
            file,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

        # 讓檔案最後保留換行。
        file.write("\n")


def append_csv_row(
    path: str | Path,
    header: list[str],
    row: list[Any],
) -> None:
    output_path = Path(path)

    if not header:
        raise ValueError(
            "CSV header must not be empty."
        )

    if len(header) != len(row):
        raise ValueError(
            "CSV header and row have different lengths: "
            f"header={len(header)}, row={len(row)}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    if not write_header:
        with output_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.reader(file)
            existing_header = next(
                reader,
                None,
            )

        if existing_header != header:
            raise RuntimeError(
                "The existing CSV header does not match "
                f"the requested header.\n"
                f"File: {output_path}\n"
                f"Existing: {existing_header}\n"
                f"Requested: {header}"
            )

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow(header)

        writer.writerow(row)


def file_size_mb(
    path: str | Path,
) -> float:
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return file_path.stat().st_size / (
        1024 * 1024
    )