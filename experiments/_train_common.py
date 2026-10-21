from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import VFIDataset, split_indices_by_video
from src.losses import CharbonnierLoss
from src.utils import (
    Timer,
    append_csv_row,
    count_params,
    ensure_dir,
    file_size_mb,
    save_json,
    set_seed,
)


@dataclass
class TrainConfig:
    data_root: str
    base_ckpt: str = ""
    save_dir: str = "checkpoints/tmp"
    dataset_type: str = "custom"
    split: str = "train"
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    resize: int = 256
    val_ratio: float = 0.05
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    in_frames: int = 4
    base_ch: int = 32


def validate_config(cfg: TrainConfig) -> None:
    data_root = Path(cfg.data_root)

    if not data_root.is_dir():
        raise NotADirectoryError(
            f"Training data directory not found: {data_root}"
        )

    if cfg.dataset_type != "custom":
        raise ValueError(
            "Only dataset_type='custom' is supported."
        )

    if cfg.split != "train":
        raise ValueError(
            "Training must use split='train'. "
            "The test split must not be used for training."
        )

    if cfg.epochs <= 0:
        raise ValueError(
            f"epochs must be greater than 0, got {cfg.epochs}"
        )

    if cfg.batch_size <= 0:
        raise ValueError(
            f"batch_size must be greater than 0, "
            f"got {cfg.batch_size}"
        )

    if cfg.lr <= 0:
        raise ValueError(
            f"lr must be greater than 0, got {cfg.lr}"
        )

    if cfg.resize <= 0:
        raise ValueError(
            f"resize must be greater than 0, got {cfg.resize}"
        )

    if not 0.0 <= cfg.val_ratio < 1.0:
        raise ValueError(
            "val_ratio must satisfy 0 <= val_ratio < 1, "
            f"got {cfg.val_ratio}"
        )

    if cfg.num_workers < 0:
        raise ValueError(
            "num_workers must be greater than or equal to 0, "
            f"got {cfg.num_workers}"
        )

    if cfg.in_frames <= 0:
        raise ValueError(
            f"in_frames must be greater than 0, "
            f"got {cfg.in_frames}"
        )

    if cfg.base_ch <= 0:
        raise ValueError(
            f"base_ch must be greater than 0, "
            f"got {cfg.base_ch}"
        )


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device_name}' was requested, "
            "but CUDA is not available."
        )

    return device


def make_dataset(cfg: TrainConfig) -> VFIDataset:
    if cfg.dataset_type != "custom":
        raise ValueError(
            "This four-frame model only supports "
            "the custom continuous-frame dataset."
        )

    dataset = VFIDataset(
        cfg.data_root,
        resize=cfg.resize,
        in_frames=cfg.in_frames,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training samples were found in: {cfg.data_root}"
        )

    return dataset


def make_loaders(
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[
    VFIDataset,
    DataLoader,
    Optional[DataLoader],
]:
    dataset = make_dataset(cfg)

    train_indices, val_indices = split_indices_by_video(
        dataset,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    if len(train_indices) == 0:
        raise RuntimeError(
            "The training split is empty. Check the dataset "
            "and val_ratio settings."
        )

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    val_dataset = Subset(
        dataset,
        val_indices,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loader: Optional[DataLoader]

    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
    else:
        val_loader = None
        print(
            "[Warning] Validation split is empty. "
            "Training loss will be used to select best.pth."
        )

    print(f"[Data] total_samples={len(dataset):,}")
    print(f"[Data] train_samples={len(train_dataset):,}")
    print(f"[Data] val_samples={len(val_dataset):,}")

    return dataset, train_loader, val_loader


@torch.no_grad()
def run_val(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    criterion: torch.nn.Module,
) -> Optional[float]:
    if loader is None:
        return None

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(
            device,
            non_blocking=device.type == "cuda",
        )
        y = y.to(
            device,
            non_blocking=device.type == "cuda",
        )

        prediction = model(x)
        loss = criterion(prediction, y)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "A non-finite validation loss was detected: "
                f"{loss.item()}"
            )

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return None

    val_loss = total_loss / total_samples

    if not math.isfinite(val_loss):
        raise FloatingPointError(
            f"Invalid validation loss: {val_loss}"
        )

    return val_loss


def train_loop(
    cfg: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_state_fn: Callable[
        [torch.nn.Module],
        dict[str, torch.Tensor],
    ],
    method_name: str = "unknown",
) -> None:
    validate_config(cfg)
    set_seed(cfg.seed)

    device = resolve_device(cfg.device)

    save_dir = Path(cfg.save_dir)
    ensure_dir(save_dir)

    config_path = save_dir / "config.json"

    # 使用一般 dictionary 儲存，避免 JSON 函式不支援 dataclass。
    save_json(
        config_path,
        asdict(cfg),
    )

    _, train_loader, val_loader = make_loaders(
        cfg,
        device=device,
    )

    model.to(device)

    criterion = CharbonnierLoss()

    stats = count_params(model)

    if stats["trainable"] <= 0:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    print(f"[Train] device={device}")
    print(f"[Train] method={method_name}")
    print(f"[Train] dataset_type={cfg.dataset_type}")
    print(
        f"[Train] trainable_params="
        f"{stats['trainable']:,} / "
        f"total={stats['total']:,}"
    )

    best_metric = float("inf")
    best_epoch = -1

    total_train_time = 0.0
    last_epoch_time = 0.0

    best_path = save_dir / "best.pth"

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        with Timer() as timer:
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{cfg.epochs}",
                ncols=120,
                leave=True,
            )

            for x, y in progress_bar:
                x = x.to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                y = y.to(
                    device,
                    non_blocking=device.type == "cuda",
                )

                optimizer.zero_grad(set_to_none=True)

                prediction = model(x)
                loss = criterion(prediction, y)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "A non-finite training loss was detected "
                        f"at epoch {epoch}: {loss.item()}"
                    )

                loss.backward()
                optimizer.step()

                batch_size = x.size(0)

                total_train_loss += loss.item() * batch_size
                total_train_samples += batch_size

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": (
                            f"{optimizer.param_groups[0]['lr']:.1e}"
                        ),
                    }
                )

        if total_train_samples == 0:
            raise RuntimeError(
                "No samples were processed during training."
            )

        train_loss = (
            total_train_loss / total_train_samples
        )

        val_loss = run_val(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
        )

        epoch_time = timer.seconds
        total_train_time += epoch_time
        last_epoch_time = epoch_time

        if val_loss is not None:
            print(
                f"[Epoch {epoch}] "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"time={epoch_time:.1f}s"
            )

            metric = val_loss
        else:
            print(
                f"[Epoch {epoch}] "
                f"train={train_loss:.6f} "
                f"time={epoch_time:.1f}s"
            )

            metric = train_loss

        if not math.isfinite(metric):
            raise FloatingPointError(
                f"Invalid checkpoint metric: {metric}"
            )

        epoch_path = save_dir / f"epoch{epoch}.pth"

        torch.save(
            save_state_fn(model),
            epoch_path,
        )

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch

            torch.save(
                save_state_fn(model),
                best_path,
            )

            print(
                f"[Checkpoint] New best model: "
                f"epoch={best_epoch}, "
                f"metric={best_metric:.6f}"
            )

    final_path = save_dir / "final.pth"

    torch.save(
        save_state_fn(model),
        final_path,
    )

    if not best_path.is_file():
        raise RuntimeError(
            "Training finished without creating best.pth."
        )

    final_size_mb = file_size_mb(final_path)
    best_size_mb = file_size_mb(best_path)

    log_csv = Path("results/logs/train_log.csv")
    ensure_dir(log_csv.parent)

    header = [
        "method",
        "dataset_type",
        "split",
        "data_root",
        "save_dir",
        "base_ckpt",
        "epochs",
        "batch_size",
        "lr",
        "resize",
        "trainable_params",
        "total_params",
        "best_metric",
        "best_epoch",
        "total_train_time_sec",
        "last_epoch_time_sec",
        "best_ckpt_size_mb",
        "final_ckpt_size_mb",
    ]

    row = [
        method_name,
        cfg.dataset_type,
        cfg.split,
        cfg.data_root,
        cfg.save_dir,
        cfg.base_ckpt,
        cfg.epochs,
        cfg.batch_size,
        cfg.lr,
        cfg.resize,
        stats["trainable"],
        stats["total"],
        f"{best_metric:.6f}",
        best_epoch,
        f"{total_train_time:.2f}",
        f"{last_epoch_time:.2f}",
        f"{best_size_mb:.3f}",
        f"{final_size_mb:.3f}",
    ]

    append_csv_row(
        log_csv,
        header,
        row,
    )

    print(f"[Done] saved to {save_dir}")
    print(f"[Log] config saved to {config_path}")
    print(f"[Log] train log appended to {log_csv}")