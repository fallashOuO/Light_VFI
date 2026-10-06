from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.losses import CharbonnierLoss
from src.utils import (
    Timer,
    count_params,
    ensure_dir,
    save_json,
    append_csv_row,
    file_size_mb,
    set_seed,
)
from src.data.dataset import VFIDataset, split_indices_by_video
from src.data.vimeo_dataset import VimeoTriplet


@dataclass
class TrainConfig:
    data_root: str
    base_ckpt: str = ""
    save_dir: str = "checkpoints/tmp"
    dataset_type: str = "custom"   # custom / vimeo
    split: str = "train"           # vimeo 用
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


def make_dataset(cfg: TrainConfig):
    if cfg.dataset_type != "custom":
        raise ValueError(
            "This four-frame model only supports "
            "the custom continuous-frame dataset."
        )

    return VFIDataset(
        cfg.data_root,
        resize=cfg.resize,
        in_frames=cfg.in_frames,
    )


def make_loaders(cfg: TrainConfig):
    ds = make_dataset(cfg)

    tr_idx, va_idx = split_indices_by_video(
        ds,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    tr = Subset(ds, tr_idx)
    va = Subset(ds, va_idx)

    tr_loader = DataLoader(
        tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    va_loader = DataLoader(
        va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return ds, tr_loader, va_loader


@torch.no_grad()
def run_val(model, loader, device) -> float:
    if loader is None:
        return -1.0

    model.eval()
    crit = CharbonnierLoss()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = crit(pred, y)
        total += loss.item()
        n += 1

    return total / max(1, n)


def train_loop(
    cfg: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_state_fn,
    method_name: str = "unknown",
):
    set_seed(cfg.seed)

    if cfg.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available!"

    device = torch.device(cfg.device)
    ensure_dir(cfg.save_dir)

    config_path = Path(cfg.save_dir) / "config.json"
    save_json(config_path, cfg)

    _, tr_loader, va_loader = make_loaders(cfg)

    model.to(device)
    crit = CharbonnierLoss()

    stats = count_params(model)
    print(f"[Train] device={device}")
    print(f"[Train] dataset_type={cfg.dataset_type}")
    print(f"[Train] trainable_params={stats['trainable']:,} / total={stats['total']:,}")

    best_metric = float("inf")
    best_epoch = -1
    total_train_time = 0.0
    last_epoch_time = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        with Timer() as t:
            pbar = tqdm(
                tr_loader,
                desc=f"Epoch {epoch}/{cfg.epochs}",
                ncols=120,
                leave=True,
            )
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss = crit(pred, y)
                loss.backward()
                optimizer.step()

                running += loss.item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                })

        train_loss = running / max(1, len(tr_loader))
        val_loss = run_val(model, va_loader, device=device)
        epoch_time = t.seconds
        total_train_time += epoch_time
        last_epoch_time = epoch_time

        if val_loss >= 0:
            print(f"[Epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f} time={epoch_time:.1f}s")
            metric = val_loss
        else:
            print(f"[Epoch {epoch}] train={train_loss:.6f} time={epoch_time:.1f}s")
            metric = train_loss

        ckpt = Path(cfg.save_dir) / f"epoch{epoch}.pth"
        torch.save(save_state_fn(model), ckpt)

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            best_path = Path(cfg.save_dir) / "best.pth"
            torch.save(save_state_fn(model), best_path)

    final_path = Path(cfg.save_dir) / "final.pth"
    torch.save(save_state_fn(model), final_path)

    best_path = Path(cfg.save_dir) / "best.pth"
    final_size_mb = file_size_mb(final_path)
    best_size_mb = file_size_mb(best_path)

    log_csv = Path("results/logs/train_log.csv")
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
    append_csv_row(log_csv, header, row)

    print(f"[Done] saved to {cfg.save_dir}")
    print(f"[Log] config saved to {config_path}")
    print(f"[Log] train log appended to {log_csv}")