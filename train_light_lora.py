# train_light_lora.py
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VideoFramesVFITrainDataset
from light_vfi_model import LightVFIUNet2D
from lora_utils import add_lora_to_lightvfi, iter_lora_parameters, get_lora_state_dict


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def train_light_lora(
    frames_root: str,
    base_ckpt: str,
    save_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    resize_h: int = 256,
    resize_w: int = 256,
    device: str = "cuda",
    num_workers: int = 4,
    in_frames: int = 4,
    base_channels: int = 32,
    r: int = 8,
    lora_alpha: int = 16,
    dropout: float = 0.0,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LightLoRA] 使用裝置: {device}")

    dataset = VideoFramesVFITrainDataset(
        frames_root=frames_root,
        num_input_frames=in_frames,
        resize_h=resize_h,
        resize_w=resize_w,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"[Dataset] 總 samples: {len(dataset)}")

    # 建 base 模型並載入權重
    model = LightVFIUNet2D(in_frames=in_frames, base_channels=base_channels)
    base_ckpt_path = Path(base_ckpt)
    if not base_ckpt_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")
    state = torch.load(base_ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    print(f"Loaded LightVFI base checkpoint from {base_ckpt_path}")

    # 注入 LoRA
    model = add_lora_to_lightvfi(model, r=r, lora_alpha=lora_alpha, dropout=dropout)

    # 冷凍所有 base 參數，只訓練 LoRA
    for p in model.parameters():
        p.requires_grad = False
    for p in iter_lora_parameters(model):
        p.requires_grad = True

    model.to(device)
    criterion = CharbonnierLoss()
    optimizer = torch.optim.Adam(iter_lora_parameters(model), lr=lr)

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"[LoRA] Epoch {epoch}/{epochs}", ncols=100)
        for batch in pbar:
            frames, target = batch
            frames = frames.to(device)   # [B,T,3,H,W]
            target = target.to(device)   # [B,3,H,W]

            b, t, c, h, w = frames.shape
            assert t == in_frames

            x = frames.view(b, t * c, h, w)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(loader)
        print(f"[LoRA Epoch {epoch}] avg_loss={avg_loss:.6f}")

        # 保存 LoRA-only state_dict
        lora_state = get_lora_state_dict(model)
        ckpt_lora = save_dir / f"light_vfi_lora_epoch{epoch}.pth"
        torch.save(lora_state, ckpt_lora)
        print(f"  -> Saved LoRA-only checkpoint to {ckpt_lora}")

    final_lora = save_dir / "light_vfi_lora_final.pth"
    torch.save(get_lora_state_dict(model), final_lora)
    print(f"[Done] Saved final LoRA checkpoint to {final_lora}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="weights_light_lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resize_h", type=int, default=256)
    parser.add_argument("--resize_w", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--in_frames", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    train_light_lora(
        frames_root=args.frames_root,
        base_ckpt=args.base_ckpt,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        device=args.device,
        num_workers=args.num_workers,
        in_frames=args.in_frames,
        base_channels=args.base_channels,
        r=args.r,
        lora_alpha=args.lora_alpha,
        dropout=args.dropout,
    )
