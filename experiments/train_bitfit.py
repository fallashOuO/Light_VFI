import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.models.light_vfi import LightVFI
from experiments._train_common import TrainConfig, train_loop


def enable_bias_only(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            m.bias.requires_grad = True

def get_bitfit_state_dict(model: nn.Module):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--save_dir", default="checkpoints/bitfit/domainX")
    ap.add_argument("--dataset_type", default="custom", choices=["custom", "vimeo"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        base_ckpt=args.base_ckpt,
        save_dir=args.save_dir,
        dataset_type=args.dataset_type,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resize=args.resize,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = LightVFI(in_frames=cfg.in_frames, base_ch=cfg.base_ch)

    state = torch.load(Path(cfg.base_ckpt), map_location="cpu")
    model.load_state_dict(state, strict=True)
    print(f"Loaded base checkpoint: {cfg.base_ckpt}")

    enable_bias_only(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg.lr)

    train_loop(
        cfg,
        model,
        opt,
        save_state_fn=get_bitfit_state_dict,
        method_name="bitfit",
    )


if __name__ == "__main__":
    main()