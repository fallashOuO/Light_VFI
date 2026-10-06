# experiments/train_lora.py
import argparse
from pathlib import Path

import torch

from src.models.light_vfi import LightVFI
from src.peft.lora import (
    inject_lora,
    lora_parameters,
    get_lora_state_dict,
)
from src.utils import set_seed
from experiments._train_common import TrainConfig, train_loop


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument(
        "--save_dir",
        default="checkpoints/lora/domainX",
    )
    ap.add_argument(
        "--dataset_type",
        default="custom",
        choices=["custom"],
    )
    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)

    args = ap.parse_args()

    # 必須在建立模型與注入 LoRA 以前設定
    set_seed(args.seed)

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

    model = LightVFI(
        in_frames=cfg.in_frames,
        base_ch=cfg.base_ch,
    )

    state = torch.load(
        Path(cfg.base_ckpt),
        map_location="cpu",
    )
    model.load_state_dict(state, strict=True)
    print(f"Loaded base checkpoint: {cfg.base_ckpt}")

    model = inject_lora(
        model,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
    )

    for p in model.parameters():
        p.requires_grad = False

    for p in lora_parameters(model):
        p.requires_grad = True

    trainable_params = list(lora_parameters(model))

    if not trainable_params:
        raise RuntimeError(
            "No LoRA parameters were found after injection."
        )

    opt = torch.optim.Adam(
        trainable_params,
        lr=cfg.lr,
    )

    train_loop(
        cfg,
        model,
        opt,
        save_state_fn=get_lora_state_dict,
        method_name="lora",
    )


if __name__ == "__main__":
    main()