import argparse
from pathlib import Path
import csv

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.models.light_vfi import LightVFI
from src.data.dataset import VFIDataset
from src.peft.lora import inject_lora, load_lora_state_dict
from src.utils import file_size_mb


def make_dataset(
    data_root,
    dataset_type="custom",
    split="test",
    resize=256,
):
    if dataset_type != "custom":
        raise ValueError(
            "This four-frame model only supports "
            "the custom continuous-frame dataset."
        )

    return VFIDataset(
        data_root,
        resize=resize,
        in_frames=4,
    )


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    psnr_list = []
    ssim_list = []

    pbar = tqdm(loader, desc="Evaluating", ncols=120)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x).clamp(0, 1)

        pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
        y_np = y.permute(0, 2, 3, 1).cpu().numpy()

        for i in range(pred_np.shape[0]):
            p = pred_np[i]
            t = y_np[i]
            psnr_list.append(psnr(t, p, data_range=1.0))
            ssim_list.append(ssim(t, p, channel_axis=2, data_range=1.0))

        if psnr_list:
            pbar.set_postfix({
                "PSNR": f"{np.mean(psnr_list):.3f}",
                "SSIM": f"{np.mean(ssim_list):.4f}",
            })

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset_type", default="custom", choices=["custom"])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--method", choices=["base", "fullft", "bitfit", "lora"], required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", default="results/metrics/metrics.csv")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = make_dataset(args.data_root, args.dataset_type, args.split, args.resize)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = LightVFI(in_frames=4, base_ch=32)
    model.load_state_dict(torch.load(Path(args.base_ckpt), map_location="cpu"), strict=True)

    actual_ckpt = args.base_ckpt

    
    if args.method == "fullft":
        if not args.ckpt:
            raise ValueError("--ckpt required for fullft")

        fullft_state = torch.load(
            Path(args.ckpt),
            map_location="cpu",
        )
        model.load_state_dict(fullft_state, strict=True)
        actual_ckpt = args.ckpt

    elif args.method == "bitfit":
        if not args.ckpt:
            raise ValueError("--ckpt required for bitfit")

        bitfit_state = torch.load(
            Path(args.ckpt),
            map_location="cpu",
        )

        incompatible = model.load_state_dict(
            bitfit_state,
            strict=False,
        )

        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys in BitFit checkpoint: "
                f"{incompatible.unexpected_keys}"
            )

        if not bitfit_state:
            raise RuntimeError("BitFit checkpoint is empty.")

        print(
            f"Loaded {len(bitfit_state)} BitFit parameters "
            f"from: {args.ckpt}"
        )
        actual_ckpt = args.ckpt

    elif args.method == "lora":
        if not args.ckpt:
            raise ValueError("--ckpt required for lora")
        model = inject_lora(model, r=args.r, alpha=args.alpha, dropout=args.dropout)
        lora_sd = torch.load(Path(args.ckpt), map_location="cpu")
        load_lora_state_dict(model, lora_sd)
        actual_ckpt = args.ckpt

    model.to(device)

    p, s = eval_model(model, loader, device)
    ckpt_size = file_size_mb(actual_ckpt)

    print(f"[Eval] method={args.method} PSNR={p:.4f} SSIM={s:.4f}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "dataset_type",
                "data_root",
                "split",
                "method",
                "ckpt",
                "ckpt_size_mb",
                "psnr",
                "ssim",
            ])
        writer.writerow([
            args.dataset_type,
            args.data_root,
            args.split,
            args.method,
            actual_ckpt,
            f"{ckpt_size:.3f}",
            f"{p:.6f}",
            f"{s:.6f}",
        ])


if __name__ == "__main__":
    main()