import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VFIDataset
from src.models.light_vfi import LightVFI
from src.peft.lora import inject_lora, load_lora_state_dict
from src.utils import file_size_mb


def positive_int(value: str) -> int:
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, got {value}"
        )

    return parsed_value


def non_negative_float(value: str) -> float:
    parsed_value = float(value)

    if parsed_value < 0:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative number, got {value}"
        )

    return parsed_value


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format: "
            "checkpoint must be a dictionary."
        )

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
    ):
        value = checkpoint.get(key)

        if isinstance(value, dict):
            checkpoint = value
            break

    if not checkpoint:
        raise ValueError(
            "The checkpoint state dictionary is empty."
        )

    if not all(
        isinstance(name, str)
        for name in checkpoint.keys()
    ):
        raise TypeError(
            "Invalid checkpoint: parameter names "
            "must be strings."
        )

    if not all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        raise TypeError(
            "Invalid checkpoint: the selected state "
            "dictionary contains non-tensor values."
        )

    if all(
        name.startswith("module.")
        for name in checkpoint.keys()
    ):
        checkpoint = {
            name.removeprefix("module."): value
            for name, value in checkpoint.items()
        }

    return checkpoint


def load_checkpoint_file(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    return extract_state_dict(checkpoint)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device_name}' was requested, "
            "but CUDA is not available."
        )

    return device


def make_dataset(
    data_root: Path,
    resize: int,
) -> VFIDataset:
    if not data_root.is_dir():
        raise NotADirectoryError(
            f"Evaluation data directory not found: {data_root}"
        )

    dataset = VFIDataset(
        str(data_root),
        resize=resize,
        in_frames=4,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No evaluation samples were found in: {data_root}"
        )

    return dataset


def load_base_model(
    base_checkpoint_path: Path,
) -> LightVFI:
    model = LightVFI(
        in_frames=4,
        base_ch=32,
    )

    base_state_dict = load_checkpoint_file(
        base_checkpoint_path
    )

    model.load_state_dict(
        base_state_dict,
        strict=True,
    )

    print(
        f"Loaded Base Model checkpoint: "
        f"{base_checkpoint_path}"
    )

    return model


def load_bitfit_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    bitfit_state_dict = load_checkpoint_file(
        checkpoint_path
    )

    invalid_names = [
        name
        for name in bitfit_state_dict
        if not name.endswith(".bias")
    ]

    if invalid_names:
        raise RuntimeError(
            "BitFit checkpoint contains non-bias parameters: "
            + ", ".join(invalid_names)
        )

    model_parameter_names = {
        name
        for name, _ in model.named_parameters()
    }

    unknown_names = [
        name
        for name in bitfit_state_dict
        if name not in model_parameter_names
    ]

    if unknown_names:
        raise RuntimeError(
            "BitFit checkpoint contains unknown parameters: "
            + ", ".join(unknown_names)
        )

    incompatible = model.load_state_dict(
        bitfit_state_dict,
        strict=False,
    )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys in BitFit checkpoint: "
            f"{incompatible.unexpected_keys}"
        )

    print(
        f"Loaded {len(bitfit_state_dict)} BitFit parameters "
        f"from: {checkpoint_path}"
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    psnr_values: list[float] = []
    ssim_values: list[float] = []

    progress_bar = tqdm(
        loader,
        desc="Evaluating",
        ncols=120,
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

        prediction = model(x).clamp(0.0, 1.0)

        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                "The model produced NaN or Inf values."
            )

        prediction_np = (
            prediction
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        target_np = (
            y
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        for index in range(prediction_np.shape[0]):
            predicted_image = prediction_np[index]
            target_image = target_np[index]

            psnr_value = peak_signal_noise_ratio(
                target_image,
                predicted_image,
                data_range=1.0,
            )

            ssim_value = structural_similarity(
                target_image,
                predicted_image,
                channel_axis=2,
                data_range=1.0,
            )

            psnr_values.append(float(psnr_value))
            ssim_values.append(float(ssim_value))

        if psnr_values:
            progress_bar.set_postfix(
                {
                    "PSNR": f"{np.mean(psnr_values):.3f}",
                    "SSIM": f"{np.mean(ssim_values):.4f}",
                }
            )

    if not psnr_values or not ssim_values:
        raise RuntimeError(
            "No samples were evaluated."
        )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    if math.isnan(mean_psnr) or math.isnan(mean_ssim):
        raise FloatingPointError(
            "Evaluation produced a NaN metric."
        )

    return mean_psnr, mean_ssim


def calculate_checkpoint_sizes(
    method: str,
    base_checkpoint_path: Path,
    method_checkpoint_path: Path | None,
) -> tuple[float, float, float]:
    base_size_mb = file_size_mb(
        base_checkpoint_path
    )

    if method == "base":
        method_size_mb = 0.0
        total_required_size_mb = base_size_mb

    elif method == "fullft":
        if method_checkpoint_path is None:
            raise ValueError(
                "FullFT checkpoint path is missing."
            )

        method_size_mb = file_size_mb(
            method_checkpoint_path
        )

        # FullFT checkpoint本身就是完整模型，
        # 推論時不需要另外保存 Base checkpoint。
        total_required_size_mb = method_size_mb

    else:
        if method_checkpoint_path is None:
            raise ValueError(
                f"{method} checkpoint path is missing."
            )

        method_size_mb = file_size_mb(
            method_checkpoint_path
        )

        # LoRA 和 BitFit 推論時都需要：
        # Base Model + 額外訓練參數。
        total_required_size_mb = (
            base_size_mb + method_size_mb
        )

    return (
        base_size_mb,
        method_size_mb,
        total_required_size_mb,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LightVFI models."
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help=(
            "Path to an independent test dataset. "
            "Do not use the training data directory."
        ),
    )
    parser.add_argument(
        "--base_ckpt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=[
            "base",
            "fullft",
            "bitfit",
            "lora",
        ],
        required=True,
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=(
            "Required for fullft, bitfit and lora; "
            "not used for base."
        ),
    )
    parser.add_argument(
        "--resize",
        type=positive_int,
        default=256,
    )
    parser.add_argument(
        "--batch_size",
        type=positive_int,
        default=8,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/metrics/metrics.csv"),
    )
    parser.add_argument(
        "--r",
        type=positive_int,
        default=8,
    )
    parser.add_argument(
        "--alpha",
        type=positive_int,
        default=16,
    )
    parser.add_argument(
        "--dropout",
        type=non_negative_float,
        default=0.0,
    )

    args = parser.parse_args()

    if args.resize < 7:
        raise ValueError(
            "resize must be at least 7 for SSIM calculation."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num_workers must be greater than or equal to 0."
        )

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(
            "dropout must satisfy 0 <= dropout < 1."
        )

    if args.method == "base" and args.ckpt is not None:
        raise ValueError(
            "--ckpt must not be provided when method='base'."
        )

    if args.method != "base" and args.ckpt is None:
        raise ValueError(
            f"--ckpt is required when method='{args.method}'."
        )

    device = resolve_device(args.device)

    dataset = make_dataset(
        data_root=args.data_root,
        resize=args.resize,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = load_base_model(
        args.base_ckpt
    )

    if args.method == "fullft":
        fullft_state_dict = load_checkpoint_file(
            args.ckpt
        )

        model.load_state_dict(
            fullft_state_dict,
            strict=True,
        )

        print(
            f"Loaded FullFT checkpoint: {args.ckpt}"
        )

    elif args.method == "bitfit":
        load_bitfit_checkpoint(
            model=model,
            checkpoint_path=args.ckpt,
        )

    elif args.method == "lora":
        model = inject_lora(
            model,
            r=args.r,
            alpha=args.alpha,
            dropout=args.dropout,
        )

        lora_state_dict = load_checkpoint_file(
            args.ckpt
        )

        load_lora_state_dict(
            model,
            lora_state_dict,
        )

        print(
            f"Loaded LoRA checkpoint: {args.ckpt}"
        )

    model.to(device)

    mean_psnr, mean_ssim = evaluate_model(
        model=model,
        loader=loader,
        device=device,
    )

    (
        base_size_mb,
        method_size_mb,
        total_required_size_mb,
    ) = calculate_checkpoint_sizes(
        method=args.method,
        base_checkpoint_path=args.base_ckpt,
        method_checkpoint_path=args.ckpt,
    )

    print(
        f"[Eval] method={args.method} "
        f"PSNR={mean_psnr:.4f} "
        f"SSIM={mean_ssim:.4f}"
    )

    print(
        f"[Size] base={base_size_mb:.3f} MB "
        f"method={method_size_mb:.3f} MB "
        f"total_required={total_required_size_mb:.3f} MB"
    )

    output_path = args.out_csv
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow(
                [
                    "dataset_type",
                    "data_root",
                    "split",
                    "method",
                    "base_ckpt",
                    "method_ckpt",
                    "base_ckpt_size_mb",
                    "method_ckpt_size_mb",
                    "total_required_size_mb",
                    "psnr",
                    "ssim",
                ]
            )

        writer.writerow(
            [
                "custom",
                str(args.data_root),
                "test",
                args.method,
                str(args.base_ckpt),
                (
                    str(args.ckpt)
                    if args.ckpt is not None
                    else ""
                ),
                f"{base_size_mb:.3f}",
                f"{method_size_mb:.3f}",
                f"{total_required_size_mb:.3f}",
                f"{mean_psnr:.6f}",
                f"{mean_ssim:.6f}",
            ]
        )

    print(
        f"[Done] metrics appended to: {output_path}"
    )


if __name__ == "__main__":
    main()