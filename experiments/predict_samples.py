from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from src.data.dataset import VFIDataset
from src.models.light_vfi import LightVFI
from src.peft.lora import inject_lora, load_lora_state_dict
from src.utils import ensure_dir, set_seed


def load_tensor_dict(path: str | Path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu")

    if not isinstance(state, dict):
        raise TypeError(
            f"Checkpoint must contain a state dictionary: {path}"
        )

    return state


def build_model(args) -> torch.nn.Module:
    model = LightVFI(
        in_frames=4,
        base_ch=32,
    )

    base_state = load_tensor_dict(args.base_ckpt)
    model.load_state_dict(base_state, strict=True)

    print(f"[Load] Base checkpoint: {args.base_ckpt}")

    if args.method == "base":
        if args.ckpt:
            print(
                "[Warning] --ckpt is ignored because "
                "--method is set to base."
            )

    elif args.method == "fullft":
        if not args.ckpt:
            raise ValueError(
                "--ckpt is required when --method fullft is used."
            )

        fullft_state = load_tensor_dict(args.ckpt)
        model.load_state_dict(fullft_state, strict=True)

        print(f"[Load] Full FT checkpoint: {args.ckpt}")

    elif args.method == "bitfit":
        if not args.ckpt:
            raise ValueError(
                "--ckpt is required when --method bitfit is used."
            )

        bitfit_state = load_tensor_dict(args.ckpt)

        if not bitfit_state:
            raise RuntimeError("BitFit checkpoint is empty.")

        incompatible = model.load_state_dict(
            bitfit_state,
            strict=False,
        )

        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys in BitFit checkpoint: "
                f"{incompatible.unexpected_keys}"
            )

        print(
            f"[Load] BitFit checkpoint: {args.ckpt} "
            f"({len(bitfit_state)} tensors)"
        )

    elif args.method == "lora":
        if not args.ckpt:
            raise ValueError(
                "--ckpt is required when --method lora is used."
            )

        model = inject_lora(
            model,
            r=args.r,
            alpha=args.alpha,
            dropout=args.dropout,
        )

        lora_state = load_tensor_dict(args.ckpt)
        load_lora_state_dict(model, lora_state)

        print(f"[Load] LoRA checkpoint: {args.ckpt}")
        print(
            f"[LoRA] r={args.r}, "
            f"alpha={args.alpha}, "
            f"dropout={args.dropout}"
        )

    else:
        raise ValueError(f"Unsupported method: {args.method}")

    return model


def chw_tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    array = np.transpose(array, (1, 2, 0))
    array = np.rint(array * 255.0).astype(np.uint8)

    return array


def add_label(
    image: np.ndarray,
    label: str,
) -> np.ndarray:
    output = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, image.shape[1] / 700.0)
    thickness = max(1, int(round(image.shape[1] / 350.0)))

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    padding = 8

    cv2.rectangle(
        output,
        (0, 0),
        (
            min(output.shape[1] - 1, text_width + padding * 2),
            min(
                output.shape[0] - 1,
                text_height + baseline + padding * 2,
            ),
        ),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        output,
        label,
        (padding, text_height + padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )

    return output


def save_rgb_image(
    path: Path,
    rgb: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to save image: {path}")


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    prediction_float = prediction.astype(np.float32) / 255.0
    target_float = target.astype(np.float32) / 255.0

    psnr_value = peak_signal_noise_ratio(
        target_float,
        prediction_float,
        data_range=1.0,
    )

    ssim_value = structural_similarity(
        target_float,
        prediction_float,
        channel_axis=2,
        data_range=1.0,
    )

    return float(psnr_value), float(ssim_value)


def choose_indices(
    dataset_size: int,
    num_samples: int,
    seed: int,
) -> list[int]:
    if dataset_size <= 0:
        raise RuntimeError("Dataset is empty.")

    if num_samples <= 0:
        raise ValueError("--num_samples must be greater than zero.")

    count = min(dataset_size, num_samples)

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        dataset_size,
        size=count,
        replace=False,
    )

    return sorted(int(index) for index in indices)


def make_error_map(
    prediction: np.ndarray,
    target: np.ndarray,
    amplification: float,
) -> np.ndarray:
    prediction_float = prediction.astype(np.float32)
    target_float = target.astype(np.float32)

    error = np.abs(prediction_float - target_float)
    error = np.mean(error, axis=2)

    error = np.clip(
        error * amplification,
        0.0,
        255.0,
    ).astype(np.uint8)

    # 使用彩色熱圖顯示誤差：
    # 藍色代表誤差較小，紅色代表誤差較大。
    error_bgr = cv2.applyColorMap(
        error,
        cv2.COLORMAP_JET,
    )

    return cv2.cvtColor(
        error_bgr,
        cv2.COLOR_BGR2RGB,
    )


def write_metrics_csv(
    output_path: Path,
    rows: list[list],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "sample_index",
            "video",
            "target_frame",
            "method",
            "psnr",
            "ssim",
        ])

        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate visual prediction comparisons for Light_VFI."
        )
    )

    parser.add_argument(
        "--data_root",
        required=True,
        help="Dataset root containing video subdirectories.",
    )
    parser.add_argument(
        "--base_ckpt",
        required=True,
        help="Path to the base-model checkpoint.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["base", "fullft", "bitfit", "lora"],
    )
    parser.add_argument(
        "--ckpt",
        default="",
        help=(
            "Fine-tuned checkpoint. Required for "
            "fullft, bitfit, and lora."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="results/predictions",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--error_amplification",
        type=float,
        default=4.0,
        help="Amplification applied to the error-map intensity.",
    )

    parser.add_argument(
        "--r",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available. "
            "Use --device cpu or install CUDA-enabled PyTorch."
        )

    device = torch.device(args.device)

    dataset = VFIDataset(
        args.data_root,
        resize=args.resize,
        in_frames=4,
    )

    model = build_model(args)
    model.to(device)
    model.eval()

    output_dir = (
        Path(args.output_dir)
        / args.method
    )
    ensure_dir(output_dir)

    sample_indices = choose_indices(
        dataset_size=len(dataset),
        num_samples=args.num_samples,
        seed=args.seed,
    )

    print(f"[Predict] device={device}")
    print(f"[Predict] dataset_samples={len(dataset)}")
    print(f"[Predict] selected_samples={len(sample_indices)}")
    print(f"[Predict] output_dir={output_dir}")

    metrics_rows = []
    psnr_values = []
    ssim_values = []

    for output_number, sample_index in enumerate(
        sample_indices,
        start=1,
    ):
        input_tensor, target_tensor = dataset[sample_index]

        batched_input = (
            input_tensor
            .unsqueeze(0)
            .to(device, non_blocking=True)
        )

        prediction_tensor = (
            model(batched_input)[0]
            .clamp(0.0, 1.0)
        )

        input_frames = [
            chw_tensor_to_rgb_uint8(
                input_tensor[index * 3:(index + 1) * 3]
            )
            for index in range(4)
        ]

        prediction = chw_tensor_to_rgb_uint8(
            prediction_tensor
        )
        target = chw_tensor_to_rgb_uint8(
            target_tensor
        )

        psnr_value, ssim_value = calculate_metrics(
            prediction,
            target,
        )

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

        error_map = make_error_map(
            prediction,
            target,
            amplification=args.error_amplification,
        )

        video_dir, frame_files, center_index = (
            dataset.index[sample_index]
        )

        target_frame_name = frame_files[center_index].name
        video_name = video_dir.name

        sample_name = (
            f"{output_number:03d}_"
            f"idx{sample_index:06d}_"
            f"{video_name}_"
            f"{Path(target_frame_name).stem}"
        )

        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        save_rgb_image(
            sample_dir / "input_i_minus_2.png",
            input_frames[0],
        )
        save_rgb_image(
            sample_dir / "input_i_minus_1.png",
            input_frames[1],
        )
        save_rgb_image(
            sample_dir / "input_i_plus_1.png",
            input_frames[2],
        )
        save_rgb_image(
            sample_dir / "input_i_plus_2.png",
            input_frames[3],
        )
        save_rgb_image(
            sample_dir / "prediction.png",
            prediction,
        )
        save_rgb_image(
            sample_dir / "ground_truth.png",
            target,
        )
        save_rgb_image(
            sample_dir / "error_map.png",
            error_map,
        )

        panels = [
            add_label(input_frames[0], "Input i-2"),
            add_label(input_frames[1], "Input i-1"),
            add_label(prediction, "Prediction"),
            add_label(
                target,
                f"Ground Truth | PSNR {psnr_value:.2f}",
            ),
            add_label(input_frames[2], "Input i+1"),
            add_label(input_frames[3], "Input i+2"),
            add_label(
                error_map,
                f"Error x{args.error_amplification:g}",
            ),
        ]

        comparison = np.concatenate(
            panels,
            axis=1,
        )

        save_rgb_image(
            sample_dir / "comparison.png",
            comparison,
        )

        metrics_rows.append([
            sample_index,
            video_name,
            target_frame_name,
            args.method,
            f"{psnr_value:.6f}",
            f"{ssim_value:.6f}",
        ])

        print(
            f"[{output_number}/{len(sample_indices)}] "
            f"video={video_name} "
            f"target={target_frame_name} "
            f"PSNR={psnr_value:.4f} "
            f"SSIM={ssim_value:.4f}"
        )

    metrics_path = output_dir / "metrics.csv"

    write_metrics_csv(
        metrics_path,
        metrics_rows,
    )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    print()
    print(
        f"[Result] Mean PSNR={mean_psnr:.4f}"
    )
    print(
        f"[Result] Mean SSIM={mean_ssim:.4f}"
    )
    print(
        f"[Done] Predictions saved to: {output_dir}"
    )
    print(
        f"[Done] Metrics saved to: {metrics_path}"
    )


if __name__ == "__main__":
    main()