import argparse
from pathlib import Path
from typing import Any

import torch

from src.models.light_vfi import LightVFI
from src.utils import set_seed
from experiments._train_common import TrainConfig, train_loop


def positive_int(value: str) -> int:
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, got {value}"
        )

    return parsed_value


def positive_float(value: str) -> float:
    parsed_value = float(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive number, got {value}"
        )

    return parsed_value


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    支援以下常見 checkpoint 格式：

    1. model.state_dict() 直接儲存
    2. {"state_dict": model.state_dict()}
    3. {"model_state_dict": model.state_dict()}
    4. {"model": model.state_dict()}
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format: "
            "checkpoint must be a dictionary."
        )

    possible_keys = (
        "state_dict",
        "model_state_dict",
        "model",
    )

    for key in possible_keys:
        value = checkpoint.get(key)

        if isinstance(value, dict):
            checkpoint = value
            break

    if not checkpoint:
        raise ValueError("The checkpoint state dictionary is empty.")

    if not all(
        isinstance(key, str)
        for key in checkpoint.keys()
    ):
        raise TypeError(
            "Invalid checkpoint: all parameter names must be strings."
        )

    if not all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        raise TypeError(
            "Invalid checkpoint: the selected state dictionary "
            "contains non-tensor values."
        )

    # 支援由 DataParallel 儲存、參數名稱含有 module. 前綴的 checkpoint
    if all(
        key.startswith("module.")
        for key in checkpoint.keys()
    ):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    return checkpoint


def load_base_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Base checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(f"Loaded base checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning for LightVFI."
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Path to the custom training dataset.",
    )
    parser.add_argument(
        "--base_ckpt",
        type=Path,
        required=True,
        help="Path to the pretrained base-model checkpoint.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("checkpoints/fullft/domainX"),
        help="Directory used to save full fine-tuning checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=5,
    )
    parser.add_argument(
        "--batch_size",
        type=positive_int,
        default=8,
    )
    parser.add_argument(
        "--lr",
        type=positive_float,
        default=1e-4,
    )
    parser.add_argument(
        "--resize",
        type=positive_int,
        default=256,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise NotADirectoryError(
            f"Training data directory not found: {args.data_root}"
        )

    if args.num_workers < 0:
        raise ValueError(
            f"num_workers must be greater than or equal to 0, "
            f"got {args.num_workers}"
        )

    set_seed(args.seed)

    cfg = TrainConfig(
        data_root=str(args.data_root),
        base_ckpt=str(args.base_ckpt),
        save_dir=str(args.save_dir),
        dataset_type="custom",
        split="train",
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

    load_base_checkpoint(
        model=model,
        checkpoint_path=args.base_ckpt,
    )

    # Full Fine-Tuning：解凍模型的全部參數
    for parameter in model.parameters():
        parameter.requires_grad = True

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    if trainable_parameter_count == 0:
        raise RuntimeError(
            "No trainable parameters were found."
        )

    if trainable_parameter_count != total_parameter_count:
        raise RuntimeError(
            "Full fine-tuning requires every model parameter "
            "to be trainable."
        )

    print(f"Total parameters: {total_parameter_count:,}")
    print(f"Trainable parameters: {trainable_parameter_count:,}")

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=cfg.lr,
    )

    train_loop(
        cfg,
        model,
        optimizer,
        save_state_fn=lambda current_model: current_model.state_dict(),
        method_name="fullft",
    )


if __name__ == "__main__":
    main()