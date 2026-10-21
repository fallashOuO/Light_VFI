import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from experiments._train_common import TrainConfig, train_loop
from src.models.light_vfi import LightVFI
from src.utils import set_seed


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


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """
    支援常見的 Base Model checkpoint 格式：

    1. model.state_dict()
    2. {"state_dict": model.state_dict()}
    3. {"model_state_dict": model.state_dict()}
    4. {"model": model.state_dict()}
    """
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
            "Invalid checkpoint: all parameter names "
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

    # 支援由 DataParallel 儲存的 checkpoint
    if all(
        name.startswith("module.")
        for name in checkpoint.keys()
    ):
        checkpoint = {
            name.removeprefix("module."): value
            for name, value in checkpoint.items()
        }

    return checkpoint


def load_base_checkpoint(
    model: nn.Module,
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


def enable_bias_only(
    model: nn.Module,
) -> list[str]:
    """
    BitFit：
    凍結全部參數，只解凍名稱以 .bias 結尾的參數。
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_names: list[str] = []

    for name, parameter in model.named_parameters():
        if name.endswith(".bias"):
            parameter.requires_grad = True
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError(
            "No bias parameters were found in the model."
        )

    return trainable_names


def get_bitfit_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """
    只儲存 BitFit 訓練過的 bias 參數，
    不重複儲存整個 Base Model。
    """
    state_dict = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if not state_dict:
        raise RuntimeError(
            "No trainable BitFit parameters were found "
            "when saving the checkpoint."
        )

    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BitFit training for LightVFI."
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
        help="Path to the pretrained Base Model checkpoint.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("checkpoints/bitfit/domainX"),
        help="Directory used to save BitFit checkpoints.",
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
        default=5e-4,
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
            f"Training data directory not found: "
            f"{args.data_root}"
        )

    if args.num_workers < 0:
        raise ValueError(
            "num_workers must be greater than or equal to 0, "
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

    trainable_names = enable_bias_only(model)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # 再次確認沒有非 bias 參數被誤解凍
    invalid_trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.endswith(".bias")
    ]

    if invalid_trainable_names:
        raise RuntimeError(
            "BitFit incorrectly enabled non-bias parameters: "
            + ", ".join(invalid_trainable_names)
        )

    print("[BitFit] Trainable bias parameters:")

    for name in trainable_names:
        print(f"  - {name}")

    print(
        f"[BitFit] Trainable parameters: "
        f"{trainable_parameter_count:,} / "
        f"{total_parameter_count:,}"
    )

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=cfg.lr,
    )

    train_loop(
        cfg,
        model,
        optimizer,
        save_state_fn=get_bitfit_state_dict,
        method_name="bitfit",
    )


if __name__ == "__main__":
    main()