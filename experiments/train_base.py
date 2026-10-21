import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the LightVFI base model from scratch."
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Path to the custom training dataset.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("checkpoints/base"),
        help="Directory used to save base-model checkpoints.",
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
            "num_workers must be greater than or equal to 0, "
            f"got {args.num_workers}"
        )

    set_seed(args.seed)

    cfg = TrainConfig(
        data_root=str(args.data_root),
        base_ckpt="",
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

    print("Training the base model from scratch.")
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
        save_state_fn=lambda current_model: (
            current_model.state_dict()
        ),
        method_name="base",
    )


if __name__ == "__main__":
    main()