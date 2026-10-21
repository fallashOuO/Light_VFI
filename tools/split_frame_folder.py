from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split one continuous frame folder into separate "
            "training and validation folders."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing continuous video frames.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Dataset root for the two output folders.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction assigned to training.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=10,
        help="Frames discarded between train and validation.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Copy is safer; move uses less extra disk space.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory not found: {input_dir}"
        )

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1.")

    if args.gap < 4:
        raise ValueError(
            "--gap must be at least 4 for the four-frame dataset."
        )

    frames = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(frames) < 20:
        raise RuntimeError(
            f"Not enough frames: found only {len(frames)}."
        )

    split_index = int(len(frames) * args.train_ratio)
    val_start = split_index + args.gap

    train_frames = frames[:split_index]
    val_frames = frames[val_start:]

    if len(train_frames) < 5 or len(val_frames) < 5:
        raise RuntimeError(
            "The split leaves fewer than five frames in one set."
        )

    train_dir = output_root / "video_001_train"
    val_dir = output_root / "video_001_val"

    if train_dir.exists() or val_dir.exists():
        raise FileExistsError(
            "Output folders already exist. Remove or rename them first:\n"
            f"  {train_dir}\n"
            f"  {val_dir}"
        )

    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    operation = shutil.copy2 if args.mode == "copy" else shutil.move

    print(f"[INFO] Total frames: {len(frames)}")
    print(f"[INFO] Train frames: {len(train_frames)}")
    print(f"[INFO] Discarded gap: {args.gap}")
    print(f"[INFO] Validation frames: {len(val_frames)}")
    print(f"[INFO] Mode: {args.mode}")

    try:
        from tqdm import tqdm

        train_iterator = tqdm(
            train_frames,
            desc="Preparing train frames",
            unit="frame",
            dynamic_ncols=True,
        )
    except ImportError:
        train_iterator = train_frames

    for index, source in enumerate(train_iterator):
        destination = train_dir / (
            f"frame_{index:08d}{source.suffix.lower()}"
        )
        operation(source, destination)

    try:
        from tqdm import tqdm

        val_iterator = tqdm(
            val_frames,
            desc="Preparing validation frames",
            unit="frame",
            dynamic_ncols=True,
        )
    except ImportError:
        val_iterator = val_frames

    for index, source in enumerate(val_iterator):
        destination = val_dir / (
            f"frame_{index:08d}{source.suffix.lower()}"
        )
        operation(source, destination)

    print("[DONE] Dataset split completed.")
    print(f"[DONE] Train: {train_dir}")
    print(f"[DONE] Validation: {val_dir}")


if __name__ == "__main__":
    main()