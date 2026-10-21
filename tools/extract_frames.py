from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
from tqdm import tqdm


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, got {value}"
        )
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer, got {value}"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames with a progress bar."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=positive_int, default=1)
    parser.add_argument(
        "--ext",
        choices=["png", "jpg", "jpeg"],
        default="png",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
    )
    parser.add_argument(
        "--max_frames",
        type=non_negative_int,
        default=0,
        help="Maximum saved frames; 0 means unlimited.",
    )
    parser.add_argument(
        "--start_frame",
        type=non_negative_int,
        default=0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(
            f"Video file not found: {args.video}"
        )

    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError(
            "--jpeg_quality must be between 1 and 100."
        )

    args.output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open video: {args.video}"
        )

    saved_count = 0

    try:
        source_fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )
        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if (
            total_frames > 0
            and args.start_frame >= total_frames
        ):
            raise ValueError(
                f"--start_frame {args.start_frame} is outside "
                f"the video ({total_frames} frames)."
            )

        if args.start_frame > 0:
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                args.start_frame,
            )

        remaining_frames = max(
            0,
            total_frames - args.start_frame,
        )

        estimated_outputs = (
            math.ceil(remaining_frames / args.step)
            if total_frames > 0
            else None
        )

        if args.max_frames > 0:
            estimated_outputs = (
                min(estimated_outputs, args.max_frames)
                if estimated_outputs is not None
                else args.max_frames
            )

        output_fps = (
            source_fps / args.step
            if source_fps > 0
            else 0.0
        )

        print(f"[INFO] Video: {args.video}")
        print(f"[INFO] Source FPS: {source_fps:.3f}")
        print(f"[INFO] Output FPS: {output_fps:.3f}")
        print(
            f"[INFO] Reported total frames: "
            f"{total_frames:,}"
        )
        print(f"[INFO] Start frame: {args.start_frame:,}")
        print(f"[INFO] Step: {args.step}")
        print(f"[INFO] Format: {args.ext}")
        print(f"[INFO] Output directory: {args.output}")

        source_offset = 0

        progress = tqdm(
            total=estimated_outputs,
            desc=f"Extracting {args.video.name}",
            unit="frame",
            dynamic_ncols=True,
        )

        try:
            while True:
                if (
                    args.max_frames > 0
                    and saved_count >= args.max_frames
                ):
                    break

                ok, frame = capture.read()
                if not ok:
                    break

                if source_offset % args.step == 0:
                    output_path = args.output / (
                        f"frame_{saved_count:08d}.{args.ext}"
                    )

                    if (
                        output_path.exists()
                        and not args.overwrite
                    ):
                        raise FileExistsError(
                            f"Output already exists: {output_path}\n"
                            "Use a new output directory or "
                            "add --overwrite."
                        )

                    if args.ext in {"jpg", "jpeg"}:
                        success = cv2.imwrite(
                            str(output_path),
                            frame,
                            [
                                cv2.IMWRITE_JPEG_QUALITY,
                                args.jpeg_quality,
                            ],
                        )
                    else:
                        success = cv2.imwrite(
                            str(output_path),
                            frame,
                        )

                    if not success:
                        raise RuntimeError(
                            f"Failed to save frame: "
                            f"{output_path}"
                        )

                    saved_count += 1
                    progress.update(1)

                source_offset += 1
        finally:
            progress.close()
    finally:
        capture.release()

    if saved_count == 0:
        raise RuntimeError(
            "No frames were extracted."
        )

    print()
    print(
        f"[DONE] Saved {saved_count:,} frames to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
