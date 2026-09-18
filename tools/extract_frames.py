import argparse
import cv2
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    step: int = 1,
    ext: str = "png",
):
    """
    step = 1  → 保留全部幀 (例如 60fps)
    step = 2  → 每兩幀取一幀 (60fps → 30fps)
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Video: {video_path.name}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total frames: {total_frames}")
    print(f"[INFO] Step: {step}")
    print(f"[INFO] Output dir: {output_dir}")

    idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            out_path = output_dir / f"frame_{saved_idx:06d}.{ext}"

            if ext == "jpg":
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(out_path), frame)

            saved_idx += 1

        idx += 1

    cap.release()

    print(f"[DONE] Saved {saved_idx} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Output frame directory")
    parser.add_argument("--step", type=int, default=1, help="Frame step (1=all, 2=half)")
    parser.add_argument("--ext", default="png", choices=["png", "jpg"])

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        step=args.step,
        ext=args.ext,
    )