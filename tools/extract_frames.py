from pathlib import Path

import cv2
from tqdm import tqdm

# 前面保留原本的影片資訊與設定
saved_count = 0
frame_index = 0

# 如果 step=1，total_output_frames 就約等於影片總幀數
total_output_frames = (
    (total_frames + step - 1) // step
    if total_frames > 0
    else None
)

progress = tqdm(
    total=total_output_frames,
    desc=f"Extracting {video_path.name}",
    unit="frame",
    dynamic_ncols=True,
)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_index % step == 0:
        output_path = output_dir / (
            f"{saved_count:08d}.{output_format}"
        )

        if output_format.lower() in {"jpg", "jpeg"}:
            success = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
        else:
            success = cv2.imwrite(str(output_path), frame)

        if not success:
            progress.close()
            cap.release()
            raise RuntimeError(
                f"Failed to save frame: {output_path}"
            )

        saved_count += 1
        progress.update(1)

    frame_index += 1

progress.close()
cap.release()

print(f"[DONE] Saved {saved_count} frames to: {output_dir}")