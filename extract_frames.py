# extract_frames.py
import os
from pathlib import Path

import cv2


def extract_frames_from_videos(
    video_dir: str = "videos",
    output_root: str = "frames",
    img_ext: str = "png",
    every_nth: int = 1,
):
    """
    把 video_dir 底下所有影片逐幀切出來，存到 output_root。
    每部影片一個子資料夾，例如：
      videos/mc_1.mp4 -> frames/mc_1/frame_000001.png
    """
    video_dir = Path(video_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_files = []
    for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov"):
        video_files.extend(list(video_dir.glob(ext)))

    if not video_files:
        print(f"在 {video_dir} 底下沒有找到影片檔。請放影片到 videos/ 再試一次。")
        return

    print(f"找到 {len(video_files)} 部影片。開始切幀...")

    for vid_path in video_files:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"⚠ 無法開啟影片：{vid_path}")
            continue

        # 影片名稱（不含副檔名）當子資料夾名
        vid_name = vid_path.stem
        out_dir = output_root / vid_name
        out_dir.mkdir(parents=True, exist_ok=True)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{vid_name}] 幀數約 {total}，開始匯出...")

        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % every_nth == 0:
                # OpenCV 讀進來是 BGR，不用轉也沒關係，之後 Dataset 會轉成 RGB
                out_path = out_dir / f"frame_{idx:06d}.{img_ext}"
                cv2.imwrite(str(out_path), frame)
                saved += 1

            idx += 1

        cap.release()
        print(f"[{vid_name}] 匯出 {saved} 張影像到 {out_dir}")

    print("全部影片切幀完成。")


if __name__ == "__main__":
    # 預設：從 videos/ 切到 frames/，每幀都存
    extract_frames_from_videos(
        video_dir="videos",
        output_root="frames",
        img_ext="png",
        every_nth=1,
    )
