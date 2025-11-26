# datasets.py
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoFramesVFITrainDataset(Dataset):
    """
    給 LightVFI 用的簡單 VFI dataset：
    - 假設資料夾結構：
        frames/
          video1/
            000000.png
            000001.png
            ...
          video2/
            000000.png
            ...
    - 每個 sample 取 4 幀 [i-1, i, i+1, i+2]（邊界 clamp）
      input : 4 幀疊成 [12, H, W]  (四張圖，各 3 channel)
      target: 中間幀 i+0.5 的「近似」，這裡先用第 i+1 幀當作 supervision
    """

    def __init__(
        self,
        frames_root: str,
        in_frames: int = 4,
        resize_h: int | None = None,
        resize_w: int | None = None,
        **kwargs,  # 忽略其他多餘參數（例如 subset、min_stride 之類）
    ):
        assert in_frames == 4, "目前這個 dataset 只實作 in_frames=4"
        self.frames_root = Path(frames_root)
        self.in_frames = in_frames
        self.resize_h = resize_h
        self.resize_w = resize_w

        if not self.frames_root.is_dir():
            raise RuntimeError(f"frames_root 不存在或不是資料夾: {self.frames_root}")

        # 收集所有影片資料夾
        self.video_dirs: List[Path] = []
        for p in sorted(self.frames_root.iterdir()):
            if p.is_dir():
                self.video_dirs.append(p)

        if not self.video_dirs:
            raise RuntimeError(f"在 {self.frames_root} 底下沒有找到子資料夾（影片幀）")

        # 建立 sample 索引: (video_dir, [idx0, idx1, idx2, idx3])
        self.samples: List[Tuple[Path, List[int]]] = []

        for vdir in self.video_dirs:
            frame_files = sorted(
                [f for f in vdir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]]
            )
            num = len(frame_files)
            if num < 4:
                continue

            # 我們讓 "中心點" 走到倒數第二幀
            # sample i: 使用 [i-1, i, i+1, i+2]（邊界 clamp）
            for i in range(1, num - 2):
                idx0 = max(i - 1, 0)
                idx1 = i
                idx2 = i + 1
                idx3 = i + 2
                self.samples.append((vdir, [idx0, idx1, idx2, idx3]))

        if not self.samples:
            raise RuntimeError(f"在 {self.frames_root} 沒有找到任何足夠幀數的影片可以做 sample")

        print(f"[Dataset] 在 {self.frames_root} 找到 {len(self.video_dirs)} 個影片資料夾")
        print(f"[Dataset] 總共建立 {len(self.samples)} 筆 samples")

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, vdir: Path, idx: int) -> np.ndarray:
        # 利用檔名排序後的列表來讀
        frame_files = sorted(
            [f for f in vdir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]]
        )
        idx = max(0, min(idx, len(frame_files) - 1))
        fpath = frame_files[idx]

        img_bgr = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"無法讀取影像檔: {fpath}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.resize_w is not None and self.resize_h is not None:
            img_rgb = cv2.resize(img_rgb, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        img_rgb = img_rgb.astype(np.float32) / 255.0  # [H,W,3]
        chw = np.transpose(img_rgb, (2, 0, 1))        # [3,H,W]
        return chw  # numpy float32

    def __getitem__(self, idx: int):
        vdir, frame_indices = self.samples[idx]

        # 讀四幀
        frames_chw = [self._load_frame(vdir, fi) for fi in frame_indices]  # list of [3,H,W]
        # 中間幀 supervision（這裡用第二張，對應 time ~ i+0.5 近似）
        target_chw = frames_chw[1]  # [3,H,W]

        # 疊成 [12,H,W]
        inp_12ch = np.concatenate(frames_chw, axis=0)  # [12,H,W]

        inp_tensor = torch.from_numpy(inp_12ch)       # [12,H,W]
        tgt_tensor = torch.from_numpy(target_chw)     # [3,H,W]

        return inp_tensor, tgt_tensor
