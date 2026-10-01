# src/data/dataset.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class VFIDataset(Dataset):
    """
    root/
    video1/
        frame_000001.png ...
    video2/
        ...

    For each sample, four neighboring frames around a target frame are selected:
    input frames = [i-2, i-1, i+1, i+2]
    target frame = i

    This design avoids exposing the ground truth frame to the model and
    leverages temporal context to predict the intermediate frame.

    input:  [12, H, W]
    target: [3, H, W]
    """

    def __init__(self, root: str, resize: int = 256, in_frames: int = 4):
        assert in_frames == 4, "this research baseline uses 4 frames"
        self.root = Path(root)
        self.resize = resize
        self.in_frames = in_frames

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.videos: List[Path] = [p for p in sorted(self.root.iterdir()) if p.is_dir()]
        if not self.videos:
            raise RuntimeError(f"No video dirs found under: {self.root}")

        # build index: (video_dir, frame_files, center_i)
        self.index: List[Tuple[Path, List[Path], int]] = []
        for vdir in self.videos:
            frames = sorted([f for f in vdir.iterdir() if f.suffix.lower() in IMG_EXTS])
            if len(frames) < 5:
                continue
            for i in range(2, len(frames) - 2):
                self.index.append((vdir, frames, i))

        if not self.index:
            raise RuntimeError(
                "No valid samples (need >=5 frames per video folder)."
            )

    def __len__(self):
        return len(self.index)

    def _read_rgb_chw(self, fpath: Path) -> np.ndarray:
        bgr = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {fpath}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0  # [H,W,3]
        chw = np.transpose(rgb, (2, 0, 1))    # [3,H,W]
        return chw

    def __getitem__(self, idx: int):
        _, frames, i = self.index[idx]

        # 使用前兩幀 + 後兩幀（不包含 i）
        idxs = [i - 2, i - 1, i + 1, i + 2]

        imgs = [self._read_rgb_chw(frames[j]) for j in idxs]
        inp = np.concatenate(imgs, axis=0)  # [12,H,W]

        tgt = self._read_rgb_chw(frames[i])

        return torch.from_numpy(inp), torch.from_numpy(tgt)


def split_indices_by_video(
    dataset: VFIDataset,
    val_ratio: float = 0.05,
    seed: int = 42,
):
    video_dirs = sorted({
        video_dir
        for video_dir, _, _ in dataset.index
    })

    if len(video_dirs) < 2:
        raise RuntimeError(
            "Need at least 2 valid video folders to create "
            "separate train and validation sets."
        )

    rng = np.random.RandomState(seed)
    shuffled_videos = video_dirs.copy()
    rng.shuffle(shuffled_videos)

    val_video_count = max(
        1,
        int(round(len(shuffled_videos) * val_ratio))
    )

    # 至少保留一部影片作為 training data
    val_video_count = min(
        val_video_count,
        len(shuffled_videos) - 1
    )

    val_videos = set(shuffled_videos[:val_video_count])
    train_videos = set(shuffled_videos[val_video_count:])

    train_idx = []
    val_idx = []

    for sample_idx, (video_dir, _, _) in enumerate(dataset.index):
        if video_dir in train_videos:
            train_idx.append(sample_idx)
        elif video_dir in val_videos:
            val_idx.append(sample_idx)

    if not train_idx:
        raise RuntimeError("Training set is empty.")

    if not val_idx:
        raise RuntimeError("Validation set is empty.")

    print(
        f"[Split] train_videos={len(train_videos)}, "
        f"val_videos={len(val_videos)}, "
        f"train_samples={len(train_idx)}, "
        f"val_samples={len(val_idx)}"
    )

    return train_idx, val_idx