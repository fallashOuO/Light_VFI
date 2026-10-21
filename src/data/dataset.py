# src/data/dataset.py

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


def natural_sort_key(path: Path) -> list[object]:
    """
    使用自然順序排列檔名。

    例如：
    frame_1.png
    frame_2.png
    frame_10.png

    而不是：
    frame_1.png
    frame_10.png
    frame_2.png
    """
    parts = re.split(
        r"(\d+)",
        path.name.lower(),
    )

    return [
        int(part) if part.isdigit() else part
        for part in parts
    ]


class VFIDataset(Dataset):
    """
    資料夾格式：

    root/
        video1/
            frame_000001.png
            frame_000002.png
            ...
        video2/
            frame_000001.png
            frame_000002.png
            ...

    對於中心幀 i：

    輸入：
        [i-2, i-1, i+1, i+2]

    目標：
        i

    中心幀不會出現在輸入中，因此不會直接洩漏
    ground-truth frame。

    輸入形狀：
        [12, H, W]

    目標形狀：
        [3, H, W]
    """

    def __init__(
        self,
        root: str | Path,
        resize: int = 256,
        in_frames: int = 4,
    ) -> None:
        if in_frames != 4:
            raise ValueError(
                "VFIDataset currently requires exactly "
                f"4 input frames, got {in_frames}."
            )

        if resize <= 0:
            raise ValueError(
                f"resize must be greater than 0, got {resize}"
            )

        self.root = Path(root)
        self.resize = resize
        self.in_frames = in_frames

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Dataset root not found: {self.root}"
            )

        candidate_video_dirs = sorted(
            (
                path
                for path in self.root.iterdir()
                if path.is_dir()
            ),
            key=natural_sort_key,
        )

        if not candidate_video_dirs:
            raise RuntimeError(
                f"No video directories found under: {self.root}"
            )

        # 每個元素為：
        # (video_dir, ordered_frame_paths, center_index)
        self.index: list[
            tuple[
                Path,
                tuple[Path, ...],
                int,
            ]
        ] = []

        self.videos: list[Path] = []

        skipped_video_count = 0

        for video_dir in candidate_video_dirs:
            frame_paths = sorted(
                (
                    path
                    for path in video_dir.iterdir()
                    if (
                        path.is_file()
                        and path.suffix.lower() in IMG_EXTS
                    )
                ),
                key=natural_sort_key,
            )

            # 至少需要：
            # i-2、i-1、i、i+1、i+2
            if len(frame_paths) < 5:
                skipped_video_count += 1
                continue

            frame_paths_tuple = tuple(frame_paths)
            self.videos.append(video_dir)

            for center_index in range(
                2,
                len(frame_paths_tuple) - 2,
            ):
                self.index.append(
                    (
                        video_dir,
                        frame_paths_tuple,
                        center_index,
                    )
                )

        if not self.index:
            raise RuntimeError(
                "No valid VFI samples were found. "
                "Each video directory must contain at least "
                "5 readable image files."
            )

        print(
            f"[Dataset] valid_videos={len(self.videos)}, "
            f"skipped_videos={skipped_video_count}, "
            f"samples={len(self.index)}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def _read_rgb_chw(
        self,
        image_path: Path,
    ) -> np.ndarray:
        bgr_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if bgr_image is None:
            raise RuntimeError(
                f"Failed to read image: {image_path}"
            )

        rgb_image = cv2.cvtColor(
            bgr_image,
            cv2.COLOR_BGR2RGB,
        )

        original_height, original_width = (
            rgb_image.shape[:2]
        )

        # 縮小使用 INTER_AREA；
        # 放大使用 INTER_CUBIC。
        if (
            original_height > self.resize
            or original_width > self.resize
        ):
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC

        rgb_image = cv2.resize(
            rgb_image,
            (self.resize, self.resize),
            interpolation=interpolation,
        )

        rgb_image = (
            rgb_image.astype(np.float32) / 255.0
        )

        chw_image = np.transpose(
            rgb_image,
            (2, 0, 1),
        )

        return np.ascontiguousarray(
            chw_image,
            dtype=np.float32,
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self.index)

        if index < 0 or index >= len(self.index):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        _, frame_paths, center_index = self.index[index]

        input_indices = [
            center_index - 2,
            center_index - 1,
            center_index + 1,
            center_index + 2,
        ]

        input_images = [
            self._read_rgb_chw(
                frame_paths[frame_index]
            )
            for frame_index in input_indices
        ]

        input_array = np.concatenate(
            input_images,
            axis=0,
        )

        target_array = self._read_rgb_chw(
            frame_paths[center_index]
        )

        input_tensor = torch.from_numpy(
            input_array
        )

        target_tensor = torch.from_numpy(
            target_array
        )

        return input_tensor, target_tensor


def split_indices_by_video(
    dataset: VFIDataset,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    以影片資料夾為單位切分訓練集與驗證集。

    同一部影片產生的 samples 不會同時出現在
    training 與 validation，避免相鄰影格造成資料洩漏。
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(
            "val_ratio must satisfy 0 <= val_ratio < 1, "
            f"got {val_ratio}"
        )

    video_dirs = sorted(
        {
            video_dir
            for video_dir, _, _ in dataset.index
        },
        key=natural_sort_key,
    )

    if not video_dirs:
        raise RuntimeError(
            "The dataset contains no valid video folders."
        )

    # 若資料夾明確使用 *_train 與 *_val 命名，
    # 優先依名稱固定切分，避免亂數把兩者交換。
    explicit_train_videos = {
        video_dir
        for video_dir in video_dirs
        if video_dir.name.lower().endswith("_train")
    }
    explicit_val_videos = {
        video_dir
        for video_dir in video_dirs
        if video_dir.name.lower().endswith("_val")
    }

    if explicit_train_videos or explicit_val_videos:
        if not explicit_train_videos:
            raise RuntimeError(
                "Found *_val folders but no *_train folders."
            )

        if not explicit_val_videos:
            raise RuntimeError(
                "Found *_train folders but no *_val folders."
            )

        unassigned_videos = (
            set(video_dirs)
            - explicit_train_videos
            - explicit_val_videos
        )

        if unassigned_videos:
            names = ", ".join(
                sorted(
                    video_dir.name
                    for video_dir in unassigned_videos
                )
            )
            raise RuntimeError(
                "When explicit *_train and *_val folders are used, "
                "every valid video folder must use one of those "
                f"suffixes. Unassigned folders: {names}"
            )

        train_indices: list[int] = []
        val_indices: list[int] = []

        for sample_index, (
            video_dir,
            _,
            _,
        ) in enumerate(dataset.index):
            if video_dir in explicit_train_videos:
                train_indices.append(sample_index)
            elif video_dir in explicit_val_videos:
                val_indices.append(sample_index)

        if not train_indices or not val_indices:
            raise RuntimeError(
                "Explicit training or validation split is empty."
            )

        print(
            f"[Split] mode=explicit, "
            f"train_videos={len(explicit_train_videos)}, "
            f"val_videos={len(explicit_val_videos)}, "
            f"train_samples={len(train_indices)}, "
            f"val_samples={len(val_indices)}"
        )

        return train_indices, val_indices

    # 不需要驗證集時，全部樣本都作為訓練資料。
    if val_ratio == 0.0:
        train_indices = list(
            range(len(dataset))
        )

        print(
            f"[Split] train_videos={len(video_dirs)}, "
            "val_videos=0, "
            f"train_samples={len(train_indices)}, "
            "val_samples=0"
        )

        return train_indices, []

    if len(video_dirs) < 2:
        raise RuntimeError(
            "At least 2 valid video folders are required "
            "when val_ratio is greater than 0."
        )

    random_generator = np.random.RandomState(seed)

    shuffled_videos = video_dirs.copy()
    random_generator.shuffle(shuffled_videos)

    val_video_count = max(
        1,
        int(round(
            len(shuffled_videos) * val_ratio
        )),
    )

    # 至少保留一部影片作為訓練資料。
    val_video_count = min(
        val_video_count,
        len(shuffled_videos) - 1,
    )

    val_videos = set(
        shuffled_videos[:val_video_count]
    )

    train_videos = set(
        shuffled_videos[val_video_count:]
    )

    train_indices: list[int] = []
    val_indices: list[int] = []

    for sample_index, (
        video_dir,
        _,
        _,
    ) in enumerate(dataset.index):
        if video_dir in train_videos:
            train_indices.append(sample_index)

        elif video_dir in val_videos:
            val_indices.append(sample_index)

    if not train_indices:
        raise RuntimeError(
            "The training set is empty."
        )

    if not val_indices:
        raise RuntimeError(
            "The validation set is empty."
        )

    print(
        f"[Split] train_videos={len(train_videos)}, "
        f"val_videos={len(val_videos)}, "
        f"train_samples={len(train_indices)}, "
        f"val_samples={len(val_indices)}"
    )

    return train_indices, val_indices