from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float


@dataclass(frozen=True)
class VideoSegment:
    video: VideoInfo
    first_center: int
    stop_center: int


class VideoVFIDataset(IterableDataset):
    """直接從影片串流產生四輸入幀與一個中心目標幀。

    每個樣本使用連續五幀：輸入為 [i-2, i-1, i+1, i+2]，
    目標為 i。資料只存在記憶體，不會寫成影格檔案。
    """

    def __init__(
        self,
        root: str | Path,
        resize: int = 256,
        in_frames: int = 4,
        sample_stride: int = 1,
        segment_samples: int = 4096,
        shuffle_buffer: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if in_frames != 4:
            raise ValueError("VideoVFIDataset requires exactly 4 input frames.")
        if resize <= 0:
            raise ValueError("resize must be greater than 0.")
        if sample_stride <= 0:
            raise ValueError("sample_stride must be greater than 0.")
        if segment_samples <= 0:
            raise ValueError("segment_samples must be greater than 0.")
        if shuffle_buffer < 0:
            raise ValueError("shuffle_buffer must be non-negative.")

        self.root = Path(root)
        self.resize = resize
        self.in_frames = in_frames
        self.sample_stride = sample_stride
        self.segment_samples = segment_samples
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        if not self.root.is_dir():
            raise NotADirectoryError(f"Video directory not found: {self.root}")

        video_paths = sorted(
            path
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        )
        if not video_paths:
            raise RuntimeError(f"No supported video files found in: {self.root}")

        self.videos: list[VideoInfo] = []
        self.segments: list[VideoSegment] = []
        self.sample_count = 0

        for path in video_paths:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open video: {path}")
            try:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()

            if frame_count < 5:
                print(f"[VideoDataset] skipped_short_video={path.name}")
                continue

            info = VideoInfo(path=path, frame_count=frame_count, fps=fps)
            self.videos.append(info)

            centers = list(range(2, frame_count - 2, sample_stride))
            self.sample_count += len(centers)

            for offset in range(0, len(centers), segment_samples):
                segment_centers = centers[offset:offset + segment_samples]
                self.segments.append(
                    VideoSegment(
                        video=info,
                        first_center=segment_centers[0],
                        stop_center=segment_centers[-1] + sample_stride,
                    )
                )

        if not self.videos or self.sample_count == 0:
            raise RuntimeError("No valid video samples were found.")

        print(
            f"[VideoDataset] root={self.root}, videos={len(self.videos)}, "
            f"samples={self.sample_count:,}, segments={len(self.segments)}, "
            f"stride={self.sample_stride}"
        )

    def __len__(self) -> int:
        return self.sample_count

    def _to_chw(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        interpolation = (
            cv2.INTER_AREA
            if height > self.resize or width > self.resize
            else cv2.INTER_CUBIC
        )
        rgb_image = cv2.resize(
            rgb_image,
            (self.resize, self.resize),
            interpolation=interpolation,
        )
        chw = np.transpose(rgb_image.astype(np.float32) / 255.0, (2, 0, 1))
        return np.ascontiguousarray(chw, dtype=np.float32)

    def _make_sample(
        self,
        frames: deque[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(frames) != 5:
            raise RuntimeError("A VFI sample requires exactly five frames.")
        arrays = [self._to_chw(frame) for frame in frames]
        input_array = np.concatenate(
            [arrays[0], arrays[1], arrays[3], arrays[4]],
            axis=0,
        )
        return torch.from_numpy(input_array), torch.from_numpy(arrays[2])

    def _iter_segment(
        self,
        segment: VideoSegment,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        capture = cv2.VideoCapture(str(segment.video.path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {segment.video.path}")

        try:
            start_frame = segment.first_center - 2
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            window: deque[np.ndarray] = deque(maxlen=5)
            for _ in range(5):
                ok, frame = capture.read()
                if not ok:
                    return
                window.append(frame)

            center = segment.first_center
            while center < segment.stop_center:
                yield self._make_sample(window)

                next_center = center + self.sample_stride
                if next_center >= segment.stop_center:
                    break

                for _ in range(self.sample_stride):
                    ok, frame = capture.read()
                    if not ok:
                        return
                    window.append(frame)
                center = next_center
        finally:
            capture.release()

    def _iter_raw(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        if worker is None:
            assigned_segments = self.segments
        else:
            assigned_segments = self.segments[worker.id::worker.num_workers]

        for segment in assigned_segments:
            yield from self._iter_segment(segment)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        raw_iterator = self._iter_raw()
        if self.shuffle_buffer <= 1:
            yield from raw_iterator
            return

        worker = get_worker_info()
        worker_seed = self.seed if worker is None else self.seed + worker.id
        rng = random.Random(worker_seed)
        buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

        for sample in raw_iterator:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(sample)
                continue
            selected = rng.randrange(len(buffer))
            yield buffer[selected]
            buffer[selected] = sample

        rng.shuffle(buffer)
        yield from buffer
