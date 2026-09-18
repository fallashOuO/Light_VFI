from pathlib import Path
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class VimeoTriplet(Dataset):
    def __init__(self, root, split="train", resize=256):
        self.root = Path(root)
        self.resize = resize

        if split == "train":
            list_file = "tri_trainlist.txt"
        else:
            list_file = "tri_testlist.txt"

        list_path = self.root / list_file
        if not list_path.is_file():
            raise FileNotFoundError(f"List file not found: {list_path}")

        with open(list_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.samples)

    def read_img(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, idx):
        seq_path = self.root / "sequences" / self.samples[idx]

        im1 = self.read_img(seq_path / "im1.png")
        im2 = self.read_img(seq_path / "im2.png")
        im3 = self.read_img(seq_path / "im3.png")

        inp = np.concatenate([im1, im1, im3, im3], axis=0)
        tgt = im2

        return torch.from_numpy(inp), torch.from_numpy(tgt)