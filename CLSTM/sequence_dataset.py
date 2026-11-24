import argparse
import re
import numpy as np
import torch
from torch.utils.data import Dataset

def positive_scale(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number")
    return number


def read_sample(path, scale):
    sample = np.load(path, allow_pickle=False)
    if sample.ndim != 2 or any(size % 4 for size in sample.shape):
        raise ValueError(f"{path}: expected a 2-D sample with dimensions divisible by four")
    return torch.from_numpy(sample.astype(np.float32) / scale).unsqueeze(0).unsqueeze(0)


class SamplePairs(Dataset):
    def __init__(self, folder, scale):
        self.scale = scale
        if folder.is_file():
            paths = folder.read_text(encoding="utf-8-sig").splitlines()
            self.samples = [folder.parent / line.strip() for line in paths if line.strip()]
        else:
            self.samples = sorted(folder.glob("*.npy"), key=lambda p: [
                int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)
            ])
        if len(self.samples) < 2:
            raise ValueError(f"{folder}: adjacent pairs require at least two samples")

    def __len__(self):
        return len(self.samples) - 1

    def __getitem__(self, i):
        return read_sample(self.samples[i], self.scale), read_sample(self.samples[i + 1], self.scale)
