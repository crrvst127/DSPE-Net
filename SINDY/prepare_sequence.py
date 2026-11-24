"""Build an [x,y,t] field from an ordered list of two-dimensional arrays."""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def prepare(manifest, output, times):
    entries = manifest.read_text(encoding="utf-8-sig").splitlines()
    paths = [manifest.parent / line.strip() for line in entries if line.strip()]
    if not paths:
        raise ValueError("The input list is empty")
    times = np.asarray(times, dtype=float).reshape(-1)
    if len(times) != len(paths) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Supply one strictly increasing acquisition time (minutes) per sample")
    for index, path in enumerate(paths):
        sample = np.load(path, allow_pickle=False)
        if sample.ndim != 2:
            raise ValueError(f"Expected a two-dimensional array: {path}")
        if index == 0:
            height, width = sample.shape
            field = np.empty((width, height, len(paths)), dtype=np.float32)
        if sample.shape != (height, width):
            raise ValueError(f"Array dimensions differ: {path}")
        field[:, :, index] = sample.T
    savemat(output, dict(field=field, x=np.arange(width), y=np.arange(height), t=times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--times", type=Path, required=True, help="Text file with acquisition times in minutes")
    args = parser.parse_args()
    prepare(args.manifest, args.output, np.loadtxt(args.times))
