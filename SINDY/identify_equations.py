"""Estimate local image-field evolution operators with thresholded ridge regression."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .operators import TERMS, library


def stridge(matrix, target, alpha=0.2, threshold=0.001):
    """Solve in field units, retaining only terms above the absolute threshold."""
    active = np.arange(matrix.shape[1])
    result = np.zeros(matrix.shape[1])
    while active.size:
        design = matrix[:, active]
        # Augmented least squares avoids squaring the condition number.
        ridge = np.sqrt(alpha) * np.eye(active.size)
        weights = np.linalg.lstsq(np.vstack((design, ridge)),
                                  np.r_[target, np.zeros(active.size)], rcond=None)[0]
        keep = np.abs(weights) >= threshold
        if keep.all():
            result[active] = weights
            break
        active = active[keep]
    return result


def fit_tile(block, dx, dy, times):
    # The outer pixel ring supplies centered stencils; only core samples enter the fit.
    u = block[1:-1, 1:-1]
    ux = (block[2:, 1:-1] - block[:-2, 1:-1]) / (2 * dx)
    uy = (block[1:-1, 2:] - block[1:-1, :-2]) / (2 * dy)
    uxx = (block[2:, 1:-1] - 2*u + block[:-2, 1:-1]) / dx**2
    uyy = (block[1:-1, 2:] - 2*u + block[1:-1, :-2]) / dy**2
    uxy = (block[2:, 2:] - block[2:, :-2] - block[:-2, 2:] + block[:-2, :-2]) / (4*dx*dy)
    matrix = np.column_stack([v.ravel() for v in library(u, ux, uy, uxx, uyy, uxy)])
    ut = np.gradient(u, times, axis=2, edge_order=2)
    return stridge(matrix, ut.ravel())


def identify(mat, start, count, tile, window, field_key, scenario):
    field = np.asarray(mat[field_key])
    x, y, t = (np.asarray(mat[key], dtype=float).ravel() for key in ("x", "y", "t"))
    if field.shape != (len(x), len(y), len(t)) or np.iscomplexobj(field):
        raise ValueError("Expected a real field[x,y,t] aligned with the coordinate vectors")
    if tile < 4 or min(len(x), len(y)) < tile or window < 3 or start < 0 or count < 1 or start+count+window-1 > len(t):
        raise ValueError("Incomplete spatial tile or time window")
    for axis in (x, y, t):
        if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("Coordinates must be finite and strictly increasing")
    dx, dy = x[1]-x[0], y[1]-y[0]
    if not np.allclose(np.diff(x), dx) or not np.allclose(np.diff(y), dy):
        raise ValueError("Expected a uniform spatial grid")
    blocks = {}
    for k in range(start, start + count):
        times = t[k:k+window]
        for i in range(len(x)//tile):
            left = len(x)%tile//2 + i*tile
            for j in range(len(y)//tile):
                top = len(y)%tile//2 + j*tile
                block = np.asarray(field[left:left+tile, top:top+tile, k:k+window], dtype=float)
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite field in tile {(i,j,k)}")
                coeff = fit_tile(block, dx, dy, times)
                blocks[f"{i}_{j}_{k}"] = dict(coefficients=coeff.tolist(),
                    position=dict(x_range_start=left+1, x_range_end=left+tile-1,
                                  y_range_start=top+1, y_range_end=top+tile-1,
                                  t_range_start=k, t_range_end=k+window-1, end_time=float(times[-1])))
    return dict(method="STRidge", state="u", alpha=0.2, threshold=0.001, terms=TERMS,
                scenario=scenario, dx=dx, dy=dy, shape=[len(y),len(x)], blocks=blocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat")
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenario", choices=("numerical","satellite"), required=True)
    parser.add_argument("--field-key", default="field")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=5)
    args = parser.parse_args()
    result = identify(loadmat(args.mat), args.start, args.count, args.tile_size, args.window_length, args.field_key, args.scenario)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
