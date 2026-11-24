"""Fit both temporal anchors for one step and save the local coordinate models."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_recursive import read_blocks
from PINN.refinement import refine
from sequence_dataset import positive_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("equations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--initial-time", type=float, required=True)
    parser.add_argument("--scenario", choices=("numerical", "satellite"), required=True)
    parser.add_argument("--dt", type=positive_scale)
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--domain-points", type=int, default=1024)
    parser.add_argument("--anchor-points", type=int, default=1024)
    args = parser.parse_args()
    previous, coarse = (np.asarray(np.load(p), dtype=np.float32) for p in (args.previous, args.coarse))
    if previous.ndim != 2 or previous.shape != coarse.shape or not all(np.isfinite(a).all() for a in (previous,coarse)):
        parser.error("Supply two finite fields on the same spatial grid")
    if not np.isfinite(args.initial_time) or min(args.iterations,args.domain_points,args.anchor_points) < 1:
        parser.error("Invalid origin or training budget")
    blocks, args.dx, args.dy = read_blocks(args.equations, args.window, previous.shape, args.scenario, args.initial_time)
    args.data_weight = 1 if args.scenario == "numerical" else 10
    args.dt = args.dt or (10 if args.scenario == "numerical" else 1)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    def save(bounds, model, offset, amplitude):
        filename = f"block_{len(records)}.pt"
        torch.save(model.state_dict(), args.output / filename)
        records.append(dict(bounds=bounds, file=filename, offset=offset, amplitude=amplitude))
    refine(previous, coarse, blocks, args, device, save)
    np.save(args.output / "coarse.npy", coarse)
    (args.output / "blocks.json").write_text(json.dumps(records), encoding="utf-8")


if __name__ == "__main__":
    main()
