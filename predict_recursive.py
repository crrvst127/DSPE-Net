import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "CLSTM"))
from sequence_dataset import positive_scale, read_sample
from model import build_model
from PINN.refinement import refine
from SINDY.operators import TERMS


def read_blocks(path, window, shape, scenario, initial_time):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("method") != "STRidge" or data.get("state") != "u" or tuple(data.get("terms", ())) != TERMS:
        raise ValueError("Regenerate operators with SINDY.identify_equations; expected the 13-term field library")
    if data["scenario"] != scenario or tuple(data["shape"]) != shape:
        raise ValueError("Operator scenario or grid does not match the forecast")
    blocks = []
    occupied = np.zeros(shape, dtype=bool)
    for key, record in data["blocks"].items():
        pos = record["position"]
        if pos["t_range_start"] != window:
            continue
        if pos["end_time"] > initial_time:
            raise ValueError("Identification window includes observations after the forecast origin")
        x0, x1 = pos["x_range_start"], pos["x_range_end"]
        y0, y1 = pos["y_range_start"], pos["y_range_end"]
        if not (0 <= x0 < x1 - 1 < shape[1] and 0 <= y0 < y1 - 1 < shape[0]):
            raise ValueError(f"Invalid block bounds: {key}")
        if occupied[y0:y1, x0:x1].any():
            raise ValueError(f"Overlapping core: {key}")
        occupied[y0:y1, x0:x1] = True
        coeff = np.asarray(record["coefficients"], dtype=float)
        if coeff.shape != (len(TERMS),) or not np.isfinite(coeff).all():
            raise ValueError(f"Invalid coefficients in block {key}")
        blocks.append(((y0, y1, x0, x1), coeff))
    if not blocks:
        raise ValueError(f"No equations for window {window}")
    return blocks, positive_scale(data["dx"]), positive_scale(data["dy"])


def rollout(model, current, blocks, args, device):
    args.output.mkdir(parents=True, exist_ok=False)
    for step in range(1, args.steps + 1):
        with torch.inference_mode():
            inputs = torch.as_tensor(current / args.scale, device=device)[None, None, None]
            coarse = model(inputs)[0, 0, 0].cpu().numpy() * args.scale
        if not np.isfinite(coarse).all():
            raise FloatingPointError(f"Non-finite coarse forecast at step {step}")
        args.dt = args.intervals[step-1]
        current = refine(current, coarse, blocks, args, device)
        np.save(args.output / f"coarse_{step:03}.npy", coarse)
        np.save(args.output / f"refined_{step:03}.npy", current)
        print(f"Completed step {step}/{args.steps}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("equations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=int, required=True, help="Start index of the historical operator window")
    parser.add_argument("--scenario", choices=("numerical","satellite"), required=True)
    parser.add_argument("--initial-time", type=float, required=True, help="Forecast origin in minutes on the MAT time axis")
    parser.add_argument("--intervals", type=positive_scale, nargs="+", help="Minutes per forecast step")
    parser.add_argument("--scale", type=positive_scale)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=15000, help="Adam updates per block per step")
    parser.add_argument("--domain-points", type=int, default=1024)
    parser.add_argument("--anchor-points", type=int, default=1024)
    args = parser.parse_args()
    args.data_weight = 1 if args.scenario == "numerical" else 10
    if not np.isfinite(args.initial_time):
        parser.error("Forecast origin must be finite")
    if args.intervals is None:
        args.intervals = [10 if args.scenario == "numerical" else 1] * args.steps
    if len(args.intervals) != args.steps:
        parser.error("Provide one interval per forecast step")
    for name in ("steps", "iterations", "domain_points", "anchor_points"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    current = read_sample(args.sample, 1)[0, 0].numpy()
    if not np.isfinite(current).all():
        parser.error("Input sample contains non-finite values")
    blocks, args.dx, args.dy = read_blocks(args.equations, args.window, current.shape, args.scenario, args.initial_time)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("scenario", args.scenario) != args.scenario:
        parser.error("Checkpoint belongs to a different scenario")
    if args.scale is None:
        args.scale = checkpoint.get("scale", 1.0)
    elif "scale" in checkpoint and args.scale != checkpoint["scale"]:
        parser.error("Input scale must match training")
    state = checkpoint["state_dict"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    rollout(model.eval().requires_grad_(False), current, blocks, args, device)


if __name__ == "__main__":
    main()
