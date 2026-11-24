import argparse
from pathlib import Path
import torch
from model import build_model
import numpy as np
from sequence_dataset import positive_scale, read_sample

def forecast(args, model, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.scale is None:
        args.scale = checkpoint.get("scale", 1.0)
    elif "scale" in checkpoint and args.scale != checkpoint["scale"]:
        raise ValueError("Input scale must match training")
    weights = checkpoint["state_dict"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()})
    model.eval()
    current = read_sample(args.sample, args.scale).unsqueeze(0).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for step in range(1, args.steps + 1):
            current = model(current)
            np.save(args.output / f"coarse_{step:03}.npy", current[0, 0, 0].cpu().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--scale", type=positive_scale)
    args = parser.parse_args()
    torch.manual_seed(1996)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    forecast(args, model, device)


if __name__ == "__main__":
    main()
