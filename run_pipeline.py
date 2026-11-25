"""Identify, train and forecast from the repository data layout."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from CLSTM.sequence_dataset import ordered_samples, positive_scale

ROOT = Path(__file__).resolve().parent


def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/run")
    parser.add_argument("--scenario", choices=("numerical", "satellite"), default="numerical")
    parser.add_argument("--scale", type=positive_scale, default=1)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--intervals", type=positive_scale, nargs="+")
    args = parser.parse_args()
    data, out = args.data.resolve(), args.output.resolve()
    development, evaluation, history = (ordered_samples(data / name) for name in
                                         ("development", "evaluation", "history"))
    if len(development) < 10 or len(evaluation) < 2 or len(history) < 5:
        parser.error("Supply at least 10 development, 2 evaluation and 5 history NPY arrays (see data/README.md)")
    if min(args.steps, args.iterations) < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("Steps, iterations and epochs must be positive")
    if args.intervals and len(args.intervals) != args.steps:
        parser.error("Provide one interval per forecast step")
    shape = np.load(evaluation[0], mmap_mode="r", allow_pickle=False).shape
    if len(shape) != 2 or any(n % 4 for n in shape) or not 4 <= args.tile_size <= min(shape):
        parser.error("Use a 2-D grid divisible by four and a tile size between 4 and its shortest side")
    for path in development + evaluation + history[-5:]:
        if np.load(path, mmap_mode="r", allow_pickle=False).shape != shape:
            parser.error(f"Inconsistent grid dimensions: {path}")
    times_path = data / "history/times.txt"
    if times_path.exists():
        times = np.atleast_1d(np.loadtxt(times_path))
        if times.shape != (len(history),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or times[-1] > 0:
            parser.error("history/times.txt needs one increasing time per sample in minutes relative to forecast origin (<= 0)")
        times = times[-5:]
    else:
        interval = 10 if args.scenario == "numerical" else 1
        times = np.arange(-5, 0, dtype=float) * interval
    if out.exists():
        parser.error("Choose a new --output directory for this run")
    out.mkdir(parents=True)
    manifest = out / "history.txt"
    manifest.write_text("\n".join(str(p) for p in history[-5:]) + "\n", encoding="utf-8")
    np.savetxt(out / "times.txt", times)
    run("SINDY/prepare_sequence.py", manifest, out / "history.mat", "--times", out / "times.txt")
    run("-m", "SINDY.identify_equations", out / "history.mat", out / "operators.json",
        "--scenario", args.scenario, "--tile-size", args.tile_size)
    budget = [] if args.epochs is None else ["--epochs", args.epochs]
    run("CLSTM/train_convlstm.py", data / "development", data / "evaluation", out / "models",
        "--scenario", args.scenario, "--scale", args.scale, *budget)
    intervals = ["--intervals", *args.intervals] if args.intervals else []
    run("predict_recursive.py", evaluation[0], out / f"models/fold_{args.fold}/final.pt",
        out / "operators.json", out / "predictions", "--scenario", args.scenario,
        "--initial-time", 0, "--window", 0, "--iterations", args.iterations,
        "--steps", args.steps, *intervals)


if __name__ == "__main__":
    main()
