import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from model import build_model
from torch import nn
from torch.utils.data import DataLoader, Subset
from sequence_dataset import positive_scale, SamplePairs

def train(args, model, device, training, validation, folder):
    train_data = DataLoader(training, batch_size=args.batch_size)
    valid_data = DataLoader(validation, batch_size=args.batch_size)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for current, target in train_data:
            optimizer.zero_grad()
            loss = loss_fn(model(current.to(device)), target.to(device))
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for current, target in valid_data:
                total += loss_fn(model(current.to(device)), target.to(device)).item() * len(current)
        score = total / len(valid_data.dataset)
        scheduler.step(score)
        print(f"epoch {epoch}, validation MSE {score:.7g}")
        if score <= best:
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, scenario=args.scenario,
                            scale=args.scale, validation_mse=score), folder / "best.pt")
            best = score
    torch.save(dict(state_dict=model.state_dict(), epoch=args.epochs, scenario=args.scenario,
                    scale=args.scale, validation_mse=score), folder / "final.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("development", type=Path, help="Chronological development sequence")
    parser.add_argument("evaluation", type=Path, help="Disjoint reserved rollout sequence")
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenario", choices=("numerical","satellite"), required=True)
    parser.add_argument("--epochs", type=int, help="Override paper budget for a short run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scale", type=positive_scale, default=1.0)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 5000 if args.scenario == "numerical" else 1000
    if min(args.epochs, args.batch_size) < 1:
        parser.error("Epoch count and batch size must be positive")
    development = SamplePairs(args.development, args.scale)
    evaluation = SamplePairs(args.evaluation, args.scale)
    def fingerprints(paths):
        result = []
        for path in paths:
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1048576), b""):
                    digest.update(chunk)
                result.append(digest.digest())
        return result
    dev_hash = fingerprints(development.samples)
    if set(dev_hash).intersection(fingerprints(evaluation.samples)):
        parser.error("Development and evaluation contain a shared sample")
    n = len(development.samples)
    if n < 10 or len(set(dev_hash)) != n:
        parser.error("Five disjoint folds require ten or more distinct chronological samples")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold, indices in enumerate(np.array_split(np.arange(n), 5), 1):
        first, last = int(indices[0]), int(indices[-1])
        validation = list(range(first, last))
        training = [i for i in range(n-1) if i+1 < first or i > last]
        # Exclude pairs straddling either edge of the held-out interval.
        folder = args.output / f"fold_{fold}"
        folder.mkdir()
        (folder / "split.json").write_text(json.dumps(dict(
            train_pair_starts=training, validation_pair_starts=validation,
            development_samples=[p.name for p in development.samples],
            evaluation_samples=[p.name for p in evaluation.samples])), encoding="utf-8")
        torch.manual_seed(1996)
        model = build_model().to(device)
        train(args, model, device, Subset(development, training), Subset(development, validation), folder)


if __name__ == "__main__":
    main()
