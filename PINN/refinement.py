"""Two-endpoint PINN refinement for one forecasting interval."""
import numpy as np
import torch

from .pinn_model import network
from SINDY.operators import library


def derivative(field, points):
    return torch.autograd.grad(field.sum(), points, create_graph=True)[0]


def physics_loss(model, points, coefficients, offset, amplitude, spacing):
    # Network output is normalized intensity; identification uses the original units.
    value = model(points) * amplitude + offset
    dx, dy, dt = spacing
    grad = derivative(value, points)
    ux, uy, ut = grad[:, :1] / dx, grad[:, 1:2] / dy, grad[:, 2:] / dt
    gx, gy = derivative(ux, points), derivative(uy, points)
    terms = library(value, ux, uy, gx[:, :1] / dx, gy[:, 1:2] / dy, gx[:, 1:2] / dy)
    rhs = sum(weight * term for weight, term in zip(coefficients, terms))
    return ((ut - rhs) / amplitude).square().mean()


def fit_block(previous, coarse, bounds, coefficients, args, device):
    y0, y1, x0, x1 = bounds
    height, width = y1 - y0, x1 - x0
    a = torch.as_tensor(previous[y0:y1, x0:x1], device=device)
    b = torch.as_tensor(coarse[y0:y1, x0:x1], device=device)
    offset = a.mean()
    amplitude = (torch.maximum(a.max(), b.max()) - torch.minimum(a.min(), b.min())).clamp_min(1e-5)
    targets = [(field - offset) / amplitude for field in (a, b)]
    # x and y are in [-1,1]. Convert network derivatives to identification units.
    spacing = ((width - 1) * args.dx / 2, (height - 1) * args.dy / 2, args.dt)
    model = network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weight = args.data_weight
    for _ in range(args.iterations):
        indices = torch.randint(height * width, (args.anchor_points,), device=device)
        x = (indices % width) * (2.0 / (width - 1)) - 1
        y = (indices // width) * (2.0 / (height - 1)) - 1
        data_loss = 0
        for time, target in enumerate(targets):
            points = torch.stack((x, y, torch.full_like(x, time)), dim=1)
            error = model(points).flatten() - target.flatten()[indices]
            data_loss = data_loss + error.square().mean()
        interior = torch.rand(args.domain_points, 3, device=device)
        interior[:, :2] = 2 * interior[:, :2] - 1
        loss = weight * data_loss + physics_loss(
            model, interior.requires_grad_(), coefficients, offset, amplitude, spacing)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, offset.detach(), amplitude.detach()


def evaluate_block(model, offset, amplitude, shape, bounds, device):
    y0, y1, x0, x1 = bounds
    height, width = y1 - y0, x1 - x0
    # Evaluate two pixels beyond each core to cover both one-pixel margins
    # between neighboring tiles. Only multiply covered margins are retained.
    ys = slice(max(0, y0 - 2), min(shape[0], y1 + 2))
    xs = slice(max(0, x0 - 2), min(shape[1], x1 + 2))
    with torch.inference_mode():
        y, x = torch.meshgrid(torch.arange(ys.start, ys.stop, device=device),
                              torch.arange(xs.start, xs.stop, device=device), indexing="ij")
        points = torch.stack(((x - x0) * (2.0 / (width - 1)) - 1,
                              (y - y0) * (2.0 / (height - 1)) - 1,
                              torch.ones_like(x)), dim=-1)
        output = model(points.reshape(-1, 3)).reshape(x.shape) * amplitude + offset
    return ys, xs, output.cpu().numpy()


def refine(previous, coarse, blocks, args, device, save_model=None):
    def predictions():
        for bounds, coefficients in blocks:
            model, offset, amplitude = fit_block(previous, coarse, bounds, coefficients, args, device)
            if save_model is not None:
                save_model(bounds, model, offset.item(), amplitude.item())
            yield bounds, evaluate_block(model, offset, amplitude, previous.shape, bounds, device)
    return assemble(coarse, predictions())


def assemble(coarse, predictions):
    refined = coarse.copy()
    core = np.zeros(coarse.shape, dtype=bool)
    sums = np.zeros(coarse.shape, dtype=np.float64)
    counts = np.zeros(coarse.shape, dtype=np.int32)
    for bounds, (ys, xs, output) in predictions:
        y0, y1, x0, x1 = bounds
        refined[y0:y1, x0:x1] = output[y0-ys.start:y1-ys.start, x0-xs.start:x1-xs.start]
        core[y0:y1, x0:x1] = True
        sums[ys, xs] += output
        counts[ys, xs] += 1
    blend = ~core & (counts >= 2)
    refined[blend] = sums[blend] / counts[blend]
    if not np.isfinite(refined).all():
        raise FloatingPointError("Non-finite corrected field; stopping before feedback")
    return refined
