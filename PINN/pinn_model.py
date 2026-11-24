"""Local coordinate model and differential operator used by the PINN branch."""
import os
os.environ.setdefault("DDE_BACKEND", "pytorch")
import deepxde as dde


def network():
    return dde.nn.FNN([3] + [64] * 10 + [1], "tanh", "Glorot uniform")

