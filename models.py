"""SmallCNN (CIFAR) and FCN (Fashion-MNIST) with the hooks curvature.py needs."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from config import RunConfig, sub_seed


class ClassifierMixin:
    """forward(x) = head(features(x)); `head` is the final nn.Linear whose rows are class blocks."""

    head: nn.Linear

    def features(self, x: Tensor) -> Tensor:  # input to last layer, [B, d]
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))

    @property
    def last_layer(self) -> nn.Linear:
        return self.head

    def last_layer_param_names(self) -> tuple[str, str]:
        return ("head.weight", "head.bias")


class MaxPool2x2(nn.Module):
    """Identical to nn.MaxPool2d(2) in the forward pass, written as pairwise torch.maximum over the four corners
    of each 2x2 window so it is differentiable to all orders on every backend. Measured on MPS (torch 2.14):
    nn.MaxPool2d has no double-backward / forward-mode rule, max_pool2d(return_indices=True) gives a wrong
    forward-mode HVP (26% error), and a two-axis amax is 7x slower than this formulation."""

    def forward(self, x: Tensor) -> Tensor:
        a = torch.maximum(x[..., ::2, ::2], x[..., 1::2, ::2])
        b = torch.maximum(x[..., ::2, 1::2], x[..., 1::2, 1::2])
        return torch.maximum(a, b)


class SmallCNN(ClassifierMixin, nn.Module):
    """3 x [conv3x3-ReLU-conv3x3-ReLU-maxpool2], GAP, Linear(widths[-1], C). No BatchNorm."""

    def __init__(self, C: int, in_ch: int = 3, widths=(32, 64, 128)):
        super().__init__()
        layers, ch = [], in_ch
        for w in widths:
            layers += [
                nn.Conv2d(ch, w, 3, padding=1), nn.ReLU(inplace=False),
                nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=False),
                MaxPool2x2(),
            ]
            ch = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(ch, C)

    def features(self, x: Tensor) -> Tensor:
        return self.body(x).mean(dim=(2, 3))  # global average pool -> [B, widths[-1]]


class FCN(ClassifierMixin, nn.Module):
    """AvgPool2d(2) -> flatten (196) -> Linear(196, hidden) -> ReLU -> Linear(hidden, C)."""

    def __init__(self, C: int, hidden: int = 32):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(196, hidden)
        self.head = nn.Linear(hidden, C)

    def features(self, x: Tensor) -> Tensor:
        return torch.relu(self.fc1(self.pool(x).flatten(1)))


def build_model(cfg: RunConfig) -> nn.Module:
    torch.manual_seed(sub_seed(cfg.seed, "init"))
    return SmallCNN(cfg.C) if cfg.protocol == "cifar_merge" else FCN(cfg.C)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_offsets(model: nn.Module) -> dict[str, tuple[int, int]]:
    """name -> (start, end) into the flat vector, in `named_parameters()` order
    (the order used by torch.func.functional_call / parameters_to_vector)."""
    out, off = {}, 0
    for name, p in model.named_parameters():
        out[name] = (off, off + p.numel())
        off += p.numel()
    return out


def class_block_indices(model: nn.Module) -> list[Tensor]:
    """For each class c: flat-parameter indices of head.weight[c, :] and head.bias[c]."""
    offs = param_offsets(model)
    w_name, b_name = model.last_layer_param_names()
    W = model.last_layer.weight
    C, d = W.shape
    w0, b0 = offs[w_name][0], offs[b_name][0]
    return [torch.cat([torch.arange(w0 + c * d, w0 + (c + 1) * d), torch.tensor([b0 + c])]) for c in range(C)]
