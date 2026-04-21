"""Fully-connected network for trajectory parameter prediction."""

from __future__ import annotations

import torch.nn as nn


class FCN(nn.Module):
    """Simple MLP: input -> hidden layers with ReLU -> linear output.

    Default architecture: FCN([5, 256, 512, 256, 52])
      - Input: 5-D (grasp_x, grasp_y, target_x, target_y, distance)
      - Output: 52-D (trajectory weights + time + release timing)
    """

    def __init__(self, dimensions: list[int]) -> None:
        super().__init__()
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
