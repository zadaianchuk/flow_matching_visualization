"""Velocity field neural network."""

import torch
import torch.nn as nn


class VelocityMLP(nn.Module):
    """
    Simple MLP to learn the velocity field v(x, t).

    Takes position x and time t as input, outputs velocity.
    For 1D flow matching: input is (x, t) -> output is scalar velocity.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [nn.Linear(2, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Position tensor, shape (batch,) or (batch, 1)
            t: Time tensor, shape (batch,) or scalar

        Returns:
            Velocity tensor, shape (batch,)
        """
        # Handle different input shapes
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        if t.dim() == 0:
            t = t.expand(x.shape[0]).unsqueeze(-1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        # Concatenate x and t
        inp = torch.cat([x, t], dim=-1)  # (batch, 2)

        return self.net(inp).squeeze(-1)  # (batch,)
