"""Tests for ODE solver module."""

import pytest
import torch
import torch.nn as nn
import sys
sys.path.insert(0, str(__file__).replace("/tests/test_solver.py", "/src"))

from solver import integrate_flow, get_positions_at_time


class ConstantVelocityModel(nn.Module):
    """Test model with constant velocity."""

    def __init__(self, velocity: float = 1.0):
        super().__init__()
        self.velocity = velocity

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.velocity)


class LinearVelocityModel(nn.Module):
    """Test model with velocity = x (exponential growth)."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x


class TestIntegrateFlow:
    def test_output_shape(self):
        model = ConstantVelocityModel()
        x0 = torch.randn(50)
        trajectories = integrate_flow(model, x0, n_steps=20)
        assert trajectories.shape == (20, 50)

    def test_initial_condition(self):
        model = ConstantVelocityModel()
        x0 = torch.randn(50)
        trajectories = integrate_flow(model, x0, n_steps=100)
        # First time step should be initial condition
        assert torch.allclose(trajectories[0], x0)

    def test_constant_velocity_integration(self):
        """With v=1, particles should move from x0 to x0+1 over t=[0,1]."""
        model = ConstantVelocityModel(velocity=1.0)
        x0 = torch.zeros(10)
        trajectories = integrate_flow(model, x0, n_steps=100)
        final = trajectories[-1]
        # Should be approximately 1.0 (x0 + v*t = 0 + 1*1 = 1)
        assert torch.allclose(final, torch.ones(10), atol=0.05)

    def test_negative_velocity(self):
        """With v=-2, particles should move from x0 to x0-2."""
        model = ConstantVelocityModel(velocity=-2.0)
        x0 = torch.ones(10)
        trajectories = integrate_flow(model, x0, n_steps=100)
        final = trajectories[-1]
        # Should be approximately -1.0 (x0 + v*t = 1 + (-2)*1 = -1)
        assert torch.allclose(final, -torch.ones(10), atol=0.1)

    def test_custom_t_span(self):
        model = ConstantVelocityModel()
        x0 = torch.randn(20)
        t_span = torch.linspace(0, 0.5, 50)  # Only integrate to t=0.5
        trajectories = integrate_flow(model, x0, t_span=t_span)
        assert trajectories.shape == (50, 20)


class TestGetPositionsAtTime:
    def test_returns_correct_shape(self):
        trajectories = torch.randn(100, 50)  # 100 times, 50 particles
        pos = get_positions_at_time(trajectories, t=0.5, n_steps=100)
        assert pos.shape == (50,)

    def test_t_zero_returns_first(self):
        trajectories = torch.randn(100, 50)
        pos = get_positions_at_time(trajectories, t=0.0, n_steps=100)
        assert torch.allclose(pos, trajectories[0])

    def test_t_one_returns_last(self):
        trajectories = torch.randn(100, 50)
        pos = get_positions_at_time(trajectories, t=1.0, n_steps=100)
        assert torch.allclose(pos, trajectories[-1])

    def test_t_half_returns_middle(self):
        trajectories = torch.randn(100, 50)
        pos = get_positions_at_time(trajectories, t=0.5, n_steps=100)
        expected_idx = 49  # Index 49 for t=0.5 with 100 steps
        assert torch.allclose(pos, trajectories[expected_idx])
