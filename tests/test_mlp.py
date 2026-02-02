"""Tests for MLP module."""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).replace("/tests/test_mlp.py", "/src"))

from mlp import VelocityMLP


class TestVelocityMLP:
    def test_output_shape_batch(self):
        model = VelocityMLP()
        x = torch.randn(32)
        t = torch.rand(32)
        out = model(x, t)
        assert out.shape == (32,)

    def test_output_shape_single(self):
        model = VelocityMLP()
        x = torch.randn(1)
        t = torch.tensor(0.5)
        out = model(x, t)
        assert out.shape == (1,)

    def test_scalar_time_broadcast(self):
        model = VelocityMLP()
        x = torch.randn(10)
        t = torch.tensor(0.5)  # Scalar
        out = model(x, t)
        assert out.shape == (10,)

    def test_different_hidden_dims(self):
        for hidden_dim in [32, 64, 128]:
            model = VelocityMLP(hidden_dim=hidden_dim)
            x = torch.randn(16)
            t = torch.rand(16)
            out = model(x, t)
            assert out.shape == (16,)

    def test_different_n_layers(self):
        for n_layers in [2, 3, 4]:
            model = VelocityMLP(n_layers=n_layers)
            x = torch.randn(16)
            t = torch.rand(16)
            out = model(x, t)
            assert out.shape == (16,)

    def test_gradient_flow(self):
        model = VelocityMLP()
        x = torch.randn(32, requires_grad=True)
        t = torch.rand(32)
        out = model(x, t)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_trainable_parameters(self):
        model = VelocityMLP()
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
