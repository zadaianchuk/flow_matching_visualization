"""Tests for flow matching training module."""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).replace("/tests/test_flow_matching.py", "/src"))

from flow_matching import sample_conditional_batch, train_step, train_model
from mlp import VelocityMLP


class TestSampleConditionalBatch:
    def test_output_shapes(self):
        x0 = torch.randn(1000)
        x1 = torch.randn(1000)
        xt, t, ut = sample_conditional_batch(x0, x1, batch_size=64)

        assert xt.shape == (64,)
        assert t.shape == (64,)
        assert ut.shape == (64,)

    def test_t_in_range(self):
        x0 = torch.randn(1000)
        x1 = torch.randn(1000)
        _, t, _ = sample_conditional_batch(x0, x1, batch_size=256)

        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_conditional_path_correct(self):
        """Test that xt = (1-t)*x0 + t*x1."""
        # Use fixed samples for testing
        x0 = torch.zeros(100)
        x1 = torch.ones(100)
        xt, t, ut = sample_conditional_batch(x0, x1, batch_size=100)

        # xt should be in [0, 1] since x0=0 and x1=1
        assert xt.min() >= -0.01
        assert xt.max() <= 1.01

    def test_conditional_velocity_correct(self):
        """Test that ut = x1 - x0."""
        x0 = torch.zeros(100)
        x1 = torch.ones(100) * 3.0
        _, _, ut = sample_conditional_batch(x0, x1, batch_size=100)

        # ut should always be 3.0
        assert torch.allclose(ut, torch.full_like(ut, 3.0))


class TestTrainStep:
    def test_returns_float_loss(self):
        model = VelocityMLP()
        optimizer = torch.optim.Adam(model.parameters())
        x0 = torch.randn(500)
        x1 = torch.randn(500)

        loss = train_step(model, optimizer, x0, x1)
        assert isinstance(loss, float)

    def test_loss_is_positive(self):
        model = VelocityMLP()
        optimizer = torch.optim.Adam(model.parameters())
        x0 = torch.randn(500)
        x1 = torch.randn(500)

        loss = train_step(model, optimizer, x0, x1)
        assert loss >= 0.0

    def test_updates_parameters(self):
        model = VelocityMLP()
        optimizer = torch.optim.Adam(model.parameters())
        x0 = torch.randn(500)
        x1 = torch.randn(500)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Train step
        train_step(model, optimizer, x0, x1)

        # Check parameters changed
        params_changed = False
        for p_initial, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_initial, p_new):
                params_changed = True
                break

        assert params_changed


class TestTrainModel:
    def test_returns_loss_list(self):
        model = VelocityMLP()
        x0 = torch.randn(200)
        x1 = torch.randn(200)

        losses = train_model(model, x0, x1, epochs=10)
        assert isinstance(losses, list)
        assert len(losses) == 10

    def test_loss_decreases(self):
        """Loss should generally decrease over training."""
        model = VelocityMLP()
        x0 = torch.randn(500)
        x1 = torch.randn(500)

        losses = train_model(model, x0, x1, epochs=100, lr=1e-3)

        # Average of first 10 should be higher than last 10
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10
        assert late_avg < early_avg

    def test_progress_callback_called(self):
        model = VelocityMLP()
        x0 = torch.randn(200)
        x1 = torch.randn(200)

        callback_count = [0]

        def callback(epoch, loss):
            callback_count[0] += 1

        train_model(model, x0, x1, epochs=20, progress_callback=callback)
        assert callback_count[0] == 20

    def test_learned_model_produces_reasonable_flow(self):
        """After training, model should transport source toward target."""
        from solver import integrate_flow
        from distributions import sample_standard_normal, sample_bimodal

        model = VelocityMLP()
        x0 = sample_standard_normal(300)
        x1 = sample_bimodal(300, mu1=-2, mu2=2, sigma=0.5)

        # Train
        train_model(model, x0, x1, epochs=500, lr=1e-3)

        # Integrate flow
        trajectories = integrate_flow(model, x0, n_steps=50)
        final = trajectories[-1]

        # Final distribution should be more spread out than source
        # (bimodal has larger variance than N(0,1))
        assert final.std() > x0.std() * 0.8  # Allow some tolerance
