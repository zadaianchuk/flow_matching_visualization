"""Tests for distributions module."""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, str(__file__).replace("/tests/test_distributions.py", "/src"))

from distributions import (
    sample_standard_normal,
    sample_bimodal,
    bimodal_pdf,
    standard_normal_pdf,
)


class TestSampleStandardNormal:
    def test_returns_correct_shape(self):
        samples = sample_standard_normal(100)
        assert samples.shape == (100,)

    def test_returns_tensor(self):
        samples = sample_standard_normal(50)
        assert isinstance(samples, torch.Tensor)

    def test_mean_approximately_zero(self):
        samples = sample_standard_normal(10000)
        assert abs(samples.mean().item()) < 0.1

    def test_std_approximately_one(self):
        samples = sample_standard_normal(10000)
        assert abs(samples.std().item() - 1.0) < 0.1


class TestSampleBimodal:
    def test_returns_correct_shape(self):
        samples = sample_bimodal(100)
        assert samples.shape == (100,)

    def test_returns_tensor(self):
        samples = sample_bimodal(50)
        assert isinstance(samples, torch.Tensor)

    def test_samples_in_expected_range(self):
        samples = sample_bimodal(1000, mu1=-2, mu2=2, sigma=0.5)
        # Most samples should be within 3 sigma of either mode
        assert samples.min() > -5
        assert samples.max() < 5

    def test_weight_affects_distribution(self):
        # All weight on mode 1
        samples = sample_bimodal(1000, mu1=-3, mu2=3, weight=1.0)
        assert samples.mean() < -2

        # All weight on mode 2
        samples = sample_bimodal(1000, mu1=-3, mu2=3, weight=0.0)
        assert samples.mean() > 2

    def test_equal_weights_centered(self):
        samples = sample_bimodal(10000, mu1=-2, mu2=2, weight=0.5)
        # Mean should be approximately 0
        assert abs(samples.mean().item()) < 0.2


class TestPDFFunctions:
    def test_standard_normal_pdf_shape(self):
        x = np.linspace(-3, 3, 100)
        pdf = standard_normal_pdf(x)
        assert pdf.shape == x.shape

    def test_standard_normal_pdf_max_at_zero(self):
        x = np.linspace(-3, 3, 101)  # Include zero
        pdf = standard_normal_pdf(x)
        max_idx = np.argmax(pdf)
        assert abs(x[max_idx]) < 0.1

    def test_bimodal_pdf_shape(self):
        x = np.linspace(-5, 5, 100)
        pdf = bimodal_pdf(x, mu1=-2, mu2=2)
        assert pdf.shape == x.shape

    def test_bimodal_pdf_has_two_modes(self):
        x = np.linspace(-5, 5, 1000)
        pdf = bimodal_pdf(x, mu1=-2, mu2=2, sigma=0.5, weight=0.5)

        # Find local maxima (modes)
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(pdf)
        assert len(peaks) == 2

    def test_bimodal_pdf_integrates_to_one(self):
        x = np.linspace(-10, 10, 10000)
        pdf = bimodal_pdf(x, mu1=-2, mu2=2)
        dx = x[1] - x[0]
        integral = np.sum(pdf) * dx
        assert abs(integral - 1.0) < 0.01
