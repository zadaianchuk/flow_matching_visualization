"""Distribution samplers for source and target distributions."""

import torch
import numpy as np
from scipy.stats import norm


def sample_standard_normal(n: int) -> torch.Tensor:
    """Sample from standard normal N(0, 1)."""
    return torch.randn(n)


def sample_bimodal(
    n: int,
    mu1: float = -2.0,
    mu2: float = 2.0,
    sigma: float = 0.5,
    weight: float = 0.5,
) -> torch.Tensor:
    """
    Sample from mixture of two Gaussians.

    Args:
        n: Number of samples
        mu1: Mean of first mode
        mu2: Mean of second mode
        sigma: Standard deviation of both modes
        weight: Weight of first mode (second mode has weight 1-weight)

    Returns:
        Tensor of samples, shuffled randomly
    """
    n1 = int(n * weight)
    n2 = n - n1

    samples1 = torch.randn(n1) * sigma + mu1
    samples2 = torch.randn(n2) * sigma + mu2

    samples = torch.cat([samples1, samples2])
    # Shuffle to randomize pairing
    return samples[torch.randperm(n)]


def bimodal_pdf(
    x: np.ndarray,
    mu1: float = -2.0,
    mu2: float = 2.0,
    sigma: float = 0.5,
    weight: float = 0.5,
) -> np.ndarray:
    """
    Compute bimodal PDF for plotting.

    Args:
        x: Points at which to evaluate PDF
        mu1, mu2: Means of the two modes
        sigma: Standard deviation
        weight: Weight of first mode

    Returns:
        PDF values at x
    """
    return weight * norm.pdf(x, mu1, sigma) + (1 - weight) * norm.pdf(x, mu2, sigma)


def standard_normal_pdf(x: np.ndarray) -> np.ndarray:
    """Compute standard normal PDF for plotting."""
    return norm.pdf(x, 0, 1)
