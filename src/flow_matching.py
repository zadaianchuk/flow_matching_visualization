"""Conditional Flow Matching training logic."""

import copy
import torch
import torch.nn.functional as F
from typing import List, Callable, Dict, Optional


def sample_conditional_batch(
    x0: torch.Tensor,
    x1: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a batch for CFM training.

    Args:
        x0: Source samples, shape (N,)
        x1: Target samples, shape (N,)
        batch_size: Number of samples in batch

    Returns:
        xt: Interpolated positions, shape (batch_size,)
        t: Time values, shape (batch_size,)
        ut: Conditional velocities (targets), shape (batch_size,)
    """
    # Sample random indices
    idx = torch.randint(0, len(x0), (batch_size,))
    x0_batch = x0[idx]
    x1_batch = x1[idx]

    # Sample random times
    t = torch.rand(batch_size)

    # Conditional path: xt = (1-t)*x0 + t*x1
    xt = (1 - t) * x0_batch + t * x1_batch

    # Conditional velocity: ut = x1 - x0 (constant!)
    ut = x1_batch - x0_batch

    return xt, t, ut


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    x1: torch.Tensor,
    batch_size: int = 256,
) -> float:
    """
    Single training step for CFM.

    Args:
        model: Velocity field model
        optimizer: PyTorch optimizer
        x0: Source samples
        x1: Target samples
        batch_size: Batch size

    Returns:
        Loss value (float)
    """
    # Sample batch
    xt, t, ut = sample_conditional_batch(x0, x1, batch_size)

    # Forward pass
    v_pred = model(xt, t)

    # MSE loss
    loss = F.mse_loss(v_pred, ut)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(
    model: torch.nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    progress_callback: Callable[[int, float], None] = None,
) -> List[float]:
    """
    Train the velocity field model using CFM.

    Args:
        model: Velocity field model
        x0: Source samples
        x1: Target samples
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        progress_callback: Optional callback(epoch, loss) for progress updates

    Returns:
        List of loss values per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        loss = train_step(model, optimizer, x0, x1, batch_size)
        losses.append(loss)

        if progress_callback is not None:
            progress_callback(epoch, loss)

    return losses


def train_model_with_checkpoints(
    model: torch.nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    checkpoint_epochs: Optional[List[int]] = None,
    progress_callback: Callable[[int, float], None] = None,
) -> tuple[List[float], Dict[int, dict]]:
    """
    Train the velocity field model using CFM with checkpoint saving.

    Args:
        model: Velocity field model
        x0: Source samples
        x1: Target samples
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        checkpoint_epochs: List of epochs to save checkpoints (default: [0, 10, 50, 100, 500, epochs])
        progress_callback: Optional callback(epoch, loss) for progress updates

    Returns:
        Tuple of (losses, checkpoints) where:
        - losses: List of loss values per epoch
        - checkpoints: Dict mapping epoch -> model state_dict
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [0, 10, 50, 100, 500, epochs]

    # Filter to only include epochs within training range
    checkpoint_epochs = sorted(set(e for e in checkpoint_epochs if e <= epochs))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    checkpoints = {}

    # Save initial state (epoch 0)
    if 0 in checkpoint_epochs:
        checkpoints[0] = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        loss = train_step(model, optimizer, x0, x1, batch_size)
        losses.append(loss)

        # Save checkpoint if this epoch is in the list
        epoch_num = epoch + 1  # epoch is 0-indexed, but we want 1-indexed for checkpoints
        if epoch_num in checkpoint_epochs:
            checkpoints[epoch_num] = copy.deepcopy(model.state_dict())

        if progress_callback is not None:
            progress_callback(epoch, loss)

    return losses, checkpoints
