"""ODE solver using torchdiffeq."""

import torch
from torchdiffeq import odeint


def integrate_flow(
    model: torch.nn.Module,
    x0: torch.Tensor,
    t_span: torch.Tensor = None,
    n_steps: int = 100,
) -> torch.Tensor:
    """
    Integrate the flow ODE: dx/dt = v(x, t) from t=0 to t=1.

    Args:
        model: Velocity field model v(x, t)
        x0: Initial positions, shape (n_particles,)
        t_span: Time points for integration. If None, uses linspace(0, 1, n_steps)
        n_steps: Number of time steps if t_span is None

    Returns:
        Trajectories tensor, shape (n_times, n_particles)
    """
    if t_span is None:
        t_span = torch.linspace(0, 1, n_steps)

    def velocity_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ODE right-hand side."""
        return model(x, t)

    # Integrate ODE
    with torch.no_grad():
        trajectories = odeint(velocity_fn, x0, t_span)

    return trajectories  # (n_times, n_particles)


def get_positions_at_time(
    trajectories: torch.Tensor,
    t: float,
    t_span: torch.Tensor = None,
    n_steps: int = 100,
) -> torch.Tensor:
    """
    Get particle positions at a specific time by interpolating trajectories.

    Args:
        trajectories: Full trajectories, shape (n_times, n_particles)
        t: Time value in [0, 1]
        t_span: Time points used for integration
        n_steps: Number of steps if t_span is None

    Returns:
        Positions at time t, shape (n_particles,)
    """
    if t_span is None:
        t_span = torch.linspace(0, 1, n_steps)

    # Find nearest time index
    idx = int(t * (len(t_span) - 1))
    idx = min(max(idx, 0), len(t_span) - 1)

    return trajectories[idx]
