"""Two-panel visualization for flow matching (dark theme)."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from typing import Optional

from distributions import bimodal_pdf, standard_normal_pdf


# Dark theme colors
DARK_BG = "#1a1a2e"
DARK_PANEL = "#16213e"
ACCENT_BLUE = "#0f3460"
ACCENT_RED = "#e94560"
ACCENT_CYAN = "#00d9ff"
TEXT_COLOR = "#eaeaea"


def set_dark_style():
    """Set matplotlib dark theme."""
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_PANEL,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": "#333355",
            "grid.alpha": 0.3,
        }
    )


def plot_velocity_heatmap(
    ax: plt.Axes,
    model: torch.nn.Module,
    x_range: tuple = (-5, 5),
    n_x: int = 100,
    n_t: int = 50,
):
    """
    Plot velocity field as a heatmap (x vs t, color = v(x,t)).

    Args:
        ax: Matplotlib axes
        model: Trained velocity model
        x_range: Range of x values
        n_x: Number of x grid points
        n_t: Number of t grid points
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_x)
    t_vals = np.linspace(0, 1, n_t)

    # Create grid
    X, T = np.meshgrid(x_vals, t_vals)

    # Compute velocity at each grid point
    V = np.zeros_like(X)
    with torch.no_grad():
        for i, t in enumerate(t_vals):
            x_tensor = torch.tensor(x_vals, dtype=torch.float32)
            t_tensor = torch.full_like(x_tensor, t)
            v = model(x_tensor, t_tensor).numpy()
            V[i, :] = v

    # Plot heatmap
    vmax = np.abs(V).max()
    im = ax.pcolormesh(
        X,
        T,
        V,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )

    ax.set_xlabel("Position x", fontsize=10)
    ax.set_ylabel("Time t", fontsize=10)
    ax.set_title("Velocity Field v(x, t)", fontsize=12, color=TEXT_COLOR)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="Velocity")
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)


def plot_vector_field_with_trajectories(
    ax: plt.Axes,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    x0: np.ndarray,
    x1: np.ndarray,
    t_span: torch.Tensor,
    current_t: float = 1.0,
    mu1: float = -2.0,
    mu2: float = 2.0,
    sigma: float = 0.5,
    weight: float = 0.5,
    x_range: tuple = (-5, 5),
    n_arrows_x: int = 15,
    n_arrows_t: int = 10,
    n_trajectories: int = 30,
):
    """
    Plot vector field with particle trajectories and edge distributions.

    Args:
        ax: Matplotlib axes
        model: Trained velocity model
        trajectories: Particle trajectories, shape (n_times, n_particles)
        x0: Source samples
        x1: Target samples
        t_span: Time points
        current_t: Current time for highlighting
        mu1, mu2, sigma, weight: Target distribution parameters
        x_range: Range of x values
        n_arrows_x: Number of arrows in x direction
        n_arrows_t: Number of arrows in t direction
        n_trajectories: Number of trajectories to plot
    """
    # Create arrow grid
    x_arrows = np.linspace(x_range[0], x_range[1], n_arrows_x)
    t_arrows = np.linspace(0.05, 0.95, n_arrows_t)
    X, T = np.meshgrid(x_arrows, t_arrows)

    # Compute velocity at arrow positions
    U = np.zeros_like(X)  # dx/dt direction (will be v)
    V = np.ones_like(X)  # dt/dt direction (always 1)

    with torch.no_grad():
        for i, t in enumerate(t_arrows):
            x_tensor = torch.tensor(x_arrows, dtype=torch.float32)
            t_tensor = torch.full_like(x_tensor, t)
            vel = model(x_tensor, t_tensor).numpy()
            U[i, :] = vel

    # Normalize arrows for visualization
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / magnitude * 0.08
    V_norm = V / magnitude * 0.08

    # Plot vector field
    ax.quiver(
        T,
        X,
        V_norm,
        U_norm,
        color=ACCENT_CYAN,
        alpha=0.4,
        width=0.003,
        headwidth=4,
        headlength=5,
    )

    # Plot trajectories (subset)
    n_particles = trajectories.shape[1]
    indices = np.linspace(0, n_particles - 1, min(n_trajectories, n_particles), dtype=int)

    t_np = t_span.numpy()
    traj_np = trajectories.numpy()

    for idx in indices:
        # Color by target position
        color = plt.cm.coolwarm((x1[idx] - x_range[0]) / (x_range[1] - x_range[0]))
        ax.plot(t_np, traj_np[:, idx], color=color, alpha=0.6, linewidth=1)

    # Highlight current time
    if current_t > 0:
        ax.axvline(current_t, color=ACCENT_RED, linestyle="--", alpha=0.7, linewidth=1.5)

        # Plot current positions
        t_idx = int(current_t * (len(t_span) - 1))
        t_idx = min(t_idx, len(t_span) - 1)
        current_pos = traj_np[t_idx, indices]
        ax.scatter(
            [current_t] * len(indices),
            current_pos,
            c=[plt.cm.coolwarm((x1[i] - x_range[0]) / (x_range[1] - x_range[0])) for i in indices],
            s=30,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

    # Plot source distribution on left edge
    x_pdf = np.linspace(x_range[0], x_range[1], 200)
    source_pdf = standard_normal_pdf(x_pdf)
    source_pdf_scaled = source_pdf / source_pdf.max() * 0.15  # Scale for visibility
    ax.fill_betweenx(x_pdf, -source_pdf_scaled, 0, alpha=0.5, color="gray", label="Source N(0,1)")

    # Plot target distribution on right edge
    target_pdf = bimodal_pdf(x_pdf, mu1, mu2, sigma, weight)
    target_pdf_scaled = target_pdf / target_pdf.max() * 0.15
    ax.fill_betweenx(
        x_pdf, 1, 1 + target_pdf_scaled, alpha=0.5, color=ACCENT_RED, label="Target"
    )

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(x_range[0], x_range[1])
    ax.set_xlabel("Time t", fontsize=10)
    ax.set_ylabel("Position x", fontsize=10)
    ax.set_title("Flow Trajectories", fontsize=12, color=TEXT_COLOR)
    ax.legend(loc="upper left", fontsize=8)


def create_two_panel_figure(
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    x0: np.ndarray,
    x1: np.ndarray,
    t_span: torch.Tensor,
    current_t: float = 1.0,
    mu1: float = -2.0,
    mu2: float = 2.0,
    sigma: float = 0.5,
    weight: float = 0.5,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Create the complete two-panel visualization.

    Args:
        model: Trained velocity model
        trajectories: Particle trajectories
        x0: Source samples
        x1: Target samples
        t_span: Time points
        current_t: Current time for animation
        mu1, mu2, sigma, weight: Target distribution parameters
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    set_dark_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)

    # Left panel: Velocity heatmap
    plot_velocity_heatmap(axes[0], model)

    # Right panel: Vector field + trajectories
    plot_vector_field_with_trajectories(
        axes[1],
        model,
        trajectories,
        x0,
        x1,
        t_span,
        current_t=current_t,
        mu1=mu1,
        mu2=mu2,
        sigma=sigma,
        weight=weight,
    )

    plt.tight_layout()
    return fig
