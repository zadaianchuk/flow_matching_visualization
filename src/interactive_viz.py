"""Interactive Plotly visualization with hover highlighting."""

from typing import List, Union

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist

from distributions import bimodal_pdf, standard_normal_pdf


# Dark theme colors
DARK_BG = "#1a1a2e"
DARK_PANEL = "#16213e"
ACCENT_RED = "#e94560"
ACCENT_CYAN = "#00d9ff"
TEXT_COLOR = "#eaeaea"


def find_k_nearest_neighbors(x1: np.ndarray, idx: int, k: int = 5) -> np.ndarray:
    """Find k nearest neighbors of point idx in x1."""
    target_val = x1[idx]
    distances = np.abs(x1 - target_val)
    # Get indices of k smallest distances (excluding self)
    neighbor_indices = np.argsort(distances)[:k+1]
    return neighbor_indices


def create_interactive_flow_plot(
    trajectories: torch.Tensor,
    x0: np.ndarray,
    x1: np.ndarray,
    t_span: torch.Tensor,
    mu1: float = -2.0,
    mu2: float = 2.0,
    sigma: float = 0.5,
    weight: float = 0.5,
    n_neighbors: int = 5,
    height: int = 600,
) -> go.Figure:
    """
    Create interactive Plotly figure with hover highlighting.

    Hover over target distribution points to highlight corresponding
    trajectories and their nearest neighbors.

    Args:
        trajectories: Particle trajectories, shape (n_times, n_particles)
        x0: Source samples
        x1: Target samples
        t_span: Time points
        mu1, mu2, sigma, weight: Target distribution parameters
        n_neighbors: Number of neighboring trajectories to highlight
        height: Figure height in pixels

    Returns:
        Plotly figure with hover interactivity
    """
    t_np = t_span.numpy()
    traj_np = trajectories.numpy()
    n_particles = traj_np.shape[1]

    fig = go.Figure()

    # Add all trajectories as faint lines (background)
    for i in range(n_particles):
        fig.add_trace(go.Scatter(
            x=t_np,
            y=traj_np[:, i],
            mode='lines',
            line=dict(color='rgba(100, 100, 150, 0.15)', width=1),
            hoverinfo='skip',
            showlegend=False,
            name=f'traj_{i}',
        ))

    # Add source distribution on left edge
    x_pdf = np.linspace(-5, 5, 200)
    source_pdf = standard_normal_pdf(x_pdf)
    source_pdf_scaled = -source_pdf / source_pdf.max() * 0.12

    fig.add_trace(go.Scatter(
        x=source_pdf_scaled,
        y=x_pdf,
        mode='lines',
        fill='tozerox',
        fillcolor='rgba(150, 150, 150, 0.4)',
        line=dict(color='rgba(150, 150, 150, 0.8)', width=1),
        name='Source N(0,1)',
        hoverinfo='skip',
    ))

    # Add target distribution on right edge
    target_pdf = bimodal_pdf(x_pdf, mu1, mu2, sigma, weight)
    target_pdf_scaled = target_pdf / target_pdf.max() * 0.12 + 1

    fig.add_trace(go.Scatter(
        x=target_pdf_scaled,
        y=x_pdf,
        mode='lines',
        fill='tonextx',
        fillcolor=f'rgba(233, 69, 96, 0.4)',
        line=dict(color=ACCENT_RED, width=1),
        name='Target',
        hoverinfo='skip',
    ))

    # Precompute neighbor groups for each particle
    neighbor_groups = {}
    for i in range(n_particles):
        neighbor_groups[i] = find_k_nearest_neighbors(x1, i, n_neighbors)

    # Add CLICKABLE target points - larger and more prominent
    # Color by position for visual appeal
    colors = (x1 - x1.min()) / (x1.max() - x1.min() + 1e-8)

    # Include both index and source value in customdata
    customdata = np.column_stack([np.arange(n_particles), x0])

    fig.add_trace(go.Scatter(
        x=np.ones(n_particles),
        y=x1,
        mode='markers',
        marker=dict(
            size=14,  # Larger for easier clicking
            color=colors,
            colorscale='RdBu_r',
            line=dict(color='white', width=2),
            opacity=0.9,
        ),
        customdata=customdata,
        hovertemplate=(
            '<b>Particle %{customdata[0]:.0f}</b><br>'
            'x₀ (source): %{customdata[1]:.3f}<br>'
            'x₁ (target): %{y:.3f}<br>'
            '<extra>Click to highlight</extra>'
        ),
        name='Target points (click me!)',
        showlegend=True,
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Flow Trajectories</b> - 👆 CLICK on target points (right edge) to highlight',
            font=dict(color=TEXT_COLOR, size=16),
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            title='Time t',
            range=[-0.2, 1.2],
            gridcolor='rgba(100, 100, 150, 0.2)',
            zerolinecolor='rgba(100, 100, 150, 0.3)',
        ),
        yaxis=dict(
            title='Position x',
            range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
            zerolinecolor='rgba(100, 100, 150, 0.3)',
        ),
        hovermode='closest',
        clickmode='event+select',  # Enable click selection
        dragmode='pan',  # Pan by default, use box select from toolbar
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
        ),
    )

    return fig


def create_highlighted_flow_plot(
    trajectories: torch.Tensor,
    x0: np.ndarray,
    x1: np.ndarray,
    t_span: torch.Tensor,
    selected_idx: int,
    mu1: float = -2.0,
    mu2: float = 2.0,
    sigma: float = 0.5,
    weight: float = 0.5,
    n_neighbors: int = 5,
    height: int = 600,
) -> go.Figure:
    """
    Create plot with a specific trajectory and neighbors highlighted.

    Args:
        trajectories: Particle trajectories
        x0, x1: Source and target samples
        t_span: Time points
        selected_idx: Index of selected trajectory to highlight
        mu1, mu2, sigma, weight: Target distribution parameters
        n_neighbors: Number of neighbors to highlight
        height: Figure height

    Returns:
        Plotly figure with highlighting
    """
    t_np = t_span.numpy()
    traj_np = trajectories.numpy()
    n_particles = traj_np.shape[1]

    # Find neighbors
    neighbors = find_k_nearest_neighbors(x1, selected_idx, n_neighbors)

    fig = go.Figure()

    # Add all trajectories as very faint lines
    for i in range(n_particles):
        if i in neighbors:
            continue  # Will add highlighted version later
        fig.add_trace(go.Scatter(
            x=t_np,
            y=traj_np[:, i],
            mode='lines',
            line=dict(color='rgba(100, 100, 150, 0.1)', width=1),
            hoverinfo='skip',
            showlegend=False,
        ))

    # Add highlighted neighbor trajectories
    for i in neighbors:
        if i == selected_idx:
            continue  # Main selection handled separately
        fig.add_trace(go.Scatter(
            x=t_np,
            y=traj_np[:, i],
            mode='lines',
            line=dict(color=ACCENT_CYAN, width=2),
            opacity=0.6,
            hoverinfo='skip',
            showlegend=False,
            name=f'Neighbor {i}',
        ))

    # Add main selected trajectory (most prominent)
    fig.add_trace(go.Scatter(
        x=t_np,
        y=traj_np[:, selected_idx],
        mode='lines',
        line=dict(color=ACCENT_RED, width=3),
        name=f'Selected (x₁={x1[selected_idx]:.2f})',
        hovertemplate='t=%{x:.2f}<br>x=%{y:.3f}<extra></extra>',
    ))

    # Add source and target points for selected trajectory
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[x0[selected_idx], x1[selected_idx]],
        mode='markers',
        marker=dict(size=15, color=[ACCENT_CYAN, ACCENT_RED],
                   line=dict(color='white', width=2)),
        name='Start/End',
        hovertemplate='%{y:.3f}<extra></extra>',
    ))

    # Add distributions
    x_pdf = np.linspace(-5, 5, 200)
    source_pdf = standard_normal_pdf(x_pdf)
    source_pdf_scaled = -source_pdf / source_pdf.max() * 0.12

    fig.add_trace(go.Scatter(
        x=source_pdf_scaled,
        y=x_pdf,
        mode='lines',
        fill='tozerox',
        fillcolor='rgba(150, 150, 150, 0.3)',
        line=dict(color='rgba(150, 150, 150, 0.6)', width=1),
        name='Source N(0,1)',
        hoverinfo='skip',
    ))

    target_pdf = bimodal_pdf(x_pdf, mu1, mu2, sigma, weight)
    target_pdf_scaled = target_pdf / target_pdf.max() * 0.12 + 1

    fig.add_trace(go.Scatter(
        x=target_pdf_scaled,
        y=x_pdf,
        mode='lines',
        fill='tonextx',
        fillcolor='rgba(233, 69, 96, 0.3)',
        line=dict(color=ACCENT_RED, width=1),
        name='Target',
        hoverinfo='skip',
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>Trajectory {selected_idx}</b>: x₀={x0[selected_idx]:.3f} → x₁={x1[selected_idx]:.3f}',
            font=dict(color=TEXT_COLOR, size=16),
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            title='Time t',
            range=[-0.2, 1.2],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        yaxis=dict(
            title='Position x',
            range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        hovermode='closest',
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
        ),
    )

    return fig


def create_linear_connections_plot(
    x0: np.ndarray,
    x1: np.ndarray,
    selected_indices: Union[List[int], None] = None,
    mu1: float = -2.0,
    mu2: float = 2.0,
    sigma: float = 0.5,
    weight: float = 0.5,
    height: int = 600,
    line_opacity: float = 0.15,
) -> go.Figure:
    """
    Create interactive plot showing linear connections from targets to all sources.

    When target points are clicked, draws straight lines from each selected target
    to ALL source points. Lines have low opacity (0.3) so overlapping lines from
    multiple selections create visible intensity buildup.

    Args:
        x0: Source samples (at t=0)
        x1: Target samples (at t=1)
        selected_indices: List of selected target indices (supports multi-select)
        mu1, mu2, sigma, weight: Target distribution parameters
        height: Figure height in pixels
        line_opacity: Opacity for connection lines (default 0.3 for overlap visibility)

    Returns:
        Plotly figure with clickable target points and linear connections
    """
    n_particles = len(x0)

    fig = go.Figure()

    # Handle both single index and list of indices
    if selected_indices is None:
        selected_indices = []
    elif isinstance(selected_indices, int):
        selected_indices = [selected_indices]

    # Draw linear lines for each selected target to ALL source points
    for sel_idx in selected_indices:
        target_y = x1[sel_idx]

        # Draw lines from selected target to all source points
        for i in range(n_particles):
            source_y = x0[i]

            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[source_y, target_y],
                mode='lines',
                line=dict(
                    color=f'rgba(233, 69, 96, {line_opacity})',
                    width=1.5,
                ),
                hoverinfo='skip',
                showlegend=False,
            ))

    # Highlight the actual paired source points (cyan lines)
    for sel_idx in selected_indices:
        actual_source = x0[sel_idx]
        target_y = x1[sel_idx]
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[actual_source, target_y],
            mode='lines+markers',
            line=dict(color=ACCENT_CYAN, width=3),
            marker=dict(size=10, color=ACCENT_CYAN, line=dict(color='white', width=2)),
            name=f'Coupling {sel_idx}',
            hovertemplate=f'Target {sel_idx}<br>Source: %{{y:.3f}}<extra></extra>',
            showlegend=len(selected_indices) <= 5,  # Hide legend if too many selections
        ))

    # Add source distribution on left edge
    x_pdf = np.linspace(-5, 5, 200)
    source_pdf = standard_normal_pdf(x_pdf)
    source_pdf_scaled = -source_pdf / source_pdf.max() * 0.12

    fig.add_trace(go.Scatter(
        x=source_pdf_scaled,
        y=x_pdf,
        mode='lines',
        fill='tozerox',
        fillcolor='rgba(150, 150, 150, 0.4)',
        line=dict(color='rgba(150, 150, 150, 0.8)', width=1),
        name='Source N(0,1)',
        hoverinfo='skip',
    ))

    # Add target distribution on right edge
    target_pdf = bimodal_pdf(x_pdf, mu1, mu2, sigma, weight)
    target_pdf_scaled = target_pdf / target_pdf.max() * 0.12 + 1

    fig.add_trace(go.Scatter(
        x=target_pdf_scaled,
        y=x_pdf,
        mode='lines',
        fill='tonextx',
        fillcolor=f'rgba(233, 69, 96, 0.4)',
        line=dict(color=ACCENT_RED, width=1),
        name='Target',
        hoverinfo='skip',
    ))

    # Add source points (at t=0)
    fig.add_trace(go.Scatter(
        x=np.zeros(n_particles),
        y=x0,
        mode='markers',
        marker=dict(
            size=8,
            color='rgba(150, 150, 150, 0.7)',
            line=dict(color='white', width=1),
        ),
        name='Source points',
        hovertemplate='Source %{pointNumber}<br>x₀=%{y:.3f}<extra></extra>',
    ))

    # Add CLICKABLE target points (at t=1)
    colors = (x1 - x1.min()) / (x1.max() - x1.min() + 1e-8)
    customdata = np.column_stack([np.arange(n_particles), x0])

    # Highlight selected points differently
    marker_sizes = np.full(n_particles, 12, dtype=float)
    marker_line_widths = np.full(n_particles, 1.5, dtype=float)
    for sel_idx in selected_indices:
        marker_sizes[sel_idx] = 18
        marker_line_widths[sel_idx] = 3

    fig.add_trace(go.Scatter(
        x=np.ones(n_particles),
        y=x1,
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=colors,
            colorscale='RdBu_r',
            line=dict(color='white', width=marker_line_widths),
            opacity=0.9,
        ),
        customdata=customdata,
        hovertemplate=(
            '<b>Target %{customdata[0]:.0f}</b><br>'
            'Paired source: %{customdata[1]:.3f}<br>'
            'Target x₁: %{y:.3f}<br>'
            '<extra>Click to toggle selection</extra>'
        ),
        name='Target points (click to select)',
        showlegend=True,
    ))

    # Layout
    n_selected = len(selected_indices)
    if n_selected == 0:
        title_text = '<b>Linear Connections</b> - 👆 Click target points to see connections (multi-select supported)'
    elif n_selected == 1:
        title_text = f'<b>1 target selected</b>: showing {n_particles} connections (cyan = actual pairing)'
    else:
        title_text = f'<b>{n_selected} targets selected</b>: overlapping lines show common source regions'

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(color=TEXT_COLOR, size=16),
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            title='Time t',
            range=[-0.2, 1.2],
            gridcolor='rgba(100, 100, 150, 0.2)',
            zerolinecolor='rgba(100, 100, 150, 0.3)',
        ),
        yaxis=dict(
            title='Position x',
            range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
            zerolinecolor='rgba(100, 100, 150, 0.3)',
        ),
        hovermode='closest',
        clickmode='event+select',
        dragmode='pan',
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
        ),
    )

    return fig


def create_dual_panel_interactive(
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    x0: np.ndarray,
    x1: np.ndarray,
    t_span: torch.Tensor,
    selected_idx: int = None,
    mu1: float = -2.0,
    mu2: float = 2.0,
    sigma: float = 0.5,
    weight: float = 0.5,
    n_neighbors: int = 5,
    height: int = 500,
) -> go.Figure:
    """
    Create dual-panel figure: velocity heatmap + interactive trajectories.

    Args:
        model: Velocity field model
        trajectories: Particle trajectories
        x0, x1: Source and target samples
        t_span: Time points
        selected_idx: If provided, highlight this trajectory
        mu1, mu2, sigma, weight: Target distribution parameters
        n_neighbors: Number of neighbors to highlight
        height: Figure height

    Returns:
        Plotly figure with two panels
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Velocity Field v(x, t)', 'Flow Trajectories'),
        horizontal_spacing=0.1,
    )

    # Left panel: Velocity heatmap
    x_vals = np.linspace(-5, 5, 80)
    t_vals = np.linspace(0, 1, 40)
    X, T = np.meshgrid(x_vals, t_vals)

    V = np.zeros_like(X)
    with torch.no_grad():
        for i, t in enumerate(t_vals):
            x_tensor = torch.tensor(x_vals, dtype=torch.float32)
            t_tensor = torch.full_like(x_tensor, t)
            v = model(x_tensor, t_tensor).numpy()
            V[i, :] = v

    vmax = np.abs(V).max()

    fig.add_trace(
        go.Heatmap(
            x=x_vals,
            y=t_vals,
            z=V,
            colorscale='RdBu_r',
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(title='Velocity', x=0.45),
            hovertemplate='x=%{x:.2f}<br>t=%{y:.2f}<br>v=%{z:.3f}<extra></extra>',
        ),
        row=1, col=1
    )

    # Right panel: Trajectories
    t_np = t_span.numpy()
    traj_np = trajectories.numpy()
    n_particles = traj_np.shape[1]

    neighbors = set()
    if selected_idx is not None:
        neighbors = set(find_k_nearest_neighbors(x1, selected_idx, n_neighbors))

    # Background trajectories
    for i in range(n_particles):
        is_neighbor = i in neighbors
        is_selected = i == selected_idx

        if is_selected:
            color = ACCENT_RED
            width = 3
            opacity = 1.0
        elif is_neighbor:
            color = ACCENT_CYAN
            width = 2
            opacity = 0.7
        else:
            color = 'rgba(100, 100, 150, 0.15)'
            width = 1
            opacity = 1.0

        fig.add_trace(
            go.Scatter(
                x=t_np,
                y=traj_np[:, i],
                mode='lines',
                line=dict(color=color, width=width),
                opacity=opacity,
                hoverinfo='skip' if not (is_selected or is_neighbor) else 'all',
                showlegend=False,
                hovertemplate=f'Particle {i}<br>t=%{{x:.2f}}<br>x=%{{y:.3f}}<extra></extra>' if (is_selected or is_neighbor) else None,
            ),
            row=1, col=2
        )

    # Target points (hoverable)
    colors = (x1 - x1.min()) / (x1.max() - x1.min() + 1e-8)

    fig.add_trace(
        go.Scatter(
            x=np.ones(n_particles),
            y=x1,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale='RdBu_r',
                line=dict(color='white', width=0.5),
            ),
            customdata=np.stack([np.arange(n_particles), x0], axis=1),
            hovertemplate=(
                '<b>Particle %{customdata[0]:.0f}</b><br>'
                'x₀=%{customdata[1]:.3f}<br>'
                'x₁=%{y:.3f}<br>'
                '<extra></extra>'
            ),
            showlegend=False,
        ),
        row=1, col=2
    )

    # Layout
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode='closest',
    )

    fig.update_xaxes(title_text='Position x', row=1, col=1, gridcolor='rgba(100,100,150,0.2)')
    fig.update_yaxes(title_text='Time t', row=1, col=1, gridcolor='rgba(100,100,150,0.2)')
    fig.update_xaxes(title_text='Time t', range=[-0.1, 1.1], row=1, col=2, gridcolor='rgba(100,100,150,0.2)')
    fig.update_yaxes(title_text='Position x', range=[-5, 5], row=1, col=2, gridcolor='rgba(100,100,150,0.2)')

    # Update subplot title colors
    for annotation in fig.layout.annotations:
        annotation.font.color = TEXT_COLOR

    return fig
