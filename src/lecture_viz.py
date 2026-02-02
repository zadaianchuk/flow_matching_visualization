"""Lecture step visualizations for CFM walkthrough."""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from distributions import bimodal_pdf, standard_normal_pdf


# Dark theme colors
DARK_BG = "#1a1a2e"
DARK_PANEL = "#16213e"
ACCENT_RED = "#e94560"
ACCENT_CYAN = "#00d9ff"
ACCENT_GREEN = "#4ade80"
TEXT_COLOR = "#eaeaea"


def create_step1_distributions(
    x0: np.ndarray,
    x1: np.ndarray,
    mu1: float,
    mu2: float,
    sigma: float,
    weight: float,
    height: int = 450,
) -> go.Figure:
    """Step 1: Show source and target distributions side by side."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Source: N(0,1)", "Target: Bimodal"),
        horizontal_spacing=0.1,
    )

    x_range = np.linspace(-5, 5, 200)

    # Source distribution
    source_pdf = standard_normal_pdf(x_range)
    fig.add_trace(
        go.Scatter(
            x=x_range, y=source_pdf,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(150, 150, 150, 0.4)',
            line=dict(color='gray', width=2),
            name='Source N(0,1)',
        ),
        row=1, col=1
    )

    # Source samples
    fig.add_trace(
        go.Scatter(
            x=x0, y=np.zeros(len(x0)) - 0.02,
            mode='markers',
            marker=dict(size=6, color='gray', opacity=0.6),
            name='Source samples',
        ),
        row=1, col=1
    )

    # Target distribution
    target_pdf = bimodal_pdf(x_range, mu1, mu2, sigma, weight)
    fig.add_trace(
        go.Scatter(
            x=x_range, y=target_pdf,
            mode='lines',
            fill='tozeroy',
            fillcolor=f'rgba(233, 69, 96, 0.4)',
            line=dict(color=ACCENT_RED, width=2),
            name='Target (bimodal)',
        ),
        row=1, col=2
    )

    # Target samples
    fig.add_trace(
        go.Scatter(
            x=x1, y=np.zeros(len(x1)) - 0.02,
            mode='markers',
            marker=dict(size=6, color=ACCENT_RED, opacity=0.6),
            name='Target samples',
        ),
        row=1, col=2
    )

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        height=height,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    for i in [1, 2]:
        fig.update_xaxes(
            title_text='x', range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
            row=1, col=i
        )
        fig.update_yaxes(
            title_text='Density',
            gridcolor='rgba(100, 100, 150, 0.2)',
            row=1, col=i
        )

    for annotation in fig.layout.annotations:
        annotation.font.color = TEXT_COLOR

    return fig


def create_step2_conditional_pair(
    x0: np.ndarray,
    x1: np.ndarray,
    pair_idx: int,
    mu1: float,
    mu2: float,
    sigma: float,
    weight: float,
    height: int = 450,
) -> go.Figure:
    """Step 2: Show one highlighted conditional pair."""
    fig = go.Figure()

    n_particles = len(x0)

    # Draw all connections faintly
    for i in range(n_particles):
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[x0[i], x1[i]],
            mode='lines',
            line=dict(color='rgba(100, 100, 150, 0.1)', width=1),
            hoverinfo='skip',
            showlegend=False,
        ))

    # Highlight the selected pair
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[x0[pair_idx], x1[pair_idx]],
        mode='lines+markers',
        line=dict(color=ACCENT_CYAN, width=4),
        marker=dict(size=15, color=ACCENT_CYAN, line=dict(color='white', width=2)),
        name=f'Pair {pair_idx}: x₀={x0[pair_idx]:.2f} → x₁={x1[pair_idx]:.2f}',
    ))

    # Add distributions on edges
    x_pdf = np.linspace(-5, 5, 200)
    source_pdf = standard_normal_pdf(x_pdf)
    source_pdf_scaled = -source_pdf / source_pdf.max() * 0.1

    fig.add_trace(go.Scatter(
        x=source_pdf_scaled, y=x_pdf,
        mode='lines', fill='tozerox',
        fillcolor='rgba(150, 150, 150, 0.3)',
        line=dict(color='gray', width=1),
        name='Source',
        hoverinfo='skip',
    ))

    target_pdf = bimodal_pdf(x_pdf, mu1, mu2, sigma, weight)
    target_pdf_scaled = target_pdf / target_pdf.max() * 0.1 + 1

    fig.add_trace(go.Scatter(
        x=target_pdf_scaled, y=x_pdf,
        mode='lines', fill='tonextx',
        fillcolor='rgba(233, 69, 96, 0.3)',
        line=dict(color=ACCENT_RED, width=1),
        name='Target',
        hoverinfo='skip',
    ))

    # All source points
    fig.add_trace(go.Scatter(
        x=np.zeros(n_particles), y=x0,
        mode='markers',
        marker=dict(size=6, color='gray', opacity=0.5),
        name='Source points',
        hoverinfo='skip',
    ))

    # All target points
    fig.add_trace(go.Scatter(
        x=np.ones(n_particles), y=x1,
        mode='markers',
        marker=dict(size=6, color=ACCENT_RED, opacity=0.5),
        name='Target points',
        hoverinfo='skip',
    ))

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            title='Time t', range=[-0.15, 1.15],
            gridcolor='rgba(100, 100, 150, 0.2)',
            tickvals=[0, 1], ticktext=['t=0 (source)', 't=1 (target)'],
        ),
        yaxis=dict(
            title='Position x', range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        height=height,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)'),
    )

    return fig


def create_step3_path_animation(
    x0: np.ndarray,
    x1: np.ndarray,
    pair_idx: int,
    t: float,
    mu1: float,
    mu2: float,
    sigma: float,
    weight: float,
    height: int = 450,
) -> go.Figure:
    """Step 3: Show particle moving along conditional path."""
    fig = go.Figure()

    source_val = x0[pair_idx]
    target_val = x1[pair_idx]

    # Current position
    x_t = (1 - t) * source_val + t * target_val

    # The path line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[source_val, target_val],
        mode='lines',
        line=dict(color=ACCENT_CYAN, width=3, dash='dash'),
        name='Path: xₜ = (1-t)x₀ + tx₁',
    ))

    # Start and end markers
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[source_val, target_val],
        mode='markers',
        marker=dict(size=12, color=[ACCENT_CYAN, ACCENT_RED],
                   line=dict(color='white', width=2)),
        name='Start/End',
        hovertemplate=['x₀ = %{y:.3f}', 'x₁ = %{y:.3f}'],
    ))

    # Current position (animated point)
    fig.add_trace(go.Scatter(
        x=[t],
        y=[x_t],
        mode='markers',
        marker=dict(size=20, color=ACCENT_RED,
                   line=dict(color='white', width=3)),
        name=f'Current: xₜ = {x_t:.3f}',
    ))

    # Vertical line showing current t
    fig.add_vline(x=t, line_dash="dot", line_color=ACCENT_RED, opacity=0.5)

    # Add formula annotation
    fig.add_annotation(
        x=0.5, y=4.5,
        text=f"xₜ = (1-{t:.2f})×{source_val:.2f} + {t:.2f}×{target_val:.2f} = {x_t:.3f}",
        showarrow=False,
        font=dict(size=14, color=TEXT_COLOR),
        bgcolor='rgba(0,0,0,0.5)',
        borderpad=4,
    )

    # Add distributions on edges
    x_pdf = np.linspace(-5, 5, 200)
    source_pdf = standard_normal_pdf(x_pdf)
    source_pdf_scaled = -source_pdf / source_pdf.max() * 0.1

    fig.add_trace(go.Scatter(
        x=source_pdf_scaled, y=x_pdf,
        mode='lines', fill='tozerox',
        fillcolor='rgba(150, 150, 150, 0.3)',
        line=dict(color='gray', width=1),
        hoverinfo='skip', showlegend=False,
    ))

    target_pdf = bimodal_pdf(x_pdf, mu1, mu2, sigma, weight)
    target_pdf_scaled = target_pdf / target_pdf.max() * 0.1 + 1

    fig.add_trace(go.Scatter(
        x=target_pdf_scaled, y=x_pdf,
        mode='lines', fill='tonextx',
        fillcolor='rgba(233, 69, 96, 0.3)',
        line=dict(color=ACCENT_RED, width=1),
        hoverinfo='skip', showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            title='Time t', range=[-0.15, 1.15],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        yaxis=dict(
            title='Position x', range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        height=height,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)'),
    )

    return fig


def create_step4_constant_velocity(
    x0: np.ndarray,
    x1: np.ndarray,
    pair_idx: int,
    height: int = 450,
) -> go.Figure:
    """Step 4: Show constant velocity uₜ = x₁ - x₀."""
    fig = go.Figure()

    source_val = x0[pair_idx]
    target_val = x1[pair_idx]
    velocity = target_val - source_val

    # The path
    t_vals = np.linspace(0, 1, 50)
    x_vals = (1 - t_vals) * source_val + t_vals * target_val

    fig.add_trace(go.Scatter(
        x=t_vals, y=x_vals,
        mode='lines',
        line=dict(color=ACCENT_CYAN, width=3),
        name='Path xₜ',
    ))

    # Velocity arrows at multiple time points
    arrow_times = [0.1, 0.3, 0.5, 0.7, 0.9]
    for t in arrow_times:
        x_pos = (1 - t) * source_val + t * target_val
        # Arrow showing velocity direction
        fig.add_annotation(
            x=t, y=x_pos,
            ax=t, ay=x_pos - velocity * 0.15,  # Arrow tail
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor=ACCENT_GREEN,
        )

    # Markers at arrow positions
    arrow_x_vals = [(1 - t) * source_val + t * target_val for t in arrow_times]
    fig.add_trace(go.Scatter(
        x=arrow_times, y=arrow_x_vals,
        mode='markers',
        marker=dict(size=10, color=ACCENT_GREEN),
        name=f'Velocity uₜ = {velocity:.3f}',
    ))

    # Add velocity annotation
    fig.add_annotation(
        x=0.5, y=4.5,
        text=f"uₜ = x₁ - x₀ = {target_val:.2f} - {source_val:.2f} = {velocity:.3f} (CONSTANT!)",
        showarrow=False,
        font=dict(size=14, color=ACCENT_GREEN),
        bgcolor='rgba(0,0,0,0.5)',
        borderpad=4,
    )

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            title='Time t', range=[-0.05, 1.05],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        yaxis=dict(
            title='Position x', range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        height=height,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)'),
    )

    return fig


def create_step5_training_samples(
    x0: np.ndarray,
    x1: np.ndarray,
    n_samples: int = 500,
    height: int = 450,
) -> go.Figure:
    """Step 5: Show scatter of training data (xₜ, t) → uₜ."""
    # Generate training samples
    n_pairs = len(x0)
    indices = np.random.randint(0, n_pairs, n_samples)
    t_samples = np.random.rand(n_samples)

    x0_samples = x0[indices]
    x1_samples = x1[indices]

    # Compute xₜ and uₜ
    xt_samples = (1 - t_samples) * x0_samples + t_samples * x1_samples
    ut_samples = x1_samples - x0_samples

    fig = go.Figure()

    # Scatter plot colored by velocity
    fig.add_trace(go.Scatter(
        x=xt_samples,
        y=t_samples,
        mode='markers',
        marker=dict(
            size=6,
            color=ut_samples,
            colorscale='RdBu_r',
            cmin=-np.abs(ut_samples).max(),
            cmax=np.abs(ut_samples).max(),
            colorbar=dict(title='Target velocity uₜ'),
            opacity=0.7,
        ),
        hovertemplate='xₜ=%{x:.2f}<br>t=%{y:.2f}<br>uₜ=%{marker.color:.2f}<extra></extra>',
        name='Training samples',
    ))

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            title='Position xₜ', range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        yaxis=dict(
            title='Time t', range=[0, 1],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        height=height,
        margin=dict(l=50, r=100, t=30, b=50),
    )

    return fig


def create_step6_velocity_evolution(
    model: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    epoch: int,
    loss: float,
    height: int = 450,
) -> go.Figure:
    """Step 6: Show velocity field at current training epoch."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Learned v(x,t) at Epoch {epoch}', 'Target Velocities (sample)'),
        horizontal_spacing=0.12,
    )

    # Left: Learned velocity field heatmap
    x_vals = np.linspace(-4, 4, 60)
    t_vals = np.linspace(0, 1, 40)
    X, T = np.meshgrid(x_vals, t_vals)

    V = np.zeros_like(X)
    with torch.no_grad():
        for i, t in enumerate(t_vals):
            x_tensor = torch.tensor(x_vals, dtype=torch.float32)
            t_tensor = torch.full_like(x_tensor, t)
            v = model(x_tensor, t_tensor).numpy()
            V[i, :] = v

    vmax = max(np.abs(V).max(), 0.1)

    fig.add_trace(
        go.Heatmap(
            x=x_vals, y=t_vals, z=V,
            colorscale='RdBu_r',
            zmin=-vmax, zmax=vmax,
            colorbar=dict(title='v(x,t)', x=0.45),
        ),
        row=1, col=1
    )

    # Right: Sample of target velocities
    n_show = min(100, len(x0))
    indices = np.random.choice(len(x0), n_show, replace=False)
    t_samples = np.random.rand(n_show)
    xt_samples = (1 - t_samples) * x0[indices] + t_samples * x1[indices]
    ut_samples = x1[indices] - x0[indices]

    fig.add_trace(
        go.Scatter(
            x=xt_samples, y=t_samples,
            mode='markers',
            marker=dict(
                size=8,
                color=ut_samples,
                colorscale='RdBu_r',
                cmin=-vmax, cmax=vmax,
                colorbar=dict(title='uₜ', x=1.0),
            ),
            name='Target velocities',
        ),
        row=1, col=2
    )

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        height=height,
        margin=dict(l=50, r=80, t=50, b=50),
        showlegend=False,
    )

    fig.update_xaxes(title_text='Position x', gridcolor='rgba(100,100,150,0.2)', row=1, col=1)
    fig.update_yaxes(title_text='Time t', gridcolor='rgba(100,100,150,0.2)', row=1, col=1)
    fig.update_xaxes(title_text='Position xₜ', gridcolor='rgba(100,100,150,0.2)', range=[-4, 4], row=1, col=2)
    fig.update_yaxes(title_text='Time t', gridcolor='rgba(100,100,150,0.2)', row=1, col=2)

    for annotation in fig.layout.annotations:
        annotation.font.color = TEXT_COLOR

    return fig


def create_step7_final_flow(
    trajectories: torch.Tensor,
    x0: np.ndarray,
    x1: np.ndarray,
    t_span: torch.Tensor,
    mu1: float,
    mu2: float,
    sigma: float,
    weight: float,
    height: int = 450,
) -> go.Figure:
    """Step 7: Show final integrated trajectories."""
    fig = go.Figure()

    t_np = t_span.numpy()
    traj_np = trajectories.numpy()
    n_particles = traj_np.shape[1]

    # Draw all trajectories
    for i in range(n_particles):
        color_val = (x1[i] - x1.min()) / (x1.max() - x1.min() + 1e-8)
        color = f'rgba({int(100 + 155*color_val)}, {int(100*(1-color_val))}, {int(200*(1-color_val))}, 0.5)'
        fig.add_trace(go.Scatter(
            x=t_np, y=traj_np[:, i],
            mode='lines',
            line=dict(color=color, width=1.5),
            hoverinfo='skip',
            showlegend=False,
        ))

    # Add distributions on edges
    x_pdf = np.linspace(-5, 5, 200)
    source_pdf = standard_normal_pdf(x_pdf)
    source_pdf_scaled = -source_pdf / source_pdf.max() * 0.1

    fig.add_trace(go.Scatter(
        x=source_pdf_scaled, y=x_pdf,
        mode='lines', fill='tozerox',
        fillcolor='rgba(150, 150, 150, 0.4)',
        line=dict(color='gray', width=2),
        name='Source N(0,1)',
    ))

    target_pdf = bimodal_pdf(x_pdf, mu1, mu2, sigma, weight)
    target_pdf_scaled = target_pdf / target_pdf.max() * 0.1 + 1

    fig.add_trace(go.Scatter(
        x=target_pdf_scaled, y=x_pdf,
        mode='lines', fill='tonextx',
        fillcolor='rgba(233, 69, 96, 0.4)',
        line=dict(color=ACCENT_RED, width=2),
        name='Target (bimodal)',
    ))

    # Start and end points
    fig.add_trace(go.Scatter(
        x=np.zeros(n_particles), y=x0,
        mode='markers',
        marker=dict(size=6, color='gray', opacity=0.7),
        name='Start positions',
    ))

    fig.add_trace(go.Scatter(
        x=np.ones(n_particles), y=traj_np[-1, :],
        mode='markers',
        marker=dict(size=6, color=ACCENT_RED, opacity=0.7),
        name='Final positions',
    ))

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            title='Time t', range=[-0.15, 1.15],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        yaxis=dict(
            title='Position x', range=[-5, 5],
            gridcolor='rgba(100, 100, 150, 0.2)',
        ),
        height=height,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(x=0.3, y=0.98, bgcolor='rgba(0,0,0,0.5)'),
    )

    return fig
