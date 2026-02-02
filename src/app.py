"""Streamlit app for Flow Matching Visualization."""

import streamlit as st
import torch
import numpy as np

from distributions import sample_standard_normal, sample_bimodal
from mlp import VelocityMLP
from flow_matching import train_model
from solver import integrate_flow
from visualization import create_two_panel_figure, set_dark_style
from interactive_viz import (
    create_interactive_flow_plot,
    create_highlighted_flow_plot,
    create_dual_panel_interactive,
    create_linear_connections_plot,
)
from math_panel import get_full_explanation


# Page configuration
st.set_page_config(
    page_title="Flow Matching Visualization",
    page_icon="🌊",
    layout="wide",
)

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a1a2e;
    }
    .stSidebar {
        background-color: #16213e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables."""
    if "model" not in st.session_state:
        st.session_state.model = VelocityMLP()
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "x0" not in st.session_state:
        st.session_state.x0 = None
    if "x1" not in st.session_state:
        st.session_state.x1 = None
    if "trajectories" not in st.session_state:
        st.session_state.trajectories = None
    if "t_span" not in st.session_state:
        st.session_state.t_span = torch.linspace(0, 1, 100)
    if "losses" not in st.session_state:
        st.session_state.losses = []
    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = None
    # Multi-select for Linear Connections mode
    if "selected_indices" not in st.session_state:
        st.session_state.selected_indices = []


def main():
    initialize_session_state()

    # Title
    st.title("🌊 1D Conditional Flow Matching")
    st.markdown("*Interactive visualization of how CFM transports samples from N(0,1) to a bimodal distribution*")

    # Sidebar
    with st.sidebar:
        st.header("Target Distribution")
        mu1 = st.slider("Mode 1 (μ₁)", -5.0, 0.0, -2.0, 0.1)
        mu2 = st.slider("Mode 2 (μ₂)", 0.0, 5.0, 2.0, 0.1)
        sigma = st.slider("Spread (σ)", 0.1, 2.0, 0.5, 0.1)
        weight = st.slider("Mode 1 weight", 0.0, 1.0, 0.5, 0.05)

        st.divider()

        st.header("Sampling")
        n_particles = st.slider("Number of particles", 50, 500, 200, 50)

        if st.button("🎲 Resample"):
            st.session_state.x0 = sample_standard_normal(n_particles)
            st.session_state.x1 = sample_bimodal(n_particles, mu1, mu2, sigma, weight)
            st.session_state.trained = False
            st.session_state.trajectories = None
            st.session_state.selected_idx = None
            st.session_state.selected_indices = []

        st.divider()

        st.header("MLP Training")
        epochs = st.number_input("Epochs", 100, 5000, 1000, 100)
        batch_size = st.select_slider(
            "Batch size",
            options=[32, 64, 128, 256, 512, 1024],
            value=256,
        )
        lr = st.select_slider(
            "Learning rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
        )

        train_button = st.button("🚀 Train MLP", type="primary")

        st.divider()

        st.header("Visualization")
        viz_mode = st.radio(
            "Mode",
            ["Linear Connections", "Flow Trajectories", "Static (Matplotlib)"],
            index=0,
        )

        if viz_mode == "Static (Matplotlib)":
            current_t = st.slider("Time t", 0.0, 1.0, 1.0, 0.01)
        else:
            current_t = 1.0

        if viz_mode == "Flow Trajectories":
            n_neighbors = st.slider("Neighbors to highlight", 1, 20, 5)
        else:
            n_neighbors = 5

    # Initialize samples if not done
    if st.session_state.x0 is None:
        st.session_state.x0 = sample_standard_normal(n_particles)
        st.session_state.x1 = sample_bimodal(n_particles, mu1, mu2, sigma, weight)

    # Training
    if train_button:
        # Reset model
        st.session_state.model = VelocityMLP()
        st.session_state.selected_idx = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(epoch: int, loss: float):
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.6f}")

        with st.spinner("Training MLP..."):
            losses = train_model(
                st.session_state.model,
                st.session_state.x0,
                st.session_state.x1,
                epochs=int(epochs),
                batch_size=batch_size,
                lr=lr,
                progress_callback=progress_callback,
            )

        st.session_state.losses = losses
        st.session_state.trained = True

        # Compute trajectories
        st.session_state.trajectories = integrate_flow(
            st.session_state.model,
            st.session_state.x0,
            st.session_state.t_span,
        )

        progress_bar.empty()
        status_text.empty()
        st.success("Training complete!")
        st.rerun()

    # Main content
    has_samples = st.session_state.x0 is not None and st.session_state.x1 is not None
    has_trajectories = st.session_state.trained and st.session_state.trajectories is not None

    # Linear Connections mode only needs samples, not training
    if viz_mode == "Linear Connections" and has_samples:
        x0_np = st.session_state.x0.numpy()
        x1_np = st.session_state.x1.numpy()

        # Linear connections visualization - no training needed
        st.markdown("### 🎯 Click target points to select (multi-select supported)")

        selected_indices = st.session_state.get("selected_indices", [])

        # Clear selection button
        col_header1, col_header2 = st.columns([3, 1])
        with col_header2:
            if st.button("Clear Selection", disabled=len(selected_indices) == 0):
                st.session_state.selected_indices = []
                st.rerun()

        fig_linear = create_linear_connections_plot(
            x0=x0_np,
            x1=x1_np,
            selected_indices=selected_indices,
            mu1=mu1,
            mu2=mu2,
            sigma=sigma,
            weight=weight,
            height=550,
            line_opacity=0.15,
        )

        event = st.plotly_chart(
            fig_linear,
            use_container_width=True,
            on_select="rerun",
            key="linear_plot",
        )

        # Check if user clicked on a target point - toggle selection
        if event and event.selection and event.selection.points:
            for point in event.selection.points:
                x_val = point.get("x")
                if x_val is not None and abs(float(x_val) - 1.0) < 0.01:
                    point_idx = point.get("point_index")
                    if point_idx is not None:
                        idx = int(point_idx)
                        # Toggle: add if not present, remove if present
                        if idx in st.session_state.selected_indices:
                            st.session_state.selected_indices.remove(idx)
                        else:
                            st.session_state.selected_indices.append(idx)
                        st.rerun()

        if len(selected_indices) > 0:
            st.markdown("---")
            st.markdown(f"**{len(selected_indices)} target(s) selected:** {sorted(selected_indices)}")
            st.caption("Overlapping lines from multiple selections create visible intensity buildup. Cyan lines show actual source-target pairings.")

    elif has_trajectories:
        x0_np = st.session_state.x0.numpy()
        x1_np = st.session_state.x1.numpy()

        if viz_mode == "Flow Trajectories":
            # Interactive trajectory selector
            st.markdown("### 🎯 Click on a target point (right edge) to highlight its trajectory")

            # Get current selection
            selected_idx = st.session_state.get("selected_idx", None)

            # Create the clickable plot with target points
            fig_select = create_interactive_flow_plot(
                trajectories=st.session_state.trajectories,
                x0=x0_np,
                x1=x1_np,
                t_span=st.session_state.t_span,
                mu1=mu1,
                mu2=mu2,
                sigma=sigma,
                weight=weight,
                n_neighbors=n_neighbors,
                height=500,
            )

            # Use on_select to capture clicks on target points
            event = st.plotly_chart(
                fig_select,
                use_container_width=True,
                on_select="rerun",
                key="flow_plot",
            )

            # Check if user clicked on a target point
            if event and event.selection and event.selection.points:
                # Get the clicked point
                for point in event.selection.points:
                    # Check if this is from the target points trace (x ≈ 1.0)
                    x_val = point.get("x")
                    if x_val is not None and abs(float(x_val) - 1.0) < 0.01:
                        # This is a target point click
                        point_idx = point.get("point_index")
                        if point_idx is not None:
                            st.session_state.selected_idx = int(point_idx)
                            selected_idx = int(point_idx)
                            break

            # Show selected trajectory details
            if selected_idx is not None:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.metric("Selected", f"Particle {selected_idx}")
                with col2:
                    st.metric("x₀ (source)", f"{x0_np[selected_idx]:.3f}")
                with col3:
                    st.metric("x₁ (target)", f"{x1_np[selected_idx]:.3f}")

                # Show highlighted trajectory view
                st.markdown("### 🔍 Highlighted Trajectory + Neighbors")
                fig_detail = create_highlighted_flow_plot(
                    trajectories=st.session_state.trajectories,
                    x0=x0_np,
                    x1=x1_np,
                    t_span=st.session_state.t_span,
                    selected_idx=selected_idx,
                    mu1=mu1,
                    mu2=mu2,
                    sigma=sigma,
                    weight=weight,
                    n_neighbors=n_neighbors,
                    height=450,
                )
                st.plotly_chart(fig_detail, use_container_width=True)
            else:
                st.info("👆 Click on any target point (colored dots on the right edge) to highlight its trajectory")

        else:
            # Static matplotlib visualization
            fig = create_two_panel_figure(
                model=st.session_state.model,
                trajectories=st.session_state.trajectories,
                x0=x0_np,
                x1=x1_np,
                t_span=st.session_state.t_span,
                current_t=current_t,
                mu1=mu1,
                mu2=mu2,
                sigma=sigma,
                weight=weight,
            )
            st.pyplot(fig)

        # Loss curve
        if st.session_state.losses:
            with st.expander("📉 Training Loss", expanded=False):
                st.line_chart(st.session_state.losses)

    elif viz_mode in ["Flow Trajectories", "Static (Matplotlib)"] and not has_trajectories:
        st.info("👆 Click **Train MLP** to learn the flow and visualize trajectories!")

        # Show a placeholder with just the distributions
        set_dark_style()
        import matplotlib.pyplot as plt
        from distributions import bimodal_pdf, standard_normal_pdf

        fig, ax = plt.subplots(figsize=(10, 4))
        x_plot = np.linspace(-6, 6, 200)

        source_pdf = standard_normal_pdf(x_plot)
        target_pdf = bimodal_pdf(x_plot, mu1, mu2, sigma, weight)

        ax.fill_between(x_plot, source_pdf, alpha=0.5, label="Source N(0,1)", color="gray")
        ax.fill_between(x_plot, target_pdf, alpha=0.5, label="Target (bimodal)", color="#e94560")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.set_title("Source and Target Distributions")
        ax.legend()

        st.pyplot(fig)

    else:
        st.info("👆 Configure the target distribution to start!")

    # Math explanations (collapsible)
    with st.expander("📚 Mathematical Background"):
        st.markdown(get_full_explanation())


if __name__ == "__main__":
    main()
