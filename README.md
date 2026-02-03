# Flow Matching Visualization

An interactive educational tool for visualizing 1D Conditional Flow Matching (CFM) - a generative modeling technique that learns to transport samples from a simple distribution to a complex one.

## Overview

This Streamlit application demonstrates how CFM transforms samples from a standard normal distribution to a bimodal target distribution by learning a velocity field. It provides multiple visualization modes and a step-by-step lecture walkthrough to understand the algorithm.

## Features

- **Interactive Controls**: Configure target distribution parameters (modes, spread, weights)
- **Multiple Visualization Modes**:
  - Lecture Walkthrough - 7-step educational progression
  - Linear Connections - Visualize pairings between source and target
  - Flow Trajectories - Interactive ODE trajectory exploration
  - Static Matplotlib - Traditional visualization with time slider
- **Real-time Training**: Watch the MLP learn the velocity field with live loss curves
- **Dark Theme**: Modern dark-themed interface

## Project Structure

```
flow_matching_visualization/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── flow_matching.py    # CFM training logic
│   ├── mlp.py              # VelocityMLP neural network
│   ├── solver.py           # ODE solver (torchdiffeq)
│   ├── distributions.py    # Source/target distribution samplers
│   ├── visualization.py    # Matplotlib visualizations
│   ├── interactive_viz.py  # Plotly interactive visualizations
│   ├── lecture_viz.py      # Lecture walkthrough visualizations
│   ├── lecture_content.py  # Educational content
│   └── math_panel.py       # Mathematical explanations
├── tests/                  # Test suite
├── pyproject.toml          # Project configuration
└── uv.lock                 # Dependency lock file
```

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd flow_matching_visualization

# Create virtual environment and install dependencies
uv sync

# Or with pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

### Run the Application

```bash
streamlit run src/app.py
```

The app opens at `http://localhost:8501` with:
1. **Sidebar** - Configure distribution and training parameters
2. **Main Panel** - Visualizations and educational content

### Run Tests

```bash
pytest tests/
```

## How It Works

### Conditional Flow Matching Algorithm

1. **Sample Pairs**: Randomly pair source samples (x₀ ~ N(0,1)) with target samples (x₁ ~ bimodal)

2. **Conditional Path**: Linear interpolation between pairs:
   ```
   x_t = (1-t)·x₀ + t·x₁
   ```

3. **Target Velocity**: Constant velocity along each path:
   ```
   u_t = x₁ - x₀
   ```

4. **Learning**: MLP learns the marginal velocity field v(x,t) by minimizing:
   ```
   L = E[||v(x_t, t) - u_t||²]
   ```

5. **Generation**: Integrate learned velocity field via ODE to transform new samples

### Neural Network

The `VelocityMLP` is a 3-layer network with 64 hidden units and SiLU activation that predicts velocity given position and time.

## Dependencies

- PyTorch >= 2.0.0
- torchdiffeq >= 0.2.0
- Streamlit >= 1.30.0
- Matplotlib >= 3.8.0
- Plotly >= 5.18.0
- NumPy >= 1.24.0
- SciPy >= 1.11.0

## License

MIT
