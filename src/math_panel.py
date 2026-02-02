"""LaTeX explanations for the math panel."""

EXPLANATIONS = {
    "title": r"## Conditional Flow Matching (CFM)",
    "intro": """
Conditional Flow Matching learns to transport samples from a simple source distribution
to a complex target distribution by learning a velocity field.
""",
    "source_target": r"""
### Source & Target Distributions

- **Source**: $p_0 = \mathcal{N}(0, 1)$ (standard normal)
- **Target**: $p_1 = w \cdot \mathcal{N}(\mu_1, \sigma^2) + (1-w) \cdot \mathcal{N}(\mu_2, \sigma^2)$ (bimodal mixture)
""",
    "conditional_path": r"""
### Conditional Path

Given a pair $(x_0, x_1)$ where $x_0 \sim p_0$ and $x_1 \sim p_1$, the conditional path is:

$$x_t = (1-t) \cdot x_0 + t \cdot x_1$$

This linearly interpolates from source to target over $t \in [0, 1]$.
""",
    "conditional_velocity": r"""
### Conditional Velocity

The conditional velocity field is simply:

$$u_t(x \mid x_0, x_1) = x_1 - x_0$$

Note: This is **constant** regardless of $t$ or current position!
""",
    "marginal_velocity": r"""
### Marginal Velocity Field

The marginal velocity field averages over all conditional pairs:

$$v_t(x) = \mathbb{E}[u_t \mid x_t = x]$$

This is what the neural network learns to approximate.
""",
    "training_loss": r"""
### Training Objective

The MLP is trained to minimize:

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(0,1), (x_0, x_1)} \left[ \| v_\theta(x_t, t) - u_t \|^2 \right]$$

where $u_t = x_1 - x_0$ is the target velocity.
""",
    "ode": r"""
### Transport ODE

At inference, we solve the ODE:

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) \sim p_0$$

Using `torchdiffeq` for numerical integration.
""",
    "key_insight": r"""
### Key Insight

The beauty of CFM is that we never need to compute intractable marginals!
By conditioning on pairs $(x_0, x_1)$, the target velocity $u_t$ becomes trivially simple.
""",
}


def get_full_explanation() -> str:
    """Return the full mathematical explanation as markdown."""
    sections = [
        "title",
        "intro",
        "source_target",
        "conditional_path",
        "conditional_velocity",
        "marginal_velocity",
        "training_loss",
        "ode",
        "key_insight",
    ]
    return "\n".join(EXPLANATIONS[s] for s in sections)
