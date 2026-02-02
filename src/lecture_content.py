"""Step-by-step lecture content for CFM walkthrough."""

STEP_CONTENT = {
    1: {
        "title": "Step 1: Source & Target Distributions",
        "subtitle": "The Problem Setup",
        "explanation": """
**Goal**: Transport samples from a simple distribution to a complex one.

We have two distributions:
- **Source** $p_0$: Standard normal $\\mathcal{N}(0,1)$ - easy to sample from
- **Target** $p_1$: Bimodal mixture - the distribution we actually want

**The Question**: How do we learn a transformation that converts samples from $p_0$ into samples from $p_1$?

Flow Matching gives us a principled way to learn this transformation by defining a *velocity field* that pushes particles from source to target.
        """,
    },
    2: {
        "title": "Step 2: Conditional Pairs",
        "subtitle": "The Key Insight of CFM",
        "explanation": """
**Key Insight**: Pair each source sample with a target sample!

For each training example, we have a pair $(x_0, x_1)$ where:
- $x_0 \\sim \\mathcal{N}(0,1)$ is sampled from the source
- $x_1 \\sim p_{\\text{target}}$ is sampled from the target

These pairs define a **conditional** flow - we know exactly where each particle should go.

The cyan line shows the actual pairing: this source point $x_0$ is paired with this target point $x_1$.

*Note: The pairing is random - any source can pair with any target!*
        """,
    },
    3: {
        "title": "Step 3: The Conditional Path",
        "subtitle": "Linear Interpolation",
        "explanation": """
**The Conditional Path**: How does a particle move from $x_0$ to $x_1$?

The simplest choice: **linear interpolation**!

$$x_t = (1-t) \\cdot x_0 + t \\cdot x_1$$

At time $t=0$: particle is at $x_0$ (source)
At time $t=1$: particle is at $x_1$ (target)
At time $t=0.5$: particle is halfway between

Watch the red dot move along this straight-line path as you change $t$.

This simple linear path is the foundation of Conditional Flow Matching!
        """,
    },
    4: {
        "title": "Step 4: The Conditional Velocity",
        "subtitle": "The Target We Learn",
        "explanation": """
**The Conditional Velocity**: What velocity moves the particle along its path?

Taking the derivative of $x_t = (1-t) \\cdot x_0 + t \\cdot x_1$:

$$u_t = \\frac{{dx_t}}{{dt}} = x_1 - x_0$$

**This is constant!** The velocity doesn't depend on $t$ at all.

For this pair: $u_t = x_1 - x_0 = $ {velocity:.3f}

The arrows show this constant velocity - the particle moves at the same speed throughout its journey.

This simple, constant target velocity is what makes CFM training so stable!
        """,
    },
    5: {
        "title": "Step 5: Training Data",
        "subtitle": "What the MLP Sees",
        "explanation": """
**Training Data**: What does the neural network learn from?

For each training sample, we:
1. Pick a random pair $(x_0, x_1)$
2. Sample a random time $t \\sim \\text{Uniform}(0, 1)$
3. Compute the position $x_t = (1-t) \\cdot x_0 + t \\cdot x_1$
4. Compute the target velocity $u_t = x_1 - x_0$

The MLP input is $(x_t, t)$ and the target output is $u_t$.

The scatter plot shows many such training points:
- **X-axis**: position $x_t$
- **Y-axis**: time $t$
- **Color**: target velocity $u_t$ (red = positive, blue = negative)

Notice: at each $(x, t)$ location, we might see different target velocities because different pairs pass through the same point!
        """,
    },
    6: {
        "title": "Step 6: MLP Learning",
        "subtitle": "Learning the Marginal Velocity Field",
        "explanation": """
**The MLP's Job**: Learn the *marginal* velocity field $v(x, t)$.

The MLP is trained to minimize:
$$\\mathcal{L}(\\theta) = \\mathbb{E}_{t, x_0, x_1}\\left[ \\| v_\\theta(x_t, t) - u_t \\|^2 \\right]$$

Because multiple conditional paths can pass through the same $(x, t)$ point with different velocities, the MLP learns the **average** (marginal) velocity:

$$v_t(x) = \\mathbb{E}[u_t \\mid x_t = x]$$

Watch how the velocity field evolves during training:
- **Epoch 0**: Random noise (untrained network)
- **Early epochs**: Rough patterns emerge
- **Later epochs**: Smooth velocity field that correctly transports particles

Use the slider to scrub through training and see the learning happen!
        """,
    },
    7: {
        "title": "Step 7: The Final Flow",
        "subtitle": "Transporting All Particles",
        "explanation": """
**The Result**: A learned velocity field that transports any source sample to the target distribution!

Now we solve the ODE:
$$\\frac{{dx}}{{dt}} = v_\\theta(x, t), \\quad x(0) \\sim \\mathcal{{N}}(0,1)$$

Starting from source samples, we integrate forward using the learned velocity field.

The particles:
- Start at the source distribution (left edge)
- Flow according to $v_\\theta(x, t)$
- End up matching the target distribution (right edge)

**The magic**: We never explicitly defined how to get from source to target - the MLP learned it from simple conditional pairs!

This is the power of Flow Matching: simple training, powerful results.
        """,
    },
}


def get_step_content(step: int) -> dict:
    """Get content for a specific step."""
    return STEP_CONTENT.get(step, STEP_CONTENT[1])


def get_total_steps() -> int:
    """Get total number of steps."""
    return len(STEP_CONTENT)
