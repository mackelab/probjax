"""
Example script demonstrating the use of constraints in the probjax.stats API.

This script shows how constraints are used to define valid parameter ranges,
how parameters are automatically transformed to satisfy constraints,
and how to check if parameters satisfy constraints.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probjax.stats import norm, beta, gamma, constraints
from probjax.stats.constraints import check_constraints, transform_params

# Set a random seed for reproducibility
key = jax.random.PRNGKey(0)

print("=== Constraints Example ===")

# =====================================================
# Part 1: Inspecting distribution constraints
# =====================================================
print("\n=== Part 1: Inspecting distribution constraints ===")

# Print parameter constraints for different distributions
print("Normal distribution parameter constraints:")
print(f"  {norm.param_constraints}")
print(f"  Support: {norm.support}")

print("\nBeta distribution parameter constraints:")
print(f"  {beta.param_constraints}")
print(f"  Support: {beta.support}")

print("\nGamma distribution parameter constraints:")
print(f"  {gamma.param_constraints}")
print(f"  Support: {gamma.support}")

# =====================================================
# Part 2: Automatic parameter transformation
# =====================================================
print("\n=== Part 2: Automatic parameter transformation ===")

# Try to create distributions with invalid parameters
print("Creating a normal distribution with negative scale:")
normal_negative_scale = norm(loc=0.0, scale=-2.0)
samples = normal_negative_scale.rvs(size=1000, random_state=key)
print(f"  Mean: {jnp.mean(samples):.4f}, Std: {jnp.std(samples):.4f}")
print("  Scale was automatically transformed to positive")

print("\nCreating a beta distribution with negative shape parameters:")
beta_neg_params = beta(a=-2.0, b=-3.0)
samples = beta_neg_params.rvs(size=1000, random_state=key)
print(f"  Mean: {jnp.mean(samples):.4f}, Var: {jnp.var(samples):.4f}")
print("  Shape parameters were automatically transformed to positive")

# =====================================================
# Part 3: Manual constraint checking and transformation
# =====================================================
print("\n=== Part 3: Manual constraint checking and transformation ===")

# Check if parameters satisfy constraints
params = {'loc': 0.0, 'scale': -2.0}
print(f"Parameters: {params}")
print(
    f"Satisfy normal constraints? {check_constraints(params, norm.param_constraints)}"
)

# Transform parameters to satisfy constraints
transformed_params = transform_params(params, norm.param_constraints)
print(f"Transformed parameters: {transformed_params}")
print(
    f"Satisfy normal constraints? {check_constraints(transformed_params, norm.param_constraints)}"
)

# =====================================================
# Part 4: Visualizing constraint-related behavior
# =====================================================
print("\n=== Part 4: Visualizing constraint-related behavior ===")

# Generate examples with different parameters for beta distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Original parameters
params_list = [
    {'a': 0.5, 'b': 0.5},  # Valid
    {'a': 2.0, 'b': 5.0},  # Valid
    {'a': -1.0, 'b': 2.0},  # Invalid a
    {'a': 2.0, 'b': -3.0},  # Invalid b
    {'a': -2.0, 'b': -3.0},  # Both invalid
    {'a': 0.0, 'b': 0.0},  # Both invalid (zero)
]

x = jnp.linspace(0, 1, 1000)

for i, params in enumerate(params_list):
    # Transform parameters according to constraints
    transformed = transform_params(params, beta.param_constraints)

    # Create beta distribution with transformed parameters
    dist = beta(a=transformed['a'], b=transformed['b'])

    # Plot the PDF
    pdf = dist.pdf(x)
    axes[i].plot(x, pdf)
    axes[i].set_title(
        f"Beta(a={params['a']}, b={params['b']}) → Beta(a={transformed['a']:.2f}, b={transformed['b']:.2f})"
    )
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('Probability Density')
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('beta_constraints_example.png')
plt.close()

print(
    "  Beta distribution constraints example plot saved as 'beta_constraints_example.png'"
)
print("\nExample completed successfully!")
