"""
Example script demonstrating the SciPy-like API for distributions in ProbJAX.

This script shows how to use the SciPy-like API for working with probability
distributions, similar to the way scipy.stats is used.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probjax.stats import norm, beta, gamma
from probjax.stats import rv_continuous_frozen, rv_discrete_frozen

# Set a random seed for reproducibility
key = jax.random.PRNGKey(0)

print("=== SciPy-like Distribution API Example ===")

# =====================================================
# Part 1: Using the API in stateless mode
# =====================================================
print("\n=== Part 1: Stateless API Usage ===")

# Generate random samples from a normal distribution
x = jnp.linspace(-5, 5, 1000)
samples = norm.rvs(loc=0.0, scale=1.0, size=1000, random_state=key)

# Calculate PDF, CDF, and PPF
pdf_values = norm.pdf(x, loc=0.0, scale=1.0)
cdf_values = norm.cdf(x, loc=0.0, scale=1.0)
ppf_values = norm.ppf(jnp.array([0.025, 0.5, 0.975]), loc=0.0, scale=1.0)

print("Normal Distribution (stateless mode):")
print(f"  Mean of samples: {jnp.mean(samples):.4f}")
print(f"  PDF at x=0: {norm.pdf(0.0, loc=0.0, scale=1.0):.4f}")
print(f"  CDF at x=0: {norm.cdf(0.0, loc=0.0, scale=1.0):.4f}")
print(f"  95% confidence interval: {ppf_values[0]:.4f}, {ppf_values[2]:.4f}")

# =====================================================
# Part 2: Using frozen distributions (stateful mode)
# =====================================================
print("\n=== Part 2: Frozen Distributions ===")

# Create frozen distributions
frozen_norm = norm(loc=0.0, scale=1.0)
frozen_beta = beta(a=2.0, b=3.0)
frozen_gamma = gamma(a=2.0, scale=2.0)

# Verify these are frozen distribution instances
print(
    f"frozen_norm is rv_continuous_frozen: {isinstance(frozen_norm, rv_continuous_frozen)}"
)

# Calculate statistics
print("\nNormal Distribution (frozen):")
print(f"  Mean: {frozen_norm.mean():.4f}")
print(f"  Variance: {frozen_norm.var():.4f}")
print(f"  Standard deviation: {frozen_norm.std():.4f}")
print(f"  Entropy: {frozen_norm.entropy():.4f}")

print("\nBeta Distribution (frozen):")
print(f"  Mean: {frozen_beta.mean():.4f}")
print(f"  Variance: {frozen_beta.var():.4f}")
print(
    f"  Mode: {2.0 - 1.0} / {2.0 + 3.0 - 2.0} = {(2.0 - 1.0) / (2.0 + 3.0 - 2.0):.4f}"
)
print(f"  Entropy: {frozen_beta.entropy():.4f}")

print("\nGamma Distribution (frozen):")
print(f"  Mean: {frozen_gamma.mean():.4f}")
print(f"  Variance: {frozen_gamma.var():.4f}")
print(f"  Standard deviation: {frozen_gamma.std():.4f}")
print(f"  Entropy: {frozen_gamma.entropy():.4f}")

# =====================================================
# Part 3: Plotting the distributions
# =====================================================
print("\n=== Part 3: Plotting Distributions ===")

# Plot PDFs and samples
plt.figure(figsize=(15, 5))

# Normal distribution
x_norm = jnp.linspace(-4, 4, 1000)
plt.subplot(1, 3, 1)
plt.plot(x_norm, frozen_norm.pdf(x_norm), 'r-', label='PDF')
plt.hist(
    frozen_norm.rvs(size=1000, random_state=key),
    bins=30,
    density=True,
    alpha=0.5,
    label='Samples',
)
plt.title('Normal(0, 1)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()

# Beta distribution
x_beta = jnp.linspace(0, 1, 1000)
plt.subplot(1, 3, 2)
plt.plot(x_beta, frozen_beta.pdf(x_beta), 'g-', label='PDF')
plt.hist(
    frozen_beta.rvs(size=1000, random_state=jax.random.split(key)[0]),
    bins=30,
    density=True,
    alpha=0.5,
    label='Samples',
)
plt.title('Beta(2, 3)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()

# Gamma distribution
x_gamma = jnp.linspace(0, 15, 1000)
plt.subplot(1, 3, 3)
plt.plot(x_gamma, frozen_gamma.pdf(x_gamma), 'b-', label='PDF')
plt.hist(
    frozen_gamma.rvs(size=1000, random_state=jax.random.split(key)[1]),
    bins=30,
    density=True,
    alpha=0.5,
    label='Samples',
)
plt.title('Gamma(2, scale=2)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.savefig('scipy_like_distributions.png')
plt.close()

print("Distribution plots saved as 'scipy_like_distributions.png'")
print("\nExample completed successfully!")
