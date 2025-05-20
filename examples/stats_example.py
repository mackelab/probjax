"""
Example script for using the SciPy-like statistics API in probjax.

This script demonstrates how to use the SciPy-like API for distributions
with the JAX-based distribution implementations.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import the base classes and distribution functions
from probjax.stats import rv_continuous, rv_discrete
from probjax.stats import norm, gamma, beta, bernoulli, binom, poisson

# Set a random seed for reproducibility
key = jax.random.PRNGKey(0)

# =====================================================
# Part 1: Using stateless functions (like scipy.stats)
# =====================================================
print("=== PART 1: Using stateless functions ===")

# Generate samples from various distributions
print("Generating samples from distributions...")

# Normal distribution with mean=0, std=1
normal_samples = norm.rvs(loc=0, scale=1, size=1000, random_state=key)
print(f"Normal: mean={normal_samples.mean():.4f}, std={normal_samples.std():.4f}")

# Gamma distribution with shape=2, scale=2
gamma_samples = gamma.rvs(a=2, scale=2, size=1000, random_state=key)
print(f"Gamma: mean={gamma_samples.mean():.4f}, std={gamma_samples.std():.4f}")

# Beta distribution with a=2, b=5
beta_samples = beta.rvs(a=2, b=5, size=1000, random_state=key)
print(f"Beta: mean={beta_samples.mean():.4f}, std={beta_samples.std():.4f}")

# Compute theoretical statistics
print("\nComparing with theoretical values...")

# Normal distribution
norm_mean = norm.mean(loc=0, scale=1)
norm_var = norm.var(loc=0, scale=1)
print(f"Normal: theory mean={norm_mean:.4f}, theory var={norm_var:.4f}")

# Gamma distribution
gamma_mean = gamma.mean(a=2, scale=2)
gamma_var = gamma.var(a=2, scale=2)
print(f"Gamma: theory mean={gamma_mean:.4f}, theory var={gamma_var:.4f}")

# Beta distribution
beta_mean = beta.mean(a=2, b=5)
beta_var = beta.var(a=2, b=5)
print(f"Beta: theory mean={beta_mean:.4f}, theory var={beta_var:.4f}")

# =====================================================
# Part 2: Using stateful instances (frozen distributions)
# =====================================================
print("\n=== PART 2: Using stateful/frozen distributions ===")

# Create frozen distributions
frozen_norm = norm(loc=1.5, scale=2.0)
frozen_gamma = gamma(a=3, scale=1.5)
frozen_beta = beta(a=2, b=3)
frozen_bernoulli = bernoulli(p=0.7)
frozen_binom = binom(n=10, p=0.3)
frozen_poisson = poisson(mu=3)

# Generate samples using frozen distributions
print("Generating samples from frozen distributions...")

# We can call methods directly on the frozen distribution objects
# without specifying parameters again
norm_samples = frozen_norm.rvs(size=1000, random_state=key)
gamma_samples = frozen_gamma.rvs(size=1000, random_state=key)
beta_samples = frozen_beta.rvs(size=1000, random_state=key)
bernoulli_samples = frozen_bernoulli.rvs(size=1000, random_state=key)
binomial_samples = frozen_binom.rvs(size=1000, random_state=key)
poisson_samples = frozen_poisson.rvs(size=1000, random_state=key)

print(
    f"Frozen Normal(1.5, 2.0): mean={norm_samples.mean():.4f}, std={norm_samples.std():.4f}"
)
print(
    f"Frozen Gamma(3, 1.5): mean={gamma_samples.mean():.4f}, std={gamma_samples.std():.4f}"
)
print(
    f"Frozen Beta(2, 3): mean={beta_samples.mean():.4f}, std={beta_samples.std():.4f}"
)
print(
    f"Frozen Bernoulli(0.7): mean={bernoulli_samples.mean():.4f}, std={bernoulli_samples.std():.4f}"
)
print(
    f"Frozen Binomial(10, 0.3): mean={binomial_samples.mean():.4f}, std={binomial_samples.std():.4f}"
)
print(
    f"Frozen Poisson(3): mean={poisson_samples.mean():.4f}, std={poisson_samples.std():.4f}"
)

# Compute theoretical statistics for frozen distributions
print("\nComputing theoretical values for frozen distributions...")

# Get theoretical moments directly from the frozen distributions
norm_mean = frozen_norm.mean()
norm_var = frozen_norm.var()
print(f"Frozen Normal: theory mean={norm_mean:.4f}, theory var={norm_var:.4f}")

gamma_mean = frozen_gamma.mean()
gamma_var = frozen_gamma.var()
print(f"Frozen Gamma: theory mean={gamma_mean:.4f}, theory var={gamma_var:.4f}")

beta_mean = frozen_beta.mean()
beta_var = frozen_beta.var()
print(f"Frozen Beta: theory mean={beta_mean:.4f}, theory var={beta_var:.4f}")

# =====================================================
# Part 3: Create custom distribution by extending base class
# =====================================================
print("\n=== PART 3: Creating a custom distribution by extending base class ===")


class _custom_normal_gen(rv_continuous):
    """A custom normal distribution with different parameterization."""

    @classmethod
    def _create_dist(cls, mean=0.0, variance=1.0, **kwargs):
        """Create a Normal distribution with mean and variance parameters.

        This demonstrates how to create a custom distribution with different
        parameter names than the underlying implementation.
        """
        from probjax.distributions import Normal

        return Normal(loc=mean, scale=jnp.sqrt(variance))


# Create an instance of our custom distribution
custom_norm = _custom_normal_gen(shapes="mean,variance")

# Use it like any other distribution function
custom_samples = custom_norm.rvs(mean=2.0, variance=4.0, size=1000, random_state=key)
print(
    f"Custom Normal(2.0, 4.0): mean={custom_samples.mean():.4f}, std={custom_samples.std():.4f}"
)

# Create a frozen instance of our custom distribution
frozen_custom = custom_norm(mean=2.0, variance=4.0)
frozen_custom_samples = frozen_custom.rvs(size=1000, random_state=key)
print(
    f"Frozen Custom Normal: mean={frozen_custom_samples.mean():.4f}, std={frozen_custom_samples.std():.4f}"
)
print(
    f"Frozen Custom Normal: theory mean={frozen_custom.mean():.4f}, theory var={frozen_custom.var():.4f}"
)

# =====================================================
# Part 4: Compute PDF/PMF values and plot
# =====================================================
print("\n=== PART 4: Computing PDF/PMF values and plotting distributions ===")

plt.figure(figsize=(12, 10))

# Plot a custom distribution
plt.subplot(2, 2, 1)
x = jnp.linspace(-5, 9, 100)
plt.plot(
    x, custom_norm.pdf(x, mean=2.0, variance=4.0), 'b-', label='Custom Normal(2, 4)'
)
plt.plot(x, norm.pdf(x, loc=2.0, scale=2.0), 'r--', label='Standard Normal(2, 2)')
plt.title('Custom Distribution vs Standard')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()

# Plot continuous distributions with different parameters
plt.subplot(2, 2, 2)
plt.plot(x, norm.pdf(x, loc=0, scale=1), label='Normal(0, 1)')
plt.plot(x, norm.pdf(x, loc=0, scale=2), label='Normal(0, 2)')
plt.plot(x, norm.pdf(x, loc=2, scale=1), label='Normal(2, 1)')
plt.title('Normal Distributions')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()

# Plot discrete distributions
plt.subplot(2, 2, 3)
x_binom1 = jnp.arange(11)
x_binom2 = jnp.arange(21)
plt.bar(
    x_binom1 - 0.2,
    binom.pmf(x_binom1, n=10, p=0.3),
    width=0.4,
    alpha=0.6,
    label='Binomial(10, 0.3)',
)
plt.bar(
    x_binom1 + 0.2,
    binom.pmf(x_binom1, n=10, p=0.5),
    width=0.4,
    alpha=0.6,
    label='Binomial(10, 0.5)',
)
plt.title('Binomial Distributions')
plt.xlabel('k')
plt.ylabel('PMF')
plt.legend()

# Plot Poisson distributions
plt.subplot(2, 2, 4)
x_poisson = jnp.arange(20)
plt.bar(
    x_poisson - 0.2,
    poisson.pmf(x_poisson, mu=3),
    width=0.4,
    alpha=0.6,
    label='Poisson(3)',
)
plt.bar(
    x_poisson + 0.2,
    poisson.pmf(x_poisson, mu=7),
    width=0.4,
    alpha=0.6,
    label='Poisson(7)',
)
plt.title('Poisson Distributions')
plt.xlabel('k')
plt.ylabel('PMF')
plt.legend()

plt.tight_layout()
plt.savefig('distributions_comparison.png')
plt.close()

print("Comparison plot saved as 'distributions_comparison.png'")

print("\nExample completed successfully!")
