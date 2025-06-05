"""
Example script for using the higher-order distributions in probjax.stats.

This script demonstrates how to use the Independent, Mixture, and Transformed distributions
with the SciPy-like API.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import the necessary distributions
from probjax.stats import (
    norm,
    binom,
    independent_continuous,
    independent_discrete,
    mixture_continuous,
    mixture_discrete,
    transformed_continuous,
    transformed_discrete,
)

# Set a random seed for reproducibility
key = jax.random.PRNGKey(0)

print("=== Higher Order Distributions Example ===")

# =====================================================
# Part 1: Independent Distribution
# =====================================================
print("\n=== Part 1: Independent Distribution ===")

# Create an independent continuous distribution (makes a normal distribution multivariate)
print("Creating an Independent distribution from a normal distribution...")
mvn = independent_continuous(
    base_dist=norm,  # Use the normal distribution as the base
    reinterpreted_batch_ndims=1,  # Treat the last batch dimension as part of the event
)

# Sample from the distribution
# This will create a batch of 5 independent normal distributions
samples = mvn.rvs(
    base_dist=norm(loc=jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]), scale=1.0),
    size=1000,
    random_state=key,
)

print(f"Independent samples shape: {samples.shape}")
print(f"Mean of each dimension: {jnp.mean(samples, axis=0)}")
print(f"Standard deviation of each dimension: {jnp.std(samples, axis=0)}")

# =====================================================
# Part 2: Mixture Distribution
# =====================================================
print("\n=== Part 2: Mixture Distribution ===")

# Create a mixture of two normal distributions
print("Creating a Mixture of two normal distributions...")
mixture = mixture_continuous(
    mixing_probs=jnp.array([0.3, 0.7]),  # Mixing weights
    component_distributions=[norm(loc=-3.0, scale=1.0), norm(loc=3.0, scale=1.0)],
)

# Generate samples
x = jnp.linspace(-10, 10, 1000)
pdf_values = mixture.pdf(x)

# Plot the mixture PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_values, label='Mixture PDF')
plt.plot(
    x, 0.3 * norm.pdf(x, loc=-3.0, scale=1.0), '--', label='Component 1 (weight=0.3)'
)
plt.plot(
    x, 0.7 * norm.pdf(x, loc=3.0, scale=1.0), '--', label='Component 2 (weight=0.7)'
)
plt.title('Gaussian Mixture Model')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig('mixture_distribution.png')
plt.close()

print("Gaussian mixture plot saved as 'mixture_distribution.png'")

# Sample from the mixture
mix_samples = mixture.rvs(size=1000, random_state=key)
print(f"Mixture samples shape: {mix_samples.shape}")

# =====================================================
# Part 3: Transformed Distribution
# =====================================================
print("\n=== Part 3: Transformed Distribution ===")


# Define a transformation (exponential)
def exp_transform(x):
    return jnp.exp(x)


# Create a transformed normal distribution (becomes a log-normal)
print("Creating a transformed normal distribution (log-normal)...")
lognormal = transformed_continuous(
    base_dist=norm(loc=0.0, scale=1.0), transform=exp_transform
)

# Generate samples
ln_samples = lognormal.rvs(size=1000, random_state=key)
print(f"Log-normal samples shape: {ln_samples.shape}")
print(f"Log-normal mean: {jnp.mean(ln_samples)}")
print(f"Log-normal standard deviation: {jnp.std(ln_samples)}")

# Plot the transformed distribution
x_norm = jnp.linspace(-4, 4, 1000)
x_lognorm = jnp.linspace(0.01, 10, 1000)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(jnp.log(ln_samples), bins=30, density=True, alpha=0.6, label='log(Samples)')
plt.plot(x_norm, norm.pdf(x_norm, loc=0.0, scale=1.0), 'r-', label='Normal PDF')
plt.title('Base Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(ln_samples, bins=30, density=True, alpha=0.6, label='Samples')
plt.plot(x_lognorm, lognormal.pdf(x_lognorm), 'r-', label='Log-Normal PDF')
plt.title('Transformed (Log-Normal) Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

plt.tight_layout()
plt.savefig('transformed_distribution.png')
plt.close()

print("Transformed distribution plot saved as 'transformed_distribution.png'")

# =====================================================
# Part 4: Combining Higher-Order Distributions
# =====================================================
print("\n=== Part 4: Combining Higher-Order Distributions ===")

# Create a mixture of independent distributions
print("Creating a mixture of independent normal distributions...")

# First independent distribution: A batch of normals with different means
ind1 = independent_continuous(
    base_dist=norm(loc=jnp.array([-2.0, -1.0, 0.0]), scale=0.5),
    reinterpreted_batch_ndims=1,
)

# Second independent distribution: A batch of normals with different means
ind2 = independent_continuous(
    base_dist=norm(loc=jnp.array([2.0, 3.0, 4.0]), scale=0.5),
    reinterpreted_batch_ndims=1,
)

# Create a mixture of these two independent distributions
mixture_of_ind = mixture_continuous(
    mixing_probs=jnp.array([0.4, 0.6]), component_distributions=[ind1, ind2]
)

# Sample from the mixture of independent distributions
mixture_ind_samples = mixture_of_ind.rvs(size=1000, random_state=key)
print(f"Mixture of independent samples shape: {mixture_ind_samples.shape}")

# Create a scatter plot for the first two dimensions
plt.figure(figsize=(8, 8))
plt.scatter(mixture_ind_samples[:, 0], mixture_ind_samples[:, 1], alpha=0.5)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Mixture of Independent Distributions (Dimensions 0 and 1)')
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.grid(True, alpha=0.3)
plt.savefig('mixture_of_independent.png')
plt.close()

print("Mixture of independent distributions plot saved as 'mixture_of_independent.png'")
print("\nExample completed successfully!")
