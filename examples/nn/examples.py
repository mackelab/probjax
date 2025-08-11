import jax
import jax.numpy as jnp


class SpiralGenerator:
    def __init__(self, noise_std=0.1, n_turns=4, radius_scale=5.):
        self.noise_std = noise_std
        self.n_turns = n_turns
        self.radius_scale = radius_scale

    def generate(self, num_points, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        max_angle = 2 * jnp.pi * self.n_turns
        
        # Generate evenly spaced angles for the spiral
        t = jnp.linspace(0, max_angle, num_points)
        
        # Calculate radius that grows linearly with angle to create a spiral
        r = self.radius_scale * (t / max_angle + 0.1)
        
        # Generate spiral coordinates
        x = r * jnp.cos(t)
        y = r * jnp.sin(t)
        
        # Add noise to make it more realistic
        subkey1, subkey2 = jax.random.split(key, 2)
        x += jax.random.normal(subkey1, (num_points,)) * self.noise_std
        y += jax.random.normal(subkey2, (num_points,)) * self.noise_std
        
        return jnp.stack([x, y], axis=1)

class CheckerboardGenerator:
    def __init__(self, grid_size=3, scale=5.0, device=None):
        self.grid_size = grid_size
        self.scale = scale
        # device argument is ignored in JAX version

    def generate(self, num_points, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        grid_length = 2 * self.scale / self.grid_size
        oversample_factor = 10  # Large enough to almost always get enough points
        n_samples = num_points * oversample_factor
        key, subkey = jax.random.split(key)
        new_samples = (jax.random.uniform(subkey, (n_samples, 2)) - 0.5) * 2 * self.scale
        x_mask = jnp.floor((new_samples[:, 0] + self.scale) / grid_length) % 2 == 0
        y_mask = jnp.floor((new_samples[:, 1] + self.scale) / grid_length) % 2 == 0
        accept_mask = jnp.logical_xor(~x_mask, y_mask)
        accepted = new_samples[accept_mask]
        # If not enough, raise an error
        if accepted.shape[0] < num_points:
            raise RuntimeError(f"Not enough samples accepted: {accepted.shape[0]} < {num_points}. Increase oversample_factor.")
        return accepted[:num_points]