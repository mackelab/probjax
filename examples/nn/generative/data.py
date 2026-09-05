"""Toy 2D density-estimation datasets used by the generative-model examples.

Each generator exposes ``generate(num_points, key=None) -> (num_points, 2)``.
"""

import jax
import jax.numpy as jnp


class CrescentGenerator:
    def __init__(self, R=1.0, r=0.6, d=0.5):
        self.R = R  # Outer radius
        self.r = r  # Inner circle radius
        self.d = d  # Offset of inner circle

    def generate(self, num_points, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        target = jnp.int32(num_points)
        # Large batch size keeps the expected number of accepted samples well above target.
        batch_size = max(num_points * 4, num_points + 128)

        def sample_batch(rng):
            rng, subkey1, subkey2 = jax.random.split(rng, 3)
            theta = 2 * jnp.pi * jax.random.uniform(subkey1, (batch_size,))
            radius = self.R * jnp.sqrt(jax.random.uniform(subkey2, (batch_size,)))
            x = radius * jnp.cos(theta)
            y = radius * jnp.sin(theta)
            samples = jnp.stack((x, y), axis=1).astype(jnp.float32)
            accept = (x - self.d) ** 2 + y**2 > self.r**2
            return samples, accept, rng

        def cond_fun(state):
            _, count, _ = state
            return count < target

        def body_fun(state):
            points_out, count, rng = state
            samples, accept, rng = sample_batch(rng)
            positions = count + jnp.cumsum(accept.astype(jnp.int32)) - 1
            write_mask = accept & (positions >= 0) & (positions < target)
            write_idx = jnp.where(write_mask, positions, 0)
            write_vals = jnp.where(write_mask[:, None], samples, 0.0)
            points_out = points_out.at[write_idx].add(write_vals)
            new_count = count + write_mask.sum(dtype=jnp.int32)
            return points_out, new_count, rng

        init_points = jnp.zeros((num_points, 2), dtype=jnp.float32)
        init_state = (init_points, jnp.int32(0), key)
        points_out, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return points_out


class SpiralGenerator:
    def __init__(self, noise_std=0.01, n_turns=4, radius_scale=0.5):
        self.noise_std = noise_std
        self.n_turns = n_turns
        self.radius_scale = radius_scale

    def generate(self, num_points, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        max_angle = 2 * jnp.pi * self.n_turns
        t = jnp.linspace(0, max_angle, num_points)
        key, subkey = jax.random.split(key)
        t = t * jnp.power(jax.random.uniform(subkey, (num_points,)), 0.5)

        r = self.radius_scale * (t / max_angle + 0.1)
        x = r * jnp.cos(t)
        y = r * jnp.sin(t)

        subkey1, subkey2 = jax.random.split(key, 2)
        x += jax.random.normal(subkey1, (num_points,)) * self.noise_std
        y += jax.random.normal(subkey2, (num_points,)) * self.noise_std
        return jnp.stack([x, y], axis=1)


class CheckerboardGenerator:
    def __init__(self, grid_size=3, scale=5.0):
        self.grid_size = grid_size
        self.scale = scale
        grid_indices = []
        for i in range(grid_size):
            for j in range(grid_size):
                if (i % 2) == (j % 2):
                    grid_indices.append((i, j))
        self.allowed_cells = jnp.array(grid_indices, dtype=jnp.int32)
        self.grid_length = 2 * self.scale / self.grid_size

    def generate(self, num_points, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_cells, key_offsets = jax.random.split(key)
        chosen_ids = jax.random.randint(
            key_cells, (num_points,), minval=0, maxval=self.allowed_cells.shape[0]
        )
        cells = self.allowed_cells[chosen_ids]
        offsets = jax.random.uniform(key_offsets, (num_points, 2))
        samples = -self.scale + (cells + offsets) * self.grid_length
        return samples.astype(jnp.float32)
