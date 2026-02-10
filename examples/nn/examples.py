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
    def __init__(self, grid_size=3, scale=5.0, device=None):
        self.grid_size = grid_size
        self.scale = scale
        # device argument is ignored in JAX version
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


class TwoMoons:
    """JAX version of the SBIBM Two Moons simulator for conditional estimation demos.

    The simulator maps latent parameters -> observations. A helper method
    `sample_prior` is provided to draw parameters uniformly in [-prior_bound, prior_bound]^2.
    """

    def __init__(
        self,
        prior_bound=1.0,
        a_low=-jnp.pi / 2.0,
        a_high=jnp.pi / 2.0,
        base_offset=0.25,
        r_loc=0.1,
        r_scale=0.01,
    ):
        self.prior_bound = float(prior_bound)
        self.a_low = float(a_low)
        self.a_high = float(a_high)
        self.base_offset = float(base_offset)
        self.r_loc = float(r_loc)
        self.r_scale = float(r_scale)
        self._rot_c = jnp.cos(-jnp.pi / 4.0)
        self._rot_s = jnp.sin(-jnp.pi / 4.0)

    def sample_prior(self, num_samples, key=None):
        """Draw parameters ~ Uniform([-prior_bound, prior_bound]^2)."""
        if key is None:
            key = jax.random.PRNGKey(0)
        return jax.random.uniform(
            key,
            (num_samples, 2),
            minval=-self.prior_bound,
            maxval=self.prior_bound,
            dtype=jnp.float32,
        )

    def simulate(self, parameters, key=None):
        """Simulate observations conditioned on parameters.

        Args:
            parameters: array of shape (N, 2)
            key: PRNGKey
        Returns:
            observations array of shape (N, 2)
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        parameters = jnp.asarray(parameters, dtype=jnp.float32)
        num = parameters.shape[0]
        key_a, key_r = jax.random.split(key)
        a = jax.random.uniform(key_a, (num, 1), minval=self.a_low, maxval=self.a_high)
        r = self.r_loc + self.r_scale * jax.random.normal(key_r, (num, 1))

        base = jnp.concatenate(
            (jnp.cos(a) * r + self.base_offset, jnp.sin(a) * r), axis=1
        )

        z0 = (self._rot_c * parameters[:, 0] - self._rot_s * parameters[:, 1]).reshape(
            -1, 1
        )
        z1 = (self._rot_s * parameters[:, 0] + self._rot_c * parameters[:, 1]).reshape(
            -1, 1
        )
        translated = base + jnp.concatenate((-jnp.abs(z0), z1), axis=1)
        return translated.astype(jnp.float32)
