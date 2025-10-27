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
        # Calculate the area ratio to estimate required samples
        outer_area = jnp.pi * self.R**2
        inner_area = jnp.pi * self.r**2
        crescent_area = outer_area - inner_area

        # Estimate required samples with 20% buffer
        n_samples = int(num_points * (outer_area / crescent_area) * 1.2)
        n_samples = max(n_samples, num_points)  # Ensure we generate at least num_points

        def gen_points(key):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            theta = 2 * jnp.pi * jax.random.uniform(subkey1, (n_samples,))
            radius = self.R * jnp.sqrt(jax.random.uniform(subkey2, (n_samples,)))
            x = radius * jnp.cos(theta)
            y = radius * jnp.sin(theta)
            mask = (x - self.d)**2 + y**2 > self.r**2
            points = jnp.stack((x[mask], y[mask]), axis=1)
            return points

        points = gen_points(key)
        key, _ = jax.random.split(key)

        def cond_fun(state):
            points, key = state
            return points.shape[0] < num_points

        def body_fun(state):
            points, key = state
            key, subkey = jax.random.split(key)
            new_points = gen_points(subkey)
            points = jnp.concatenate((points, new_points), axis=0)
            return (points, key)

        points, key = jax.lax.while_loop(cond_fun, body_fun, (points, key))
        return points[:num_points].astype(jnp.float32)

class SpiralGenerator:
    def __init__(self, noise_std=0.1, n_turns=4, radius_scale=0.5):
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