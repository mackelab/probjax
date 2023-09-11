import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray

from functools import partial
from typing import Callable, Union, Optional
from jaxtyping import Array

from probjax.distributions import Distribution
from probjax.utils.linalg import is_matrix, is_diagonal_matrix, transition_matrix
from probjax.utils.sdeint import sdeint
from probjax.utils.odeint import odeint


class BaseSDE(Distribution):
    def __init__(self, drift: Callable, diffusion: Callable, p0: Distribution) -> None:
        self.drift = drift
        self.diffusion = diffusion
        self.p0 = p0

        self.t_o = None
        self.x_o = None

        super().__init__(batch_shape=p0.batch_shape, event_shape=p0.event_shape)

    def condition(
        self,
        t_o: Array,
        x_o: Array,
        measurement_projection: Optional[Array] = None,
        measurement_noise: float = 0.5,
    ) -> Distribution:
        raise NotImplementedError

    def mean(self, ts: Array, **kwargs) -> Array:
        assert jnp.all(ts >= 0), "t must be positive"

        # In general we just have to solve the ODE for the mean i.e. ignore the diffusion term.
        # TODO This may not be true for non-linear SDEs ...
        _odeint = partial(odeint, **kwargs)
        if self.batch_shape != ():
            _odeint = jax.vmap(_odeint, in_axes=(None, 0, None))
        mu0 = self.p0.mean
        mus = _odeint(self.drift, mu0, ts)

        return mus

    @property
    def var(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError

    @property
    def cross_cov(self, t1: Array, t2: Array) -> Array:
        assert jnp.all(t1 >= 0), "t1 must be positive"
        assert jnp.all(t2 >= 0), "t2 must be positive"
        raise NotImplementedError

    def sample(self, key: PRNGKeyArray, ts: Array, sample_shape=(), **kwargs) -> Array:
        """Samples from the SDE

        Args:
            key (PRNGKeyArray): Random key
            ts (Array): Number of time points to evaluate the SDE
            sample_shape (tuple, optional): Number of samples. Defaults to ().
            **kwargs: Additional arguments to pass to the solver i.e. see sdeint in probjax/utils/sdeint.py for more details

        Returns:
            Array: Samples from the SDE of shape (sample_shape, batch_shape, event_shape)
        """
        assert jnp.all(ts >= 0), "t must be positive"
        _sdeint = partial(sdeint, **kwargs)
        shape = sample_shape + ts.shape + self.batch_shape + self.event_shape
        key1, key2 = jax.random.split(key)
        x0 = self.p0.sample(key1, sample_shape)
        x0_flat = x0.reshape(-1, *self.event_shape)
        keys_flat = jax.random.split(key2, x0_flat.shape[0])
        __sdeint = jax.vmap(_sdeint, in_axes=(0, None, None, 0, None))
        ys = __sdeint(keys_flat, self.drift, self.diffusion, x0_flat, ts)
        ys = ys.reshape(*shape)
        return ys

    def log_prob(self, x: Array, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError


class LinearSDE(BaseSDE):
    def __init__(
        self,
        drift_matrix: Union[Callable, Array],
        diffusion_matrix: Union[Callable, Array],
        p0: Distribution,
    ) -> None:
        # If drift and diffusion are independent of time, then the transition matrix is also independent of time (only depends on the time difference)
        self._time_dependent = callable(drift_matrix) or callable(diffusion_matrix)
        self._independent = (
            not callable(drift_matrix)
            and is_diagonal_matrix(drift_matrix)
            and not callable(diffusion_matrix)
            and is_diagonal_matrix(diffusion_matrix)
        )

        # Time dependent or independent drift matrix
        if not callable(drift_matrix):

            def drift(t, x):
                return jnp.matmul(drift_matrix, x)

        else:

            def drift(t, x):
                return jnp.matmul(drift_matrix(t), x)

        # Time dependent or independent diffusion matrix
        if not callable(diffusion_matrix):

            def diffusion(t, x):
                return jnp.matmul(diffusion_matrix, x)

        else:

            def diffusion(t, x):
                return jnp.matmul(diffusion_matrix(t), x)

        super().__init__(drift, diffusion, p0)

        # Store the matrices
        self.diffusion_matrix = diffusion_matrix
        self.drift_matrix = drift_matrix

    def mean(self, t: Array) -> Array:
        if callable(self.drift_matrix):
            A = self.drift_matrix(t)
        else:
            A = self.drift_matrix

        mu0 = self.p0.mean
        if t == 0:
            return mu0
        else:
            if not self._time_dependent:
                return jnp.matmul(transition_matrix(A, 0, t), mu0)
            else:
                raise NotImplementedError()
