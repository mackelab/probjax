import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray

from functools import partial
from typing import Callable, Union, Optional
from jaxtyping import Array

from probjax.distributions import Distribution
from probjax.utils.linalg import (
    is_matrix,
    is_diagonal_matrix,
    transition_matrix,
    matrix_fraction_decomposition,
)
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
        self.t_o = t_o
        self.x_o = x_o
        raise NotImplementedError

    def mean(self, ts: Array, **kwargs) -> Array:
        assert jnp.all(ts >= 0), "t must be positive"
        raise NotImplementedError

    def var(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError

    def covariance_matrix(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError

    def cross_covariance(self, t1: Array, t2: Array) -> Array:
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


class LinearTimeInvariantSDE(BaseSDE):
    def __init__(
        self,
        drift_matrix: Array,
        diffusion_matrix: Array,
        p0: Distribution,
    ) -> None:
        assert (
            drift_matrix.ndim == 1 or drift_matrix.shape[1] == p0.event_shape[0]
        ), "Drift matrix must be compatible with initial distribution"
        assert (
            drift_matrix.ndim == 1 or diffusion_matrix.shape[0] == p0.event_shape[0]
        ), "Diffusion matrix must be compatible with initial distribution"

        drift = lambda t, x: jnp.matmul(drift_matrix, x)
        diffusion = lambda t, x: diffusion_matrix

        super().__init__(drift, diffusion, p0)

        # Store the matrices
        self.diffusion_matrix = diffusion_matrix
        self.drift_matrix = drift_matrix

    def mean(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        mu0 = self.p0.mean
        t = jnp.atleast_1d(t)

        P = jax.vmap(transition_matrix, in_axes=(None, None, 0))(
            self.drift_matrix, 0.0, t
        )

        if P.ndim == 3:
            return jnp.einsum("...ij,...j->...i", P, mu0)
        else:
            return P * mu0

    def covariance_matrix(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        assert (
            self.p0.event_shape != ()
        ), "Initial distribution must not be scalar, use var instead"
        Phi, Q = jax.vmap(matrix_fraction_decomposition, in_axes=(0, None, None, None))(
            t, 0.0, self.drift_matrix, self.diffusion_matrix
        )

        cov0 = self.p0.covariance_matrix

        return jnp.matmul(Phi, jnp.matmul(cov0, Phi.T)) + Q

    def variance(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        t = jnp.atleast_1d(t)
        Phi, Q = jax.vmap(matrix_fraction_decomposition, in_axes=(None, 0, None, None))(
            0.0, t, self.drift_matrix, self.diffusion_matrix
        )
        var0 = self.p0.variance
        var = Phi**2 * var0 + Q
        return jnp.squeeze(var, axis=-1)

    def std(self, t: Array):
        return jnp.sqrt(self.variance(t))


class LinearTimeVariantSDE(BaseSDE):
    def mean(self, ts: Array, **kwargs) -> Array:
        assert jnp.all(ts >= 0), "t must be positive"

        _odeint = partial(odeint, **kwargs)
        if self.batch_shape != ():
            _odeint = jax.vmap(_odeint, in_axes=(None, 0, None))
        mu0 = self.p0.mean
        mus = _odeint(self.drift, mu0, ts)

        return mus

    def var(self, t: Array, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        if self.p0.event_shape != ():
            cov = self.covariance_matrix(t, **kwargs)
            return jnp.sum(jnp.diagonal(cov, axis1=-2, axis2=-1))
        else:
            var0 = self.p0.var
            _odeint = partial(odeint, **kwargs)
            if self.batch_shape != ():
                _odeint = jax.vmap(_odeint, in_axes=(None, 0, None))

            def f(t, var):
                return self.drift(t) ** 2 * var + self.diffusion(t) ** 2

            vars = _odeint(f, var0, t)
            return vars

    def covariance_matrix(self, t: Array, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        assert (
            self.p0.event_shape != ()
        ), "Initial distribution must not be scalar, use var instead"
        _odeint = partial(odeint, **kwargs)
        if self.batch_shape != ():
            _odeint = jax.vmap(_odeint, in_axes=(None, 0, None))
        cov0 = self.p0.covariance_matrix

        def f(t, cov):
            term1 = jnp.matmul(self.drift(t), cov)
            term2 = jnp.matmul(cov, self.drift(t).T)
            term3 = jnp.matmul(self.diffusion(t), self.diffusion(t).T)
            return term1 + term2 + term3

        covs = _odeint(f, cov0, t)
        return covs
