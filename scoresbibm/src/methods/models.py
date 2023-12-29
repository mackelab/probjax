from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from probjax.distributions.sde import init_sde_related

from probjax.utils.sdeint import sdeint
from scoresbibm.src.methods.neural_nets import conditional_mlp, scalar_transformer_model
from scoresbibm.src.methods.sde import init_sde_related


class Model(ABC):
    """
    Abstract base class for models.

    Args:
        method (str): The method used by the model.
        backend (str, optional): The backend used for computation. Defaults to "jax".
    """

    def __init__(self, method: str, backend="jax"):
        self.method = method
        self.backend = backend

    @abstractmethod
    def sample(self, num_samples, x_o, rng=None, **kwargs):
        pass

    @abstractmethod
    def log_prob(self, theta, x_o, **kwargs):
        pass


class PosteriorModel(Model):
    """
    A base class for posterior models.
    """

    def set_default_x_o(self, x_o):
        """
        Set the default value for x_o.

        Args:
            x_o: The default value for x_o.
        """
        self.x_o = x_o

    def _check_x_o(self, x_o):
        """
        Check if x_o is provided, otherwise use the default value.

        Args:
            x_o: The value of x_o.

        Returns:
            The value of x_o.

        Raises:
            ValueError: If x_o is not provided and no default value is set.
        """
        if x_o is None:
            x_o = self.x_o
            if x_o is None:
                raise ValueError(
                    "Please provide x_o, either as argument or by calling set_default_x_o"
                )
        return x_o

    def sample(self, num_samples, x_o=None, rng=None, **kwargs):
        """
        Sample from the posterior model.

        Args:
            num_samples: The number of samples to generate.
            x_o: The value of x_o.
            rng: The random number generator.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated samples.
        """
        x_o = self._check_x_o(x_o)
        return self._sample(num_samples, x_o=x_o, rng=rng, **kwargs)

    def log_prob(self, theta, x_o=None, **kwargs):
        """
        Compute the log probability of theta given x_o.

        Args:
            theta: The value of theta.
            x_o: The value of x_o.
            **kwargs: Additional keyword arguments.

        Returns:
            The log probability.
        """
        x_o = self._check_x_o(x_o)
        return self._log_prob(theta, x_o=x_o, **kwargs)

    @abstractmethod
    def _sample(self, num_samples, x_o, rng=None, **kwargs):
        """
        Abstract method for sampling from the posterior model.

        Args:
            num_samples: The number of samples to generate.
            x_o: The value of x_o.
            rng: The random number generator.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def _log_prob(self, theta, x_o, **kwargs):
        """
        Abstract method for computing the log probability of theta given x_o.

        Args:
            theta: The value of theta.
            x_o: The value of x_o.
            **kwargs: Additional keyword arguments.
        """
        pass


class AllConditionalModel(PosteriorModel):
    """
    A class representing an AllConditionalModel.

    This class extends the PosteriorModel class and provides additional functionality for handling all conditional distributions.

    Args:
        method (str): The method used for modeling.
        backend (str, optional): The backend used for computation. Defaults to "jax".

    Attributes:
        node_id: The default node ID.
        condition_mask: The default condition mask.

    Methods:
        set_default_node_id: Sets the default node ID.
        set_default_condition_mask: Sets the default condition mask.
        _check_id_condition_mask: Checks and retrieves the node ID and condition mask.
        sample: Samples from the model.
        log_prob: Computes the log probability of the model.

    """

    def __init__(self, method: str, backend="jax"):
        super().__init__(method, backend)

        self.node_id = None
        self.condition_mask = None

    def set_default_node_id(self, node_id):
        """
        Sets the default node ID.

        Args:
            node_id: The default node ID.

        """
        self.node_id = node_id

    def set_default_condition_mask(self, condition_mask):
        """
        Sets the default condition mask.

        Args:
            condition_mask: The default condition mask.

        """
        self.condition_mask = condition_mask

    def _check_id_condition_mask(self, node_id, condition_mask):
        """
        Checks and retrieves the node ID and condition mask.

        If the node ID or condition mask is not provided, it retrieves the default values.

        Args:
            node_id: The node ID.
            condition_mask: The condition mask.

        Returns:
            node_id: The node ID.
            condition_mask: The condition mask.

        Raises:
            ValueError: If the node ID or condition mask is not provided.

        """
        if node_id is None:
            node_id = self.node_id
            if node_id is None:
                raise ValueError(
                    "Please provide node_id, either as argument or by calling set_default_node_id"
                )

        if condition_mask is None:
            condition_mask = self.condition_mask
            if condition_mask is not None:
                condition_mask = condition_mask.astype(jnp.bool_)
            else:
                raise ValueError(
                    "Please provide condition_mask, either as argument or by calling set_default_condition_mask"
                )
        return node_id, condition_mask

    def sample(
        self,
        num_samples,
        x_o=None,
        rng=None,
        node_id=None,
        condition_mask=None,
        **kwargs
    ):
        """
        Samples from the model.

        Args:
            num_samples: The number of samples to generate.
            x_o: The observed data.
            rng: The random number generator.
            node_id: The node ID.
            condition_mask: The condition mask.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated samples.

        """
        node_id, condition_mask = self._check_id_condition_mask(node_id, condition_mask)
        return super().sample(
            num_samples,
            x_o,
            rng=rng,
            node_id=node_id,
            condition_mask=condition_mask,
            **kwargs
        )

    def log_prob(self, theta, x_o=None, node_id=None, condition_mask=None, **kwargs):
        """
        Computes the log probability of the model.

        Args:
            theta: The model parameters.
            x_o: The observed data.
            node_id: The node ID.
            condition_mask: The condition mask.
            **kwargs: Additional keyword arguments.

        Returns:
            The log probability.

        """
        node_id, condition_mask = self._check_id_condition_mask(node_id, condition_mask)
        return super().log_prob(
            theta, x_o, node_id=node_id, condition_mask=condition_mask, **kwargs
        )


class SBIPosteriorModel(PosteriorModel):
    def __init__(self, sbi_posterior, method: str):
        super().__init__(method, backend="torch")
        self.sbi_posterior = sbi_posterior

    def _sample(self, num_samples, x_o, rng=None, **kwargs):
        if self.method in ["npe", "nle", "nre"]:
            return self.sbi_posterior.sample((num_samples,), x=x_o)
        else:
            raise NotImplementedError()

    def _log_prob(self, theta, x_o, **kwargs):
        if self.method == "npe":
            return self.sbi_posterior.log_prob(theta, x=x_o)
        else:
            raise NotImplementedError()


class ScorePosteriorModel(PosteriorModel):
    """A class for the NPSE model. This is a wrapper around the sde and the model. The model is a conditional score model."""

    def __init__(
        self, params, model_fn, sde, model_init_params={}, sde_init_params={}
    ) -> None:
        self.params = params
        self.model_fn = model_fn
        self.sde = sde

        # For sampling
        self.T_min = sde_init_params["T_min"]
        self.T_max = sde_init_params["T_max"]
        self.marginal_end_std = sde.marginal_stddev(jnp.array([self.T_min]))
        self.marginal_end_mean = sde.marginal_mean(jnp.array([self.T_max]))

        # For pickle
        self.model_init_params = model_init_params
        self.sde_init_params = sde_init_params

        super().__init__("npse", backend="jax")

    def _sample(self, num_samples, x_o, num_steps=500, rng=None, **kwargs):
        assert rng is not None, "Please provide a rng key"
        key1, key2 = jax.random.split(rng, 2)
        drift, diffusion = self._init_backward_sde(x_o)
        x_T = (
            jax.random.normal(key1, (num_samples,) + self.sde.event_shape)
            * self.marginal_end_std
            + self.marginal_end_mean
        )
        keys = jax.random.split(key2, (num_samples,))
        ys = jax.vmap(
            lambda *args: sdeint(*args, noise_type="diagonal", **kwargs),
            in_axes=(0, None, None, 0, None),
            out_axes=0,
        )(
            keys,
            drift,
            diffusion,
            x_T,
            jnp.linspace(0.0, self.T_max - self.T_min, num_steps),
        )
        return ys[:, -1, ...]

    def _log_prob(self, val, x_o, **kwargs):
        # Add backward ode to compute log_prob
        raise NotImplementedError()

    def _init_backward_sde(self, x_o):
        def drift_backward(t, x):
            t = self.T_max - t
            score = self.model_fn(self.params, jnp.atleast_1d(t), x, jnp.squeeze(x_o))
            drift = self.sde.drift(t, x) - self.sde.diffusion(t, x) ** 2 * score
            return -drift.reshape(x.shape)

        def diffusion_backward(t, x):
            t = self.T_max - t
            return self.sde.diffusion(t, x).reshape(x.shape)

        return drift_backward, diffusion_backward

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state["model_fn"] = None
        state["sde"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sde, self.T_min, self.T_max, _, output_scale_fn = init_sde_related(
            **self.sde_init_params
        )
        _, self.model_fn = conditional_mlp(
            output_scale_fn=output_scale_fn, **self.model_init_params
        )


class AllConditionalReferenceModel(AllConditionalModel):
    def __init__(self, sampling_fn, log_prob_fn=None) -> None:
        self.sampling_fn = sampling_fn
        self.log_prob_fn = log_prob_fn
        return super().__init__("reference", backend="jax")

    def _sample(self, num_samples, x_o, rng=None, **kwargs):
        return self.sampling_fn(num_samples, x_o, rng=rng, **kwargs)

    def _log_prob(self, theta, x_o, **kwargs):
        return self.log_prob_fn(theta, x_o, **kwargs)


class AllConditionalScoreModel(AllConditionalModel):
    def __init__(
        self, params, model_fn, sde, sde_init_params, model_init_params
    ) -> None:
        self.params = params
        self.model_fn = model_fn
        self.sde = sde

        self.T_min = sde_init_params["T_min"]
        self.T_max = sde_init_params["T_max"]

        self.marginal_end_std = jnp.squeeze(
            sde.marginal_stddev(jnp.array([self.T_max]))
        )
        self.marginal_end_mean = jnp.squeeze(sde.marginal_mean(jnp.array([self.T_max])))

        # For sampling
        self.edge_mask = None
        self.edge_mask_fn = None
        self.score_fn = self.model_fn  # For score modifcations ...

        # For pickle
        self.model_init_params = model_init_params
        self.sde_init_params = sde_init_params

        super().__init__("score_transformer", backend="jax")

    def _check_edge_mask(self, edge_mask, node_id, condition_mask):
        if edge_mask is None:
            if self.edge_mask_fn is not None:
                edge_mask = self.edge_mask_fn(node_id, condition_mask[None, ...])
        return edge_mask

    def _sample(
        self,
        num_samples,
        x_o,
        num_steps=500,
        node_id=None,
        condition_mask=None,
        edge_mask=None,
        rng=None,
        **kwargs
    ):
        edge_mask = self._check_edge_mask(edge_mask, node_id, condition_mask)
        key1, key2 = jax.random.split(rng, 2)
        drift, diffusion = self._init_backward_sde(node_id, condition_mask, edge_mask)
        x_T = (
            jax.random.normal(
                key1,
                (
                    num_samples,
                    node_id.shape[-1],
                ),
            )
            * self.marginal_end_std
            + self.marginal_end_mean
        )
        condition_mask = condition_mask.reshape(x_T.shape[-1])
        x_T = x_T.at[..., condition_mask].set(x_o.reshape(-1))
        keys = jax.random.split(key2, (num_samples,))
        ys = jax.vmap(
            lambda *args: sdeint(*args, noise_type="diagonal", **kwargs),
            in_axes=(0, None, None, 0, None),
            out_axes=0,
        )(
            keys,
            drift,
            diffusion,
            x_T,
            jnp.linspace(0.0, self.T_max - self.T_min, num_steps),
        )
        final_samples = ys[:, -1, ...][:, ~condition_mask]
        final_samples = final_samples.reshape((num_samples, -1))
        return final_samples

    def _log_prob(self, val, x_o, **kwargs):
        # Add backward ode to compute log_prob
        raise NotImplementedError()

    def set_default_edge_mask_fn(self, edge_mask_fn):
        self.edge_mask_fn = edge_mask_fn

    def set_default_score_fn(self, score_fn):
        self.score_fn = score_fn

    def _init_backward_sde(self, node_id=None, condition_mask=None, edge_mask=None):
        def drift_backward(t, x):
            t = self.T_max - t
            score = self.score_fn(
                self.params,
                jnp.atleast_1d(t),
                x.reshape(-1, x.shape[-1], 1),
                node_id,
                condition_mask,
                edge_mask=edge_mask,
            ).reshape(x.shape)
            drift = self.sde.drift(t, x) - self.sde.diffusion(t, x) ** 2 * score
            return -drift.reshape(x.shape) * (1 - condition_mask.reshape(x.shape))

        def diffusion_backward(t, x):
            t = self.T_max - t
            return self.sde.diffusion(t, x).reshape(x.shape) * (
                1 - condition_mask.reshape(x.shape)
            )

        return drift_backward, diffusion_backward

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state["model_fn"] = None
        state["sde"] = None
        state["score_fn"] = None
        state["edge_mask_fn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sde, self.T_min, self.T_max, _, output_scale_fn = init_sde_related(
            **self.sde_init_params
        )
        _, self.model_fn = scalar_transformer_model(
            output_scale_fn=output_scale_fn, **self.model_init_params
        )
        self.score_fn = self.model_fn
