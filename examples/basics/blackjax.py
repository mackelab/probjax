# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from datetime import date

import jax
from jax import numpy as jnp

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

# %%
hmc_parameters = dict(
    step_size=1e-4, inverse_mass_matrix=jnp.ones((1,)), num_integration_steps=1
)

# %%
n_particles = 5000
import matplotlib.pyplot as plt 
plt.plot(jnp.arange(10))
# %%
from jax.scipy.stats import multivariate_normal


def V(x):
    return 5 * jnp.sum(jnp.square(x**2 - 1))


def prior_log_prob(x):
    d = x.shape[0]
    return multivariate_normal.logpdf(x, jnp.zeros((d,)), jnp.eye(d))


loglikelihood = lambda x: -V(x)


def density():
    linspace = jnp.linspace(-2, 2, 5000).reshape(-1, 1)
    lambdas = jnp.linspace(0.0, 1.0, 5)
    prior_logvals = jnp.vectorize(prior_log_prob, signature="(d)->()")(linspace)
    potential_vals = jnp.vectorize(V, signature="(d)->()")(linspace)
    log_res = prior_logvals.reshape(1, -1) - jnp.expand_dims(
        lambdas, 1
    ) * potential_vals.reshape(1, -1)

    density = jnp.exp(log_res)
    normalizing_factor = jnp.sum(density, axis=1, keepdims=True) * (
        linspace[1] - linspace[0]
    )
    density /= normalizing_factor
    return density


# %%
def initial_particles_multivariate_normal(dimensions, key, n_samples):
    return jax.random.multivariate_normal(
        key, jnp.zeros(dimensions), jnp.eye(dimensions) * 2, (n_samples,)
    )


# %%
from blackjax import adaptive_tempered_smc, irmh
from blackjax.smc import extend_params, solver
from blackjax.smc import resampling as resampling


def irmh_experiment(dimensions, target_ess, num_mcmc_steps):
    mean = jnp.zeros(dimensions)
    cov = jnp.diag(jnp.ones(dimensions)) * 2

    def irmh_proposal_distribution(rng_key):
        return jax.random.multivariate_normal(rng_key, mean, cov)

    def proposal_logdensity_fn(proposal, state):
        return jnp.log(
            jax.scipy.stats.multivariate_normal.pdf(state.position, mean=mean, cov=cov)
        )
    def step(key, state, logdensity):
        return irmh(logdensity, irmh_proposal_distribution,proposal_logdensity_fn).step(key, state)

    fixed_proposal_kernel = adaptive_tempered_smc(
        prior_log_prob,
        loglikelihood,
        step,
        irmh.init,
        mcmc_parameters={},
        resampling_fn=resampling.systematic,
        target_ess=target_ess,
        root_solver=solver.dichotomy,
        num_mcmc_steps=num_mcmc_steps,
    )

    def inference_loop(kernel, rng_key, initial_state):
        def cond(carry):
            _, state, *_ = carry
            return state.lmbda < 1

        def body(carry):
            i, state, op_key, curr_loglikelihood = carry
            op_key, subkey = jax.random.split(op_key, 2)
            state, info = kernel(subkey, state)
            return (
                i + 1,
                state,
                op_key,
                curr_loglikelihood + info.log_likelihood_increment,
            )

        total_iter, final_state, _, log_likelihood = jax.lax.while_loop(
            cond, body, (0, initial_state, rng_key, 0.0)
        )

        return total_iter, final_state.particles

    return fixed_proposal_kernel, inference_loop


# %%
from blackjax.smc.inner_kernel_tuning import inner_kernel_tuning
from blackjax.smc.tuning.from_particles import (
    particles_means,
    particles_stds,
)


def tuned_irmh_loop(kernel, rng_key, initial_state):
    def cond(carry):
        _, state, *_ = carry
        return state.sampler_state.lmbda < 1

    def body(carry):
        i, state, op_key = carry
        op_key, subkey = jax.random.split(op_key, 2)
        state, info = kernel(subkey, state)
        return i + 1, state, op_key

    def f(initial_state, key):
        total_iter, final_state, _ = jax.lax.while_loop(
            cond, body, (0, initial_state, key)
        )
        return total_iter, final_state

    total_iter, final_state = f(initial_state, rng_key)
    return total_iter, final_state.sampler_state.particles


def tuned_irmh_experiment(dimensions, target_ess, num_mcmc_steps):
    kernel = irmh.build_kernel()
    def step_fn(key, state, logdensity, means, stds):
        cov = jnp.square(jnp.diag(stds))
        proposal_distribution = lambda key: jax.random.multivariate_normal(
            key, means, cov
        )

        def proposal_logdensity_fn(proposal, state):
            return jnp.log(
                jax.scipy.stats.multivariate_normal.pdf(
                    state.position, mean=means, cov=cov
                )
            )

        return kernel(key, state, logdensity, proposal_distribution, proposal_logdensity_fn)


    kernel_tuned_proposal = inner_kernel_tuning(
        logprior_fn=prior_log_prob,
        loglikelihood_fn=loglikelihood,
        mcmc_step_fn=step_fn,
        mcmc_init_fn=irmh.init,
        resampling_fn=resampling.systematic,
        smc_algorithm=adaptive_tempered_smc,
        mcmc_parameter_update_fn=lambda state, info: extend_params(n_particles,
                                                                                {"means":particles_means(state.particles),
                                                                                 "stds":particles_stds(state.particles)}),
        initial_parameter_value=extend_params(n_particles, {"means":jnp.zeros(dimensions), "stds":jnp.ones(dimensions) * 2}),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    return kernel_tuned_proposal, tuned_irmh_loop
