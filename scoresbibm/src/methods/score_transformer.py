import jax
import jax.numpy as jnp

import optax

from functools import partial
from probjax.nn.loss_fn import denoising_score_matching_loss
from probjax.utils.graph import faithfull_mask, min_faithfull_mask
from scoresbibm.src.methods.sde import init_sde_related
from scoresbibm.src.methods.models import AllConditionalScoreModel
from scoresbibm.src.methods.neural_nets import scalar_transformer_model


def run_train_transformer_model(
    key,
    params,
    opt_state,
    data,
    node_id,
    total_number_steps,
    batch_size,
    update,
    batch_sampler,
    loss_fn,
    print_every=100,
    val_every=100,
    validation_fraction=0.05,
    val_repeat=2,
    val_error_ratio=1.1,
):
    # Set up stuff for multi-device training
    num_devices = jax.device_count()
    batch_size_per_device = batch_size // num_devices

    # Validation loss
    data_val, data_train = jnp.split(
        data, [max(int(validation_fraction * data.shape[0]), 1)], axis=0
    )
    data_val = jnp.repeat(
        data_val, val_repeat, axis=0
    )  # Multiple Monte Carlo samples for validation loss
    sampler = partial(
        batch_sampler, data=data_train, node_id=node_id, num_devices=num_devices
    )

    # Replicated for multiple devices
    replicated_params = jax.tree_map(lambda x: jnp.array([x] * num_devices), params)
    replicated_opt_state = jax.tree_map(
        lambda x: jnp.array([x] * num_devices), opt_state
    )

    early_stopping_counter = 0

    l_val = None
    l_train = None
    min_l_val = 1e10
    early_stopping_params = None

    for j in range(total_number_steps):
        key, key_batch, key_update, key_val = jax.random.split(key, 4)
        data_batch, node_id_batch = sampler(key_batch, batch_size_per_device)
        loss, replicated_params, replicated_opt_state = update(
            replicated_params,
            replicated_opt_state,
            jax.random.split(key_update, (num_devices,)),
            data_batch,
            node_id_batch,
        )
        # Train loss
        if j == 0:
            l_train = loss[0]
        else:
            l_train = 0.9 * l_train + 0.1 * loss[0]

        # Validation loss
        if validation_fraction > 0 and ((j % val_every) == 0) and j > 50:
            l_val = loss_fn(
                jax.tree_map(lambda x: x[0], replicated_params),
                key_val,
                data_val,
                node_id,
            )

            if l_val / l_train > val_error_ratio:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if l_val < min_l_val:
                min_l_val = l_val
                early_stopping_params = jax.tree_map(lambda x: x[0], replicated_params)

        if early_stopping_counter > 5:
            return early_stopping_params, jax.tree_map(
                lambda x: x[0], replicated_opt_state
            )

        # Print
        if (j % print_every) == 0:
            print("Train loss: ", l_train)
            if l_val is not None:
                print("Validation loss: ", l_val, early_stopping_counter)

    params = jax.tree_map(lambda x: x[0], replicated_params)
    opt_state = jax.tree_map(lambda x: x[0], replicated_opt_state)
    
    del replicated_opt_state
    del replicated_params
    
    return params, opt_state


partial(jax.jit, static_argnums=(1, 4))
def base_batch_sampler(key, batch_size, data, node_id, num_devices=1):
    assert data.ndim == 3, "Data must be 3D, (num_samples, num_nodes, dim)"
    assert (
        node_id.ndim == 2 or node_id.ndim == 1
    ), "Node id must be 2D or 1D, (num_nodes, dim) or (num_nodes,)"

    data_batch = jax.random.choice(key, data, shape=(num_devices, batch_size), axis=0)
    node_id_batch = jnp.repeat(node_id[None, ...], num_devices, axis=0).astype(
        jnp.int32
    )

    return data_batch, node_id_batch


def get_edge_mask_fn(name, task):
    base_mask_fn = task.get_base_mask_fn()
    if name.lower() == "faithfull":

        def faithfull_edge_mask(node_id, condition_mask):
            base_mask = base_mask_fn(node_id, None)
            return jax.vmap(faithfull_mask, in_axes=(None, 0))(
                base_mask, condition_mask
            )

        return faithfull_edge_mask
    elif name.lower() == "min_faithfull":
        def min_faithfull_edge_mask(node_id, condition_mask):
            base_mask = base_mask_fn(node_id, None)
    
            return jax.vmap(min_faithfull_mask, in_axes=(None, 0))(
                base_mask, condition_mask
            )

        return min_faithfull_edge_mask
    elif name.lower() == "none":
        return lambda node_id, condition_mask, *args: None
    else:
        raise NotImplementedError()


def sample_random_conditional_mask(
    key, num_samples, theta_dim, x_dim, alpha=1.0, beta=4.0
):
    # More likely to condition on a few nodes
    key1, key2 = jax.random.split(key, 2)
    condition_mask = jax.random.bernoulli(
        key1,
        jax.random.beta(key2, alpha, beta, shape=(num_samples, 1)),
        shape=(num_samples, theta_dim + x_dim),
    ).astype(jnp.bool_)
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask


def joint_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[False] * (theta_dim + x_dim)]*num_samples)


def posterior_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[False] * theta_dim + [True] * x_dim]*num_samples)


def likelihood_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[True] * theta_dim + [False] * x_dim]*num_samples)


def sample_strutured_conditional_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    p_joint=0.2,
    p_posterior=0.2,
    p_likelihood=0.2,
    p_rnd1=0.2,
    p_rnd2=0.2,
    rnd1_prob=0.3,
    rnd2_prob=0.7,
):
    # Joint, posterior, likelihood, random1_mask, random2_mask
    key1, key2, key3 = jax.random.split(key, 3)
    condition_mask = jax.random.choice(
        key1,
        jnp.array(
            [[False] * (theta_dim + x_dim)]
            + [[False] * theta_dim + [True] * x_dim]
            + [
                [True] * theta_dim + [False] * x_dim,
                jax.random.bernoulli(
                    key2, rnd1_prob, shape=(theta_dim + x_dim,)
                ).astype(jnp.bool_),
                jax.random.bernoulli(
                    key3, rnd2_prob, shape=(theta_dim + x_dim,)
                ).astype(jnp.bool_),
            ]
        ),
        shape=(num_samples,),
        p=jnp.array([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2]),
        axis=0,
    )
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask


def get_condition_mask_fn(name, **kwargs):
    if name.lower() == "structured_random":
        return partial(sample_strutured_conditional_mask, **kwargs)
    elif name.lower() == "random":
        return partial(sample_random_conditional_mask, **kwargs)
    elif name.lower() == "joint":
        return partial(joint_conditional_mask, **kwargs)
    elif name.lower() == "posterior":
        return partial(posterior_conditional_mask, **kwargs)
    elif name.lower() == "likelihood":
        return partial(likelihood_conditional_mask, **kwargs)
    else:
        raise NotImplementedError()


def train_transformer_model(task, thetas, xs, method_cfg, rng):
    device = method_cfg.device
    sde_params = dict(method_cfg.sde)
    model_params = dict(method_cfg.model)
    train_params = dict(method_cfg.train)

    # Data
    data = jnp.hstack([thetas, xs])
    theta_dim = thetas.shape[-1]
    x_dim = xs.shape[-1]
    data = data[..., None]

    # Initialize stuff
    sde, T_min, T_max, _weight_fn, output_scale_fn = init_sde_related(
        data, name=sde_params.pop("name"), **sde_params
    )
    weight_fn = lambda t: _weight_fn(t).reshape(-1, 1, 1)
    if not model_params.pop("use_output_scale_fn", True):
        output_scale_fn = None
    init_fn, model_fn = scalar_transformer_model(
        theta_dim + x_dim, output_scale_fn=output_scale_fn, **model_params
    )

    rng, rng_init = jax.random.split(rng)
    node_id = jnp.arange(theta_dim + x_dim)
    params = init_fn(
        rng_init, jnp.ones((10,)), data[:10], node_id, jnp.zeros_like(data[:10])
    )

    # Training params
    total_number_steps = int(
        max(
            min(
                data.shape[0] * train_params["total_number_steps_scaling"],
                train_params["max_number_steps"],
            ),
            train_params["min_number_steps"],
        )
    )
    batch_size = train_params["training_batch_size"]

    print_every = total_number_steps // 10
    val_every = total_number_steps // train_params["val_every"]
    learning_rate = train_params["learning_rate"]
    schedule = optax.linear_schedule(
        learning_rate, train_params["min_learning_rate"], total_number_steps // 2, total_number_steps // 2
    )
    optimizer = optax.chain(
        optax.adaptive_grad_clip(train_params["clip_max_norm"]), optax.adam(schedule)
    )
    opt_state = optimizer.init(params)

    condition_mask_params = dict(train_params["condition_mask_fn"])
    edge_mask_params = dict(train_params["edge_mask_fn"])
    
    # Get possible condition and edge mask functions
    condition_mask_fn = get_condition_mask_fn(
        condition_mask_params.pop("name", "structured"), **condition_mask_params
    )
    edge_mask_fn = get_edge_mask_fn(edge_mask_params.pop("name"), task)

    # Training loop
    def loss_fn(params, key, data, node_id):
        key_times, key_loss, key_condition = jax.random.split(key, 3)
        times = jax.random.uniform(
            key_times, (data.shape[0],), minval=T_min, maxval=T_max
        )

        # Structured conditioning
        condition_mask = condition_mask_fn(
            key_condition, data.shape[0], theta_dim, x_dim
        )
        edge_mask = edge_mask_fn(node_id, condition_mask)

        loss = denoising_score_matching_loss(
            params,
            key_loss,
            times,
            data,
            loss_mask=condition_mask,
            model_fn=model_fn,
            mean_fn=sde.marginal_mean,
            std_fn=sde.marginal_stddev,
            weight_fn=weight_fn,
            data_id=node_id,
            condition_mask=condition_mask,
            edge_mask=edge_mask,
        )
        return loss

    @partial(jax.pmap, axis_name="num_devices")
    def update(params, opt_state, key, data, node_id):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, data, node_id)

        loss = jax.lax.pmean(loss, axis_name="num_devices")
        grads = jax.lax.pmean(grads, axis_name="num_devices")

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    rng, rng_train = jax.random.split(rng)
    batch_sampler = partial(base_batch_sampler, data=data, node_id=node_id)
    params, opt_state = run_train_transformer_model(
        rng_train,
        params,
        opt_state,
        data,
        node_id,
        total_number_steps,
        batch_size,
        update,
        batch_sampler,
        loss_fn,
        print_every=print_every,
        val_every=val_every,
        validation_fraction=train_params["validation_fraction"],
        val_repeat=train_params["val_repeat"],
    )

    sde_init_params = {"data": jax.device_put(data, jax.devices("cpu")[0]) , **dict(method_cfg.sde)}
    model_init_params = {"num_nodes": theta_dim + x_dim, **dict(method_cfg.model)}
    model = AllConditionalScoreModel(
        params,
        model_fn,
        sde,
        sde_init_params=sde_init_params,
        model_init_params=model_init_params,
    )
    # Posterior as default
    default_conditon_mask = jnp.array([0] * theta_dim + [1] * x_dim, dtype=jnp.bool_)
    model.set_default_condition_mask(default_conditon_mask)
    model.set_default_node_id(node_id)
    model.set_default_edge_mask_fn(edge_mask_fn)

    return model
