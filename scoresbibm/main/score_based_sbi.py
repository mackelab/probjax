

import jax 
import jax.numpy as jnp


import haiku as hk
import optax

from probjax.nn.transformers import Transformer
from probjax.nn.tokenizer import scalarize, ScalarTokenizer
from probjax.nn.helpers import GaussianFourierEmbedding
from probjax.nn.loss_fn import denoising_score_matching_loss

from probjax.distributions.sde import VPSDE, BaseSDE
from probjax.distributions import Normal, Independent
from probjax.distributions.transformed_distribution import TransformedDistribution
from probjax.distributions.discrete import Empirical

from probjax.utils.sdeint import sdeint

from functools import partial


def conditional_mlp(output_dim:int, hidden_dim:int = 100, num_hidden:int=8, activation=jax.nn.gelu, layer_norm:bool = True, output_scale_fn=None):
    """ Just builds a conditional score model with MLPs. As the score is typically grows proportional to the variance of the marginal sde, it is useful to scale the output."""
    
    if output_scale_fn is None:
        output_scale_fn = lambda t, x: x
    
    def score_net(t, x, context):
        
        #x, context = jnp.broadcast_arrays(x, context)
        x = jnp.concatenate([x, context], axis=-1)
        
        time_embedding = GaussianFourierEmbedding(hidden_dim)(t[...,None])
        h = activation(hk.Linear(hidden_dim)(x) + time_embedding)
        
        
        for _ in range(num_hidden - 1):
            h_new = hk.Linear(hidden_dim)(h)
            h_new += time_embedding
            h = activation(h_new)
            
            if layer_norm:
                h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
            
        out = hk.Linear(output_dim)(h)
        out = output_scale_fn(t, out)
        return out
    
    init_fn, apply_fn = hk.without_apply_rng(hk.transform(score_net))
    return init_fn, apply_fn



def transformer_model(num_nodes, dim = 40, condition_token_dim=10, num_heads=8, num_layers=6, attn_size=5, widening_factor=4, output_scale_fn=None):
    
    if output_scale_fn is None:
        output_scale_fn = lambda t, x: x
    
    def model(t,data, data_id, condition_mask):
        batch_size, current_nodes, _ = data.shape
        data_id = data_id.reshape(-1,current_nodes)
        condition_mask = condition_mask.reshape(-1,current_nodes)
        

        tokenizer = ScalarTokenizer(dim, num_nodes)
        time_embeder = GaussianFourierEmbedding(128)
        
        # Embedding
        tokens = tokenizer(data_id,data)
        time = time_embeder(t[..., None])
        
        # Conditioning
        condition_token = hk.get_parameter("condition_token", shape=[1,1,condition_token_dim], init=hk.initializers.RandomNormal(0.01))
        condition_mask = condition_mask.reshape(-1,data.shape[-2],1)
        condition_token = condition_mask * condition_token
        condition_token = jnp.broadcast_to(condition_token, tokens.shape[:-1] + (condition_token_dim,))
        tokens = jnp.concatenate([tokens, condition_token], -1)
        
        
        # Forward pass 
        model = Transformer(num_heads=num_heads, num_layers=num_layers, attn_size=attn_size, widening_factor=widening_factor)

        h = model(tokens, context=time)
        out = hk.Linear(1)(h)
        out = output_scale_fn(t, out)
        return out
    
    init_fn, model_fn = hk.without_apply_rng(hk.transform(model))
    return init_fn, model_fn


def init_sde_related(data, name="vpsde", **kwargs):
    """ Initialize the sde and related functions."""
    # VPSDE 
    if name.lower() == "vpsde":
        p0 = Independent(Empirical(data), 1)
        beta_max = kwargs.get("beta_max",10.)
        beta_min = kwargs.get("beta_min", 0.01)
        sde = VPSDE(p0, beta_max=beta_max, beta_min=beta_min)
        T_max = kwargs.get("T_max", 1.)
        T_min = kwargs.get("T_min", 1e-5)

        # Train weight function
        def weight_fn(t):
            t = t.reshape(-1, 1)
            return jnp.clip(1-jnp.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t) ,a_min = 1e-4)
        
        # Model output scale function
        def output_scale_fn(t, x):
            scale = jnp.sqrt(jnp.sum(sde.marginal_variance(t[..., None], x0=jnp.ones_like(x)), -1))
            return 1/scale[..., None] * x
        
    else:
        raise NotImplementedError()
    
    return sde, T_min, T_max, weight_fn, output_scale_fn



def run_train_conditional_score_model(key, params, opt_state, data, num_epochs, num_steps, batch_size,  update, print_every=100):
    """ Runs the training loop for the conditional score model. Assumes update is compiled using jax.pmap."""
    
    # Set up stuff for multi-device training
    num_devices = jax.device_count()
    batch_size_per_device = batch_size // num_devices
    # Replicated for multiple devices
    replicated_params = jax.tree_map(lambda x: jnp.array([x] * num_devices), params)
    replicated_opt_state = jax.tree_map(lambda x: jnp.array([x] * num_devices), opt_state)

    for j in range(num_epochs):
        l = 0
        for i in range(num_steps):
            key, key_batch, key_update = jax.random.split(key, 3)
            data_batch = jax.random.choice(key_batch,data, shape=(num_devices, batch_size_per_device, ), axis=0, replace=True)
            loss, replicated_params, replicated_opt_state = update(replicated_params, replicated_opt_state, jax.random.split(key_update, (num_devices,)), data_batch)
            l += loss[0] /num_steps
        if (j % print_every) == 0:     
            print("Train loss: ",l)
            
    params = jax.tree_map(lambda x: x[0], replicated_params)
    
    return params


class NPSE:
    """ A class for the NPSE model. This is a wrapper around the sde and the model. The model is a conditional score model."""
    def __init__(self, params, model_fn, sde, model_init_params={}, sde_init_params={}) -> None:
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
        
        
        
    def sample(self, num_samples, x_o, num_steps=500, rng =None, **kwargs):
        assert rng is not None, "Please provide a rng key"
        key1, key2 = jax.random.split(rng, 2)
        drift, diffusion = self._init_backward_sde(x_o)
        x_T = jax.random.normal(key1, (num_samples,) + self.sde.event_shape) * self.marginal_end_std + self.marginal_end_mean
        keys = jax.random.split(key2, (num_samples,))
        ys = jax.vmap(lambda *args: sdeint(*args, noise_type="diagonal",**kwargs), in_axes= (0, None, None, 0, None), out_axes=0)(keys, drift, diffusion, x_T, jnp.linspace(0., self.T_max-self.T_min, num_steps))
        return ys[:, -1, ...]
    
    def log_prob(self, val, x_o, **kwargs):
        # Add backward ode to compute log_prob
        raise NotImplementedError()
    
        
    def _init_backward_sde(self, x_o):
        def drift_backward(t, x):
            t = (self.T_max-t)
            score = self.model_fn(self.params, jnp.atleast_1d(t), x, jnp.squeeze(x_o))
            drift = self.sde.drift(t, x)  - self.sde.diffusion(t, x)**2 * score
            return -drift.reshape(x.shape)
        
        def diffusion_backward(t, x):
            t = (self.T_max-t)
            return self.sde.diffusion(t, x).reshape(x.shape)
        
        return drift_backward, diffusion_backward
    
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state["model_fn"] = None 
        state["sde"] = None
        return state    
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sde, self.T_min, self.T_max, _, output_scale_fn = init_sde_related(**self.sde_init_params)
        _, self.model_fn = conditional_mlp(output_scale_fn=output_scale_fn,**self.model_init_params)
    
    

def run_train_transformer_model(key, params, opt_state, data, node_id, num_epochs, num_steps, batch_size,  update, print_every=100):
    
     # Set up stuff for multi-device training
    num_devices = jax.device_count()
    batch_size_per_device = batch_size // num_devices
    
    # Replicated for multiple devices
    replicated_params = jax.tree_map(lambda x: jnp.array([x] * num_devices), params)
    replicated_opt_state = jax.tree_map(lambda x: jnp.array([x] * num_devices), opt_state)

    for j in range(num_epochs):
        l = 0
        for i in range(num_steps):
            key, key_batch, key_update = jax.random.split(key, 3)
            data_batch = jax.random.choice(key_batch,data, shape=(num_devices, batch_size_per_device, ), axis=0)
            node_id_batch = jnp.repeat(node_id[None,...], num_devices, axis=0)
            loss, replicated_params, replicated_opt_state = update(replicated_params, replicated_opt_state, jax.random.split(key_update, (num_devices,)), data_batch, node_id_batch)
            l += loss[0] /num_steps
        if (j % print_every) == 0:     
            print("Train loss: ",l)
            
    params = jax.tree_map(lambda x: x[0], replicated_params)
    
    return params


class ACSE:
    def __init__(self, params, model_fn, sde, sde_init_params, model_init_params) -> None:
        self.params = params
        self.model_fn = model_fn
        self.sde = sde

        self.T_min = sde_init_params["T_min"]
        self.T_max = sde_init_params["T_max"]

        self.marginal_end_std = jnp.squeeze(sde.marginal_stddev(jnp.array([self.T_max])))
        self.marginal_end_mean = jnp.squeeze(sde.marginal_mean(jnp.array([self.T_max])))
        
        self.node_id = None
        self.condition_mask = None
        
        # For pickle 
        self.model_init_params = model_init_params
        self.sde_init_params = sde_init_params
        
        
        
    def sample(self, num_samples, x_o, num_steps=500, node_id = None, condition_mask = None, rng=None, **kwargs):
        if node_id is None:
            node_id = self.node_id
        if condition_mask is None:
            condition_mask = self.condition_mask
            
        assert node_id is not None, "node_id must be provided, or set as default"
        assert condition_mask is not None, "condition_mask must be provided, or set as default"
        
        key1, key2 = jax.random.split(rng, 2)
        drift, diffusion = self._init_backward_sde(node_id, condition_mask)
        x_T = jax.random.normal(key1, (num_samples, node_id.shape[-1],)) * self.marginal_end_std + self.marginal_end_mean
        condition_mask = condition_mask.reshape(x_T.shape[-1])
        x_T = x_T.at[...,condition_mask].set(x_o.reshape(-1))
        keys = jax.random.split(key2, (num_samples,))
        ys = jax.vmap(lambda *args: sdeint(*args, noise_type="diagonal",**kwargs), in_axes= (0, None, None, 0, None), out_axes=0)(keys, drift, diffusion, x_T, jnp.linspace(0., self.T_max-self.T_min, num_steps))
        final_samples = ys[:, -1, ...][:,~condition_mask]
        final_samples = final_samples.reshape((num_samples, -1))
        return final_samples
        
    
    def log_prob(self, val, x_o, **kwargs):
        # Add backward ode to compute log_prob
        raise NotImplementedError()
    
    def set_default_node_id(self, node_id):
        self.node_id = node_id
    
    def set_default_condition_mask(self, condition_mask):
        self.condition_mask = condition_mask

    
        
    def _init_backward_sde(self, node_id = None, condition_mask = None):
        def drift_backward(t, x):
            t = (self.T_max-t)
            score = self.model_fn(self.params, jnp.atleast_1d(t), x.reshape(-1, x.shape[-1], 1), node_id, condition_mask).reshape(x.shape)
            drift = self.sde.drift(t, x)  - self.sde.diffusion(t, x)**2 * score
            return -drift.reshape(x.shape) * (1-condition_mask.reshape(x.shape))
        
        def diffusion_backward(t, x):
            t = (self.T_max-t)
            return self.sde.diffusion(t, x).reshape(x.shape) * (1-condition_mask.reshape(x.shape))
        
        return drift_backward, diffusion_backward
    





        
def train_conditional_score_model(task, thetas,xs, method_cfg, rng):

    device = method_cfg.device
    sde_params = dict(method_cfg.sde_params)
    model_params = dict(method_cfg.model_params)
    train_params = dict(method_cfg.params_train)
    
    
    # Data
    data = jnp.hstack([thetas, xs])
    theta_dim = thetas.shape[-1]
    x_dim = xs.shape[-1]
    
    # Initialize stuff
    sde, T_min,T_max, weight_fn, output_scale_fn = init_sde_related(thetas, name = sde_params.pop("name"), **sde_params)
    if not model_params.pop("use_output_scale_fn", True):
        output_scale_fn = None
    init_fn, model_fn = conditional_mlp(theta_dim, output_scale_fn=output_scale_fn, **model_params)
    
    rng, rng_init = jax.random.split(rng)
    params = init_fn(rng_init, jnp.ones((10,)), thetas[:10], xs[:10])


    # Training params
    total_number_steps = int(data.shape[0] * train_params["total_number_steps_scaling"])
    batch_size = train_params["training_batch_size"]
    num_steps = data.shape[0] // batch_size + 1

    num_epochs = min(total_number_steps // num_steps + 1, train_params["max_num_epochs"])
    total_number_steps = num_epochs * num_steps
    print_every = num_epochs // 10
    learning_rate = train_params["learning_rate"]
    schedule = optax.linear_schedule(learning_rate, 0., total_number_steps//2, total_number_steps//2)
    optimizer = optax.chain(optax.adaptive_grad_clip(train_params["clip_max_norm"]), optax.adam(schedule))
    opt_state = optimizer.init(params)
    
    
    # Training loop
    @jax.jit
    def loss_fn(params, key, data):
        thetas, xs = jnp.split(data, [theta_dim,], axis=-1)
        key_times, key_loss = jax.random.split(key,2)
        times = jax.random.uniform(key_times, (data.shape[0],), minval=T_min, maxval =T_max)
        loss = denoising_score_matching_loss(params, key_loss, times, thetas, None, xs, model_fn = model_fn, mean_fn = sde.marginal_mean, std_fn=sde.marginal_stddev, weight_fn=weight_fn, axis=-1)
        return loss

    @partial(jax.pmap, axis_name="num_devices")
    def update(params, opt_state, key, data):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, data)

        loss = jax.lax.pmean(loss, axis_name="num_devices")  # Syncs loss
        grads = jax.lax.pmean(grads, axis_name="num_devices") # Syncs grads
        
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state
    
    
    rng, rng_train = jax.random.split(rng)
    params = run_train_conditional_score_model(rng_train, params, opt_state, data, num_epochs, num_steps, batch_size, update, print_every=print_every)
    
    sde_init_params = {"data": thetas, **sde_params}
    model_init_params = {"output_dim": theta_dim, **model_params}
    model = NPSE(params, model_fn, sde, sde_init_params=sde_init_params, model_init_params=model_init_params)
   
    return model


def sample_strutured_conditional_mask(key, num_samples, theta_dim, x_dim, p_joint=0.2, p_posterior=0.2, p_likelihood=0.2, p_rnd1 = 0.2, p_rnd2 =0.2):
    # Joint, posterior, likelihood, random1_mask, random2_mask
    key1,key2,key3 = jax.random.split(key,3)
    condition_mask = jax.random.choice(key1, jnp.array([[0] * (theta_dim+x_dim)] + [[0] * theta_dim + [1] * x_dim] + [[1]* theta_dim + [0] * x_dim, jax.random.bernoulli(key2,0.3, shape=(theta_dim + x_dim,)), jax.random.bernoulli(key3,0.7, shape=(theta_dim + x_dim,))] ), shape = (num_samples,), p = jnp.array([p_joint, p_posterior, p_likelihood,p_rnd1, p_rnd2]), axis=0)
    return condition_mask


def sample_random_conditional_mask(key, num_samples, theta_dim, x_dim, alpha=1., beta=4.):
    # More likely to condition on a few nodes
    key1, key2 = jax.random.split(key,2)
    condition_mask = jax.random.bernoulli(key1, jax.random.beta(key2,alpha, beta, shape=(num_samples,1)), shape=(num_samples,theta_dim + x_dim)) 
    return condition_mask

def get_condition_mask_fn(name, **kwargs):
    if name.lower() == "structured":
        return partial(sample_strutured_conditional_mask, **kwargs)
    elif name.lower() == "random":
        return partial(sample_random_conditional_mask, **kwargs)
    else:
        raise NotImplementedError()

def train_transformer_model(task, thetas,xs, method_cfg, rng):

    device = method_cfg.device
    sde_params = dict(method_cfg.sde_params)
    model_params = dict(method_cfg.model_params)
    train_params = dict(method_cfg.params_train)
    
    
    # Data
    data = jnp.hstack([thetas, xs])
    theta_dim = thetas.shape[-1]
    x_dim = xs.shape[-1]
    data = data[..., None]
    
    # Initialize stuff
    sde, T_min,T_max, _weight_fn, output_scale_fn = init_sde_related(data, name = sde_params.pop("name"), **sde_params)
    weight_fn = lambda t: _weight_fn(t).reshape(-1,1,1)
    if not model_params.pop("use_output_scale_fn", True):
        output_scale_fn = None
    init_fn, model_fn = transformer_model(theta_dim + x_dim, output_scale_fn=output_scale_fn, **model_params)
    
    rng, rng_init = jax.random.split(rng)
    node_id = jnp.arange(theta_dim + x_dim)
    params = init_fn(rng_init, jnp.ones((10,)) , data[:10], node_id, jnp.zeros_like(data[:10]))


    # Training params
    total_number_steps = int(data.shape[0] * train_params["total_number_steps_scaling"])
    batch_size = train_params["training_batch_size"]
    num_steps = data.shape[0] // batch_size + 1

    num_epochs = min(total_number_steps // num_steps + 1, train_params["max_num_epochs"])
    total_number_steps = num_epochs * num_steps
    print_every = num_epochs // 10
    learning_rate = train_params["learning_rate"]
    schedule = optax.linear_schedule(learning_rate, 0., total_number_steps//2, total_number_steps//2)
    optimizer = optax.chain(optax.adaptive_grad_clip(train_params["clip_max_norm"]), optax.adam(schedule))
    opt_state = optimizer.init(params)
    
    condition_mask_fn = get_condition_mask_fn(train_params["condition_mask_fn"], **dict(train_params.pop("condition_mask_fn_kwargs", {})))
    
    
    # Training loop
    def loss_fn(params, key, data, node_id):
        key_times, key_loss, key_condition = jax.random.split(key,3)
        times = jax.random.uniform(key_times, (data.shape[0],), minval=T_min, maxval =1.)
        
        key_choice, key_prob, key_bernoulli = jax.random.split(key_condition,3)
        
        # Structured conditioning
        condition_mask = condition_mask_fn(key_condition, data.shape[0], theta_dim, x_dim)
        condition_mask = jax.random.choice(key_choice, jnp.array([[0] * (theta_dim+x_dim)] + [[0] * theta_dim + [1] * x_dim] + [[1]* theta_dim + [0] * x_dim, jax.random.bernoulli(key_prob,0.3, shape=(data.shape[1],)), jax.random.bernoulli(key_bernoulli,0.7, shape=(data.shape[1],))] ), shape = (data.shape[0],), p = jnp.array([0.2,0.2,0.2, 0.2, 0.2]), axis=0)
        
        # Randomly choose to condition on a few nodes
        #condition_mask = jax.random.bernoulli(jax.random.PRNGKey(0), jax.random.beta(jax.random.PRNGKey(1),1., 4., shape=(data.shape[0],1)), shape=(data.shape[0],data.shape[1])) # More likely to condition on a few nodes
        loss = denoising_score_matching_loss(params, key_loss, times, data, loss_mask=condition_mask, model_fn = model_fn, mean_fn = sde.marginal_mean, std_fn=sde.marginal_stddev, weight_fn=weight_fn, data_id=node_id, condition_mask=condition_mask)
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
    params = run_train_transformer_model(rng_train, params, opt_state, data, node_id, num_epochs, num_steps, batch_size, update, print_every=print_every)
    
    sde_init_params = {"data": data, **sde_params}
    model_init_params = {"output_dim":theta_dim + x_dim , **model_params}
    model = ACSE(params, model_fn, sde, sde_init_params=sde_init_params, model_init_params=model_init_params)
    
    default_conditon_mask = jnp.array([0] * theta_dim + [1] * x_dim, dtype=jnp.bool_)
    model.set_default_condition_mask(default_conditon_mask)
    model.set_default_node_id(node_id)
   
    return model