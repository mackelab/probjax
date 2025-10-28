import numpy as np
import matplotlib.pyplot as plt


def local_permutation_indices(
    height=28, width=28, window=3, orientation='horizontal', seed=None
):
    """
    Generate a permutation of flattened indices such that each pixel is only permuted
    within a local window of size `window`x`window` in the 2D image.

    Args:
        height: Height of the image (default 28)
        width: Width of the image (default 28)
        window: Size of the local window (default 3)
        seed: Optional random seed

    Returns:
        perm: A permutation of np.arange(height*width) with local shuffling
    """
    rng = np.random.default_rng(seed)
    perm = np.arange(height * width).reshape(height, width)
    out = np.empty_like(perm)
    for i in range(0, height, window):
        for j in range(0, width, window):
            h_end = min(i + window, height)
            w_end = min(j + window, width)
            block = perm[i:h_end, j:w_end].flatten()
            rng.shuffle(block)
            out[i:h_end, j:w_end] = block.reshape(h_end - i, w_end - j)

    if orientation == 'horizontal':
        out = out.T
    else:
        out = out
    return out.flatten()


# Example usage:
local_perm = local_permutation_indices(height=28, width=28, window=7, seed=44)


plt.imshow(local_perm.reshape(28, 28), cmap='gray')

# %%

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.nets.autoregressive import AutoregressiveTransformer, AutoregressiveMLP
from probjax.nn.nets.transformer import Transformer
from probjax.nn.attention import flex_attention
from probjax.nn import Permute, Sequential
from functools import partial

from probjax.core import inverse_and_logabsdet
from probjax.stats import transformed, norm, independent


def affine_bijector(params, x):
    scale, shift = jnp.split(params, 2, axis=-1)
    scale = jnp.tanh(scale) + 1.0
    return shift + scale * x


ar_transformer1 = AutoregressiveTransformer(
    in_out_dim=1,
    bijector_dim=2,
    bijector=affine_bijector,
    rngs=nnx.Rngs(1),
    model_dim=128,
    num_layers=3,
    normalize_qk_attn=True,
)

mixing = Permute(
    local_permutation_indices(
        height=28, width=28, window=7, orientation='horizontal', seed=44
    ),
    axis=-2,
)
ar_transformer2 = AutoregressiveTransformer(
    in_out_dim=1,
    bijector_dim=2,
    bijector=affine_bijector,
    rngs=nnx.Rngs(2),
    model_dim=128,
    num_layers=3,
    normalize_qk_attn=True,
)
mixing2 = Permute(
    local_permutation_indices(
        height=28, width=28, window=7, orientation='vertical', seed=45
    ),
    axis=-2,
)
ar_transformer3 = AutoregressiveTransformer(
    in_out_dim=1,
    bijector_dim=2,
    bijector=affine_bijector,
    rngs=nnx.Rngs(3),
    model_dim=128,
    num_layers=3,
    normalize_qk_attn=True,
)
mixing3 = Permute(
    local_permutation_indices(
        height=28, width=28, window=3, orientation='horizontal', seed=46
    ),
    axis=-2,
)
ar_transformer4 = AutoregressiveTransformer(
    in_out_dim=1,
    bijector_dim=2,
    bijector=affine_bijector,
    rngs=nnx.Rngs(4),
    model_dim=128,
    num_layers=3,
    normalize_qk_attn=True,
)
mixing4 = Permute(
    local_permutation_indices(
        height=28, width=28, window=3, orientation='vertical', seed=47
    ),
    axis=-2,
)
ar_transformer5 = AutoregressiveTransformer(
    in_out_dim=1,
    bijector_dim=2,
    bijector=affine_bijector,
    rngs=nnx.Rngs(5),
    model_dim=128,
    num_layers=3,
    normalize_qk_attn=True,
)

flow = Sequential(
    ar_transformer1,
    mixing,
    ar_transformer2,
    mixing2,
    ar_transformer3,
    mixing3,
    ar_transformer4,
    mixing4,
    ar_transformer5,
)


# %%
# MNIST data
from datasets import load_dataset
import numpy as np

dataset = load_dataset('mnist', keep_in_memory=True)

dataset.set_format(type='numpy', columns=['image'])


# %%
train_data = dataset["train"]

import threading
import queue


def prefetch_generator(generator, prefetch=2):
    q = queue.Queue(prefetch)
    sentinel = object()

    def producer():
        for item in generator:
            q.put(item)
        q.put(sentinel)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


def train_data_generator(batch_size=256, prefetch=8, device=jax.devices()[0]):
    def _gen():
        while True:
            idx = np.random.randint(0, len(train_data), size=batch_size)
            xs_batch = np.array(train_data[idx]["image"], dtype=np.float32).reshape(
                batch_size, 28 * 28
            )
            # Uniform dequantization: add uniform noise in [0, 1) to each pixel, then divide by 255
            xs_batch = (xs_batch + np.random.uniform(0, 1.0, xs_batch.shape)) / 256
            # Move to device as jax array
            xs_batch = jax.device_put(xs_batch[..., None], device)

            yield xs_batch

    return prefetch_generator(_gen(), prefetch=prefetch)


datagenerator = train_data_generator()

# %%
xs_data = next(datagenerator)


loc0 = jnp.zeros((784, 1))
scale0 = jnp.ones((784, 1))
base_dist = independent(independent(norm(loc0, scale0)))

p = transformed(base_dist, flow)

graphdef, params, state = nnx.split(flow, nnx.Param, ...)

# %%
import matplotlib.pyplot as plt

plt.imshow(xs_data[0].reshape(28, 28), cmap='gray', vmin=0, vmax=1)


# %%
def loss(params, xs):
    bijector = nnx.merge(graphdef, params, state)
    return jnp.mean(-transformed.logpdf(xs, base_dist, bijector))


# %%
import optax

optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.radam(1e-4))


@jax.jit
def update(params, opt_state, xs):
    loss_val, grads = jax.value_and_grad(loss)(params, xs)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss_val, new_params, opt_state


# %%
opt_state = optimizer.init(params)

# %%
for i in range(50):
    l = 0
    for _ in range(1000):
        xs_batch = next(datagenerator)
        loss_val, params, opt_state = update(params, opt_state, xs_batch)
        l += loss_val
    print(l / 1000)

# %%
# Save params
import pickle

with open('params.pkl', 'wb') as f:
    pickle.dump(params, f)

# Load params
with open('params.pkl', 'rb') as f:
    params = pickle.load(f)

# %%
bijector = nnx.merge(graphdef, params, state)
p = transformed(base_dist, bijector)


# %%
samples = p.rvs(jax.random.PRNGKey(0), shape=(10,))

# %%
import matplotlib.pyplot as plt

plt.imshow(samples[5].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.show()

# %%


# %%
