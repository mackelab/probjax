import argparse
import contextlib
import functools
import logging
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

TEST_MODE = "--test" in sys.argv
IMAGENET_DEFAULT_DIR = Path("/mnt/lustre/datasets/imagenet-1k")


def _get_test_cpu_devices(argv: list[str], default: int = 8) -> int:
    if "--test-cpu-devices" in argv:
        idx = argv.index("--test-cpu-devices")
        if idx + 1 < len(argv):
            try:
                return max(1, int(argv[idx + 1]))
            except ValueError:
                return default
    return default


if TEST_MODE:
    cpu_devices = _get_test_cpu_devices(sys.argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    force_flag = f"--xla_force_host_platform_device_count={cpu_devices}"
    if force_flag not in xla_flags:
        os.environ["XLA_FLAGS"] = (xla_flags + " " + force_flag).strip()

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.ndimage as ndi
from datasets import load_dataset
try:
    from datasets import IterableDataset as HFIterableDataset
except Exception:  # pragma: no cover - optional import for older datasets versions
    HFIterableDataset = None
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

from probjax.nn import GaussianFourierEmbedding, GatedFuse, LearnablePosEncode, Transformer
from probjax.nn.io_util import DataLoader, chunkify, unchunkify
from probjax.nn.layers.attention import flex_attention
from probjax.nn.generative.diffusion import EDM

# NOTE: this script predates the probjax.nn.generative refactor. The import
# above was updated to the new EDM location, but `model.sample_ode(...)`
# below is old-API and no longer exists on EDM/DiffusionDenoiser (see
# `.as_dist(event_spec, mode="ode").sample(...)` in examples/nn/generative/
# for the current sampling API). This is a large ImageNet-scale training
# script kept here for reference/reuse, not verified end-to-end.


def setup_logging(log_path: Path) -> None:
    """Configure logging to both stdout and a file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def _resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive.")
    img = np.asarray(img)
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}.")

    side = min(h, w)
    if h != w:
        top = (h - side) // 2
        left = (w - side) // 2
        img = img[top : top + side, left : left + side, ...]

    if side != size:
        zoom = size / side
        if img.ndim == 2:
            img = ndi.zoom(
                img,
                (zoom, zoom),
                order=1,
                mode="reflect",
                prefilter=False,
            )
        else:
            img = ndi.zoom(
                img,
                (zoom, zoom, 1),
                order=1,
                mode="reflect",
                prefilter=False,
            )
        img = img[:size, :size, ...]
    return img


def _fix_num_channels(arr: np.ndarray, num_channels: int | None) -> np.ndarray:
    if num_channels is None:
        return arr
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}.")
    current = arr.shape[-1]
    if current == num_channels:
        return arr
    if current > num_channels:
        return arr[..., :num_channels]
    pad = num_channels - current
    pad_values = [arr[..., -1:]] * pad
    return np.concatenate([arr] + pad_values, axis=-1)


def _ensure_image_batch(
    xs_batch, target_size: int | None, num_channels: int | None
) -> np.ndarray:
    if isinstance(xs_batch, np.ndarray) and xs_batch.dtype != object:
        if xs_batch.ndim == 4:
            if num_channels is not None:
                current = xs_batch.shape[-1]
                if current > num_channels:
                    xs_batch = xs_batch[..., :num_channels]
                elif current < num_channels:
                    pad = num_channels - current
                    pad_values = np.repeat(xs_batch[..., -1:], pad, axis=-1)
                    xs_batch = np.concatenate([xs_batch, pad_values], axis=-1)
            if target_size is None:
                return xs_batch
            h = xs_batch.shape[1]
            w = xs_batch.shape[2]
            if h == target_size and w == target_size:
                return xs_batch
            images = [
                _fix_num_channels(_resize_to_square(img, target_size), num_channels)
                for img in xs_batch
            ]
            return np.stack(images, axis=0)
        if xs_batch.ndim == 3:
            if target_size is None:
                return xs_batch
            h = xs_batch.shape[1]
            w = xs_batch.shape[2]
            if h == target_size and w == target_size:
                return xs_batch
            images = [_resize_to_square(img, target_size) for img in xs_batch]
            return np.stack(images, axis=0)
        return _fix_num_channels(xs_batch, num_channels)

    if isinstance(xs_batch, np.ndarray) and xs_batch.dtype == object:
        xs_list = list(xs_batch)
    elif isinstance(xs_batch, list):
        xs_list = xs_batch
    else:
        return xs_batch

    images = []
    for img in xs_list:
        arr = np.asarray(img)
        if target_size is not None:
            arr = _resize_to_square(arr, target_size)
        arr = _fix_num_channels(arr, num_channels)
        images.append(arr)
    return np.stack(images, axis=0)


def _is_iterable_dataset(dataset) -> bool:
    if HFIterableDataset is not None and isinstance(dataset, HFIterableDataset):
        return True
    return hasattr(dataset, "__iter__") and not hasattr(dataset, "__getitem__")


def _iterable_image_batches(
    dataset,
    *,
    image_key: str,
    batch_size: int,
    loop: bool = True,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    while True:
        buf = []
        for example in dataset:
            if isinstance(example, dict):
                if image_key not in example:
                    raise KeyError(f"Missing image key '{image_key}' in dataset example.")
                img = example[image_key]
            else:
                img = example
            buf.append(np.asarray(img))
            if len(buf) == batch_size:
                yield buf
                buf = []
        if not loop:
            break


class StreamingIndexableDataset:
    """Wrap an IterableDataset to provide indexed access for DataLoader."""

    def __init__(self, dataset, *, image_key: str, length: int) -> None:
        if length <= 0:
            raise ValueError("length must be positive.")
        self._dataset = dataset
        self._image_key = image_key
        self._length = int(length)
        self._lock = threading.Lock()
        self._iter = iter(dataset)

    def __len__(self) -> int:
        return self._length

    def _next_example(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._dataset)
            return next(self._iter)

    def __getitem__(self, idxs):
        if isinstance(idxs, (int, np.integer)):
            count = 1
        else:
            count = len(idxs)
        images = []
        with self._lock:
            for _ in range(count):
                example = self._next_example()
                if isinstance(example, dict):
                    if self._image_key not in example:
                        raise KeyError(
                            f"Missing image key '{self._image_key}' in dataset example."
                        )
                    img = example[self._image_key]
                else:
                    img = example
                images.append(img)
        return {self._image_key: images}


def img_aug(
    batch: Dict[str, np.ndarray] | np.ndarray,
    *,
    image_key: str = "image",
    target_size: int | None = None,
    augment: bool = True,
    num_channels: int | None = None,
) -> np.ndarray:
    """Apply simple geometric augmentations and map pixels to [-1, 1]."""
    xs_batch = batch[image_key] if isinstance(batch, dict) else batch
    xs_batch = _ensure_image_batch(xs_batch, target_size, num_channels)

    if augment:
        if xs_batch.ndim == 4:
            # CIFAR-style: random zoom-in crop without padding + horizontal flip.
            if np.random.rand() < 0.5:
                n, h, w, c = xs_batch.shape
                scale = np.random.uniform(0.8, 1.0)
                crop_h = max(1, int(round(h * scale)))
                crop_w = max(1, int(round(w * scale)))
                if crop_h != h or crop_w != w:
                    top = np.random.randint(0, h - crop_h + 1)
                    left = np.random.randint(0, w - crop_w + 1)
                    cropped = xs_batch[:, top : top + crop_h, left : left + crop_w, :]
                    zoom_h = (h + 1) / crop_h
                    zoom_w = (w + 1) / crop_w
                    resized = np.empty((n, h, w, c), dtype=cropped.dtype)
                    for i in range(n):
                        zoomed = ndi.zoom(
                            cropped[i],
                            (zoom_h, zoom_w, 1),
                            order=1,
                            mode="reflect",
                            prefilter=False,
                        )
                        resized[i] = zoomed[:h, :w, :]
                    xs_batch = resized
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-4.0, 4.0)
                rotated = np.empty_like(xs_batch)
                for i in range(xs_batch.shape[0]):
                    rotated[i] = ndi.rotate(
                        xs_batch[i],
                        angle=angle,
                        axes=(0, 1),
                        reshape=False,
                        order=1,
                        mode="reflect",
                        prefilter=False,
                    )
                xs_batch = rotated
            if np.random.rand() < 0.5:
                xs_batch = xs_batch[:, :, ::-1, :]
        else:
            # MNIST-style: mild rotations and shifts.
            if xs_batch.ndim == 4 and xs_batch.shape[-1] == 1:
                xs_batch = xs_batch[..., 0]

            angle = np.random.uniform(-4.0, 4.0)
            xs_batch = ndi.rotate(
                xs_batch,
                angle=angle,
                axes=(1, 2),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
            )

            ty = np.random.uniform(-2.0, 2.0)
            tx = np.random.uniform(-2.0, 2.0)
            shift = (0.0, ty, tx)
            xs_batch = ndi.shift(
                xs_batch,
                shift=shift,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )

    xs_batch = xs_batch.astype(np.float32)
    if xs_batch.max() > 1.0:
        xs_batch /= 255.0

    xs_batch += np.random.uniform(0.0, 1 / 255.0, xs_batch.shape).astype(np.float32)
    xs_batch = 2.0 * xs_batch - 1.0
    xs_batch = np.clip(xs_batch, -1.0, 1.0)
    if xs_batch.ndim == 3:
        xs_batch = xs_batch[..., None]
    return xs_batch.astype(np.float32)


def compute_normalization_stats(
    train_dataset,
    *,
    image_key: str,
    repetitions: int = 5,
    max_samples: int | None = None,
    seed: int = 0,
    target_size: int | None = None,
    augment: bool = True,
    num_channels: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate dataset statistics in the inverse-tanh space."""
    logging.info("Estimating normalization statistics using %d passes.", repetitions)
    if max_samples is not None and max_samples <= 0:
        max_samples = None
    is_iterable = _is_iterable_dataset(train_dataset)
    if is_iterable and max_samples is None:
        max_samples = 50_000
        logging.info(
            "Iterable dataset detected; estimating stats from first %d samples.",
            max_samples,
        )

    dataset = train_dataset
    if not is_iterable and max_samples is not None and len(train_dataset) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_dataset), size=max_samples, replace=False)
        dataset = train_dataset.select(idx)
        logging.info(
            "Using %d/%d samples to estimate normalization statistics.",
            len(dataset),
            len(train_dataset),
        )
    mu_acc: Optional[np.ndarray] = None
    std_acc: Optional[np.ndarray] = None

    for idx in range(repetitions):
        if is_iterable:
            total_sum: Optional[np.ndarray] = None
            total_sumsq: Optional[np.ndarray] = None
            total_count = 0
            for batch in _iterable_image_batches(
                dataset, image_key=image_key, batch_size=1024, loop=False
            ):
                if max_samples is not None:
                    remaining = max_samples - total_count
                    if remaining <= 0:
                        break
                    if len(batch) > remaining:
                        batch = batch[:remaining]
                augmented = img_aug(
                    batch,
                    image_key=image_key,
                    target_size=target_size,
                    augment=augment,
                    num_channels=num_channels,
                )
                unconstrained = np.arctanh(
                    np.clip(augmented * 0.9999, -0.9999, 0.9999)
                )
                batch_sum = np.sum(unconstrained, axis=0)
                batch_sumsq = np.sum(unconstrained**2, axis=0)
                if total_sum is None:
                    total_sum = batch_sum
                    total_sumsq = batch_sumsq
                else:
                    total_sum += batch_sum
                    total_sumsq += batch_sumsq
                total_count += unconstrained.shape[0]
            if total_sum is None or total_sumsq is None or total_count == 0:
                raise ValueError("Failed to compute normalization statistics; empty dataset.")
            batch_mu = total_sum / total_count
            batch_var = total_sumsq / total_count - batch_mu**2
            batch_std = np.sqrt(np.clip(batch_var, 1e-6, None))
        else:
            augmented = img_aug(
                dataset[:],
                image_key=image_key,
                target_size=target_size,
                augment=augment,
                num_channels=num_channels,
            )
            unconstrained = np.arctanh(np.clip(augmented * 0.9999, -0.9999, 0.9999))
            batch_mu = np.mean(unconstrained, axis=0)
            batch_std = np.std(unconstrained, axis=0)

        if mu_acc is None:
            mu_acc = batch_mu
            std_acc = batch_std
        else:
            mu_acc += batch_mu
            std_acc += batch_std

        logging.info("Completed statistics pass %d/%d.", idx + 1, repetitions)

    assert mu_acc is not None and std_acc is not None
    mu = mu_acc / repetitions
    std = np.clip(std_acc / repetitions, 1e-4, None)
    return mu.astype(np.float32), std.astype(np.float32)


def make_normalize_fn(mu: np.ndarray, std: np.ndarray):
    """Create a normalization function that runs on host data."""
    def normalize(xs: np.ndarray) -> np.ndarray:
        xs_unconstrained = np.arctanh(np.clip(xs * 0.9999, -0.9999, 0.9999))
        return ((xs_unconstrained - mu) / std).astype(np.float32)

    return normalize


def denormalize_to_pixels(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_unconstrained = x * std + mu
    x_pixels = np.tanh(x_unconstrained)
    x_pixels = (x_pixels + 1.0) * 0.5
    return np.clip(x_pixels, 0.0, 1.0)


def to_seq(
    x: jnp.ndarray, *, image_size: int, num_channels: int, chunk_size: int
) -> jnp.ndarray:
    resize = lambda arr: jax.image.resize(
        arr, shape=(image_size, image_size, num_channels), method="bilinear"
    )
    tokens = jax.vmap(resize)(x)
    return jax.vmap(lambda arr: chunkify(arr, chunk_shape=(chunk_size, chunk_size)))(tokens)


def to_img(
    tokens: jnp.ndarray, *, image_size: int, num_channels: int, chunk_size: int
) -> jnp.ndarray:
    unchunk = lambda arr: unchunkify(
        arr,
        chunk_shape=(chunk_size, chunk_size),
        spatial_shape=(image_size, image_size),
    )
    imgs = jax.vmap(unchunk)(tokens)
    return jax.vmap(
        lambda arr: jax.image.resize(
            arr, shape=(image_size, image_size, num_channels), method="bilinear"
        )
    )(imgs)


class Denoiser(nnx.Module):
    """Transformer-based denoiser backbone used by EDM."""

    def __init__(
        self,
        rngs,
        *,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        attn_size: int,
        dropout_rate: float,
        image_size: int,
        num_channels: int,
        chunk_size: int,
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.image_size = image_size
        self.num_channels = num_channels
        self.chunk_size = chunk_size
        token_dim = chunk_size * chunk_size * num_channels
        self.time_embedding = GaussianFourierEmbedding(1, model_dim, rngs=rngs, learnable=True)
        self.pos_encode = LearnablePosEncode(
            model_dim, max_seq_len=(28 * 28 // 2) + 1, rngs=rngs
        )
        self.enc = nnx.Linear(token_dim, model_dim, rngs=rngs)
        self.dec = nnx.Linear(
            model_dim, token_dim, rngs=rngs, kernel_init=nnx.initializers.zeros
        )
        self.net = Transformer(
            model_dim,
            num_heads,
            num_layers,
            attn_size,
            context_dim=model_dim,
            attention_fn=flex_attention,
            normalize_qk_attn=True,
            rngs=rngs,
            dtype=jnp.bfloat16,
            precision="BF16_BF16_F32",
            preferred_element_type=jnp.float32,
            attn_fuse_cls=GatedFuse,
            dropout_rate=dropout_rate,
            sharding=sharding,
        )

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        time_embed = self.time_embedding(t)
        tokens = to_seq(
            x,
            image_size=self.image_size,
            num_channels=self.num_channels,
            chunk_size=self.chunk_size,
        )
        tokens = self.enc(tokens)
        tokens = self.pos_encode(tokens)
        tokens = self.net(tokens, context=time_embed)
        out = self.dec(tokens)
        return to_img(
            out,
            image_size=self.image_size,
            num_channels=self.num_channels,
            chunk_size=self.chunk_size,
        )


def build_model(
    rngs,
    *,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    attn_size: int,
    dropout_rate: float,
    image_size: int,
    num_channels: int,
    chunk_size: int,
    loss_type: str,
    sharding: jax.sharding.Mesh | None = None,
):
    net = Denoiser(
        rngs,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        attn_size=attn_size,
        dropout_rate=dropout_rate,
        image_size=image_size,
        num_channels=num_channels,
        chunk_size=chunk_size,
        sharding=sharding,
    )
    return EDM(net, loss_type=loss_type)


def create_model(
    seed: int,
    *,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    attn_size: int,
    dropout_rate: float,
    image_size: int,
    num_channels: int,
    chunk_size: int,
    loss_type: str,
    sharding: jax.sharding.Mesh | None = None,
) -> Tuple[nnx.GraphDef, nnx.Param, nnx.State]:
    rngs = nnx.Rngs(jax.random.PRNGKey(seed))
    model = build_model(
        rngs,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        attn_size=attn_size,
        dropout_rate=dropout_rate,
        image_size=image_size,
        num_channels=num_channels,
        chunk_size=chunk_size,
        loss_type=loss_type,
        sharding=sharding,
    )
    return nnx.split(model, nnx.Param, ...)


def create_graphdef(
    *,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    attn_size: int,
    dropout_rate: float,
    image_size: int,
    num_channels: int,
    chunk_size: int,
    loss_type: str,
    sharding: jax.sharding.Mesh | None = None,
) -> nnx.GraphDef:
    model = build_model(
        nnx.Rngs(jax.random.PRNGKey(0)),
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        attn_size=attn_size,
        dropout_rate=dropout_rate,
        image_size=image_size,
        num_channels=num_channels,
        chunk_size=chunk_size,
        loss_type=loss_type,
        sharding=sharding,
    )
    graphdef, _, _ = nnx.split(model, nnx.Param, ...)
    return graphdef


def init_model_params_state(
    rng: jax.Array,
    *,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    attn_size: int,
    dropout_rate: float,
    image_size: int,
    num_channels: int,
    chunk_size: int,
    loss_type: str,
    sharding: jax.sharding.Mesh | None = None,
):
    model = build_model(
        nnx.Rngs(rng),
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        attn_size=attn_size,
        dropout_rate=dropout_rate,
        image_size=image_size,
        num_channels=num_channels,
        chunk_size=chunk_size,
        loss_type=loss_type,
        sharding=sharding,
    )
    _, params, state = nnx.split(model, nnx.Param, ...)
    return params, state


def count_parameters(params: nnx.Param) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(jnp.shape(leaf)) for leaf in leaves))


def make_loss_fn(graphdef):
    def loss_fn(params, state, x, rng):
        model = nnx.merge(graphdef, params, state, copy=True)
        model.train()
        losses = model.loss(
            rng,
            x,
            axis=(-3, -2, -1),
            adaptive_weight_p=0.8,
            adaptive_weight_eps=1e-3,
        )
        _, _, new_state = nnx.split(model, nnx.Param, ...)
        return losses, new_state

    return loss_fn


def _is_array_like(x) -> bool:
    # Works across JAX versions (jax.Array introduced later).
    jax_array_t = getattr(jax, "Array", ())
    return isinstance(x, (np.ndarray, jnp.ndarray, jax_array_t)) or (
        hasattr(x, "shape") and hasattr(x, "dtype")
    )


def reshard_tree(tree, spec_tree, mesh: jax.sharding.Mesh | None):
    """Place a pytree of host/replicated arrays onto a mesh using per-leaf PartitionSpecs."""
    if mesh is None or spec_tree is None:
        # Keep behavior identical to before when not sharding.
        return jax.device_put(tree)

    def _put(x, spec):
        # Leave non-arrays untouched (e.g. optax.EmptyState, Python scalars, None).
        if not _is_array_like(x):
            return x
        # If spec is missing, default to replicated.
        if spec is None:
            spec = PartitionSpec()
        return jax.device_put(x, NamedSharding(mesh, spec))

    return jax.tree_util.tree_map(_put, tree, spec_tree)


def make_train_step(
    graphdef,
    opt,
    ema,
    scheduler,
):
    loss_fn = make_loss_fn(graphdef)

    def train_step(params, state, opt_state, ema_state, x, rng, step):
        (losses, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, x, rng
        )
        # Global mean over sharded batch; collectives are inserted automatically.
        loss = jnp.mean(losses)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params, ema_state = ema.update(params, ema_state)
        lr = scheduler(step)
        metrics = {"loss": loss, "learning_rate": lr}
        return params, new_state, opt_state, ema_state, ema_params, metrics

    return jax.jit(train_step, donate_argnums=(0, 1, 2, 3))


def make_sample_fn(graphdef):
    @jax.jit
    def sample_fn(params, state, eps):
        model = nnx.merge(graphdef, params, state, copy=True)
        model.eval()
        return model.sample_ode(eps)

    return sample_fn


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    params,
    state,
    opt_state,
    ema_state,
    ema_params,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{step:08d}.pkl"
    payload = {
        "step": step,
        "params": jax.device_get(params),
        "state": jax.device_get(state),
        "opt_state": jax.device_get(opt_state),
        "ema_state": jax.device_get(ema_state),
        "ema_params": jax.device_get(ema_params),
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Saved checkpoint to %s.", path)


def restore_latest_checkpoint(
    checkpoint_dir: Path,
    *,
    mesh: jax.sharding.Mesh | None,
    param_spec,
    state_spec,
    opt_spec,
    ema_state_spec,
):
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("step_*.pkl"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    with latest.open("rb") as fh:
        payload = pickle.load(fh)
    logging.info("Loaded checkpoint from %s.", latest)
    # Start from host arrays (from jax.device_get at save time), then place with correct shardings.
    host_params = payload["params"]
    host_state = payload["state"]
    host_opt_state = payload["opt_state"]
    host_ema_state = payload["ema_state"]
    host_ema_params = payload.get("ema_params", host_params)

    params = reshard_tree(host_params, param_spec, mesh)
    state = reshard_tree(host_state, state_spec, mesh)
    opt_state = reshard_tree(host_opt_state, opt_spec, mesh)
    ema_state = reshard_tree(host_ema_state, ema_state_spec, mesh)
    ema_params = reshard_tree(host_ema_params, param_spec, mesh)

    return {
        "step": int(payload.get("step", 0)),
        "params": params,
        "state": state,
        "opt_state": opt_state,
        "ema_state": ema_state,
        "ema_params": ema_params,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EDM denoiser on images.")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--log-path", type=Path, default=Path("mnist_edm.log"))
    parser.add_argument(
        "--test",
        action="store_true",
        help="Disable wandb and run on CPU with multiple host devices.",
    )
    parser.add_argument(
        "--test-cpu-devices",
        type=int,
        default=8,
        help="Number of CPU devices to emulate in --test mode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stat-repeats", type=int, default=5)
    parser.add_argument(
        "--stat-max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples used for normalization stats.",
    )
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Local data directory for imagefolder/imagenet64 datasets.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--attn-size", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--loss-type", type=str, default="v")
    parser.add_argument(
        "--mesh-shape",
        type=str,
        default="",
        help="Mesh shape for sharding, e.g. '2,4' or '2x4'.",
    )
    parser.add_argument(
        "--mesh-axes",
        type=str,
        default="data,model",
        help="Comma-separated mesh axis names matching mesh-shape.",
    )
    parser.add_argument(
        "--data-batch-spec",
        type=str,
        default="",
        help="PartitionSpec for DataLoader batches, e.g. 'data' or 'data,None,None,None'.",
    )
    parser.add_argument("--wandb-project", type=str, default="mnist-edm")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-log-interval", type=int, default=500)
    parser.add_argument("--sample-interval", type=int, default=5000)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    return parser.parse_args()


def parse_mesh_shape(value: str) -> Tuple[int, ...]:
    if not value:
        return ()
    parts = value.replace("x", ",").split(",")
    return tuple(int(part.strip()) for part in parts if part.strip())


def parse_axis_names(value: str) -> Tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_partition_spec(value: str) -> PartitionSpec | None:
    if not value:
        return None
    if value.lower() in {"none", "null"}:
        return None
    parts = [part.strip() for part in value.split(",")]
    axes = [None if part.lower() in {"none", "null", "_"} else part for part in parts]
    return PartitionSpec(*axes)


def _select_train_split(dataset):
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        if "train" in dataset:
            return dataset["train"]
        split = next(iter(dataset.keys()))
        logging.warning("Dataset has no 'train' split; using '%s'.", split)
        return dataset[split]
    return dataset


def load_hf_dataset(args) -> object:
    dataset_name = args.dataset.lower()
    if dataset_name in {"imagenet64", "imagenet-64"}:
        data_dir = args.data_dir or IMAGENET_DEFAULT_DIR
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"ImageNet directory not found: {data_dir}")
        logging.info("Loading ImageNet from %s using imagefolder.", data_dir)
        return load_dataset("webdataset", data_dir=str(data_dir), streaming=True)
    if dataset_name == "imagefolder":
        if args.data_dir is None:
            raise ValueError("--data-dir is required when using --dataset imagefolder.")
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Image dataset directory not found: {data_dir}")
        logging.info("Loading imagefolder dataset from %s.", data_dir)
        return load_dataset("imagefolder", data_dir=str(data_dir))
    return load_dataset(args.dataset, keep_in_memory=True)


def mesh_context(mesh: jax.sharding.Mesh | None):
    if mesh is None:
        return contextlib.nullcontext()
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return mesh


def get_mesh_axis_size(mesh: jax.sharding.Mesh | None, axis_name: str) -> int | None:
    if mesh is None:
        return None
    try:
        shape = mesh.shape
    except Exception:
        shape = None
    if isinstance(shape, dict):
        size = shape.get(axis_name, None)
        if size is not None:
            return int(size)
    try:
        axis_idx = mesh.axis_names.index(axis_name)
    except Exception:
        return None
    try:
        return int(mesh.devices.shape[axis_idx])
    except Exception:
        return None


def collapse_singleton_model_axis(
    mesh: jax.sharding.Mesh | None, batch_spec: PartitionSpec | None
) -> jax.sharding.Mesh | None:
    if mesh is None or "model" not in getattr(mesh, "axis_names", ()):
        return mesh
    model_size = get_mesh_axis_size(mesh, "model")
    if model_size != 1:
        return mesh
    if batch_spec is not None and "model" in batch_spec:
        raise ValueError("batch_spec references 'model' axis but model axis size is 1.")
    flat_devices = np.array(mesh.devices).reshape(-1)
    return jax.sharding.Mesh(flat_devices, ("data",))


def select_model_mesh(mesh: jax.sharding.Mesh | None) -> jax.sharding.Mesh | None:
    if mesh is None or "model" not in getattr(mesh, "axis_names", ()):
        return None
    model_size = get_mesh_axis_size(mesh, "model")
    if model_size is None or model_size <= 1:
        return None
    return mesh


def make_spec_from_sharding(tree):
    def leaf_spec(x):
        sharding = getattr(x, "sharding", None)
        spec = getattr(sharding, "spec", None)
        if spec is not None:
            return spec
        return PartitionSpec()

    return jax.tree_util.tree_map(leaf_spec, tree)


def build_mesh(mesh_shape: Tuple[int, ...], axis_names: Tuple[str, ...]):
    if not mesh_shape:
        return None
    if len(mesh_shape) != len(axis_names):
        raise ValueError("mesh-shape and mesh-axes must have the same length.")
    if hasattr(jax, "make_mesh"):
        return jax.make_mesh(mesh_shape, axis_names)
    try:
        from jax.experimental import mesh_utils

        devices = mesh_utils.create_device_mesh(mesh_shape)
    except Exception:
        devices = np.array(jax.devices()).reshape(mesh_shape)
    return jax.sharding.Mesh(devices, axis_names)


def setup_wandb(args):
    if args.test:
        logging.info("Test mode enabled: wandb logging disabled.")
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required for logging. Please install wandb.") from exc
    wandb_kwargs = {"project": args.wandb_project}
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name:
        wandb_kwargs["name"] = args.wandb_run_name
    wandb.init(**wandb_kwargs, config=vars(args))
    return wandb


def check_batch_sharding(batch, batch_spec: PartitionSpec | None, mesh):
    if batch_spec is None:
        return
    leaves = jax.tree_util.tree_leaves(batch)
    specs = [getattr(getattr(x, "sharding", None), "spec", None) for x in leaves]
    meshes = [getattr(getattr(x, "sharding", None), "mesh", None) for x in leaves]
    if not all(spec == batch_spec for spec in specs):
        raise ValueError(f"Batch specs mismatch: {specs} vs expected {batch_spec}")
    if not all(mesh_obj is mesh for mesh_obj in meshes):
        raise ValueError("Batch mesh object does not match expected mesh.")


def maybe_log_step(
    *,
    step_idx: int | None,
    metrics,
    step_time: float | None,
    total_steps: int,
    epoch: int,
    args: argparse.Namespace,
    wandb,
):
    if step_idx is None or metrics is None or step_time is None:
        return
    log_step = step_idx + 1
    if args.log_interval > 0 and log_step % args.log_interval == 0:
        host_metrics = jax.device_get(metrics)
        loss_val = float(host_metrics["loss"])
        lr_val = float(host_metrics["learning_rate"])
        logging.info(
            "Epoch %d Step %d/%d - Loss: %.4f",
            epoch + 1,
            log_step,
            total_steps,
            loss_val,
        )
        if wandb is not None:
            wandb.log(
                {
                    "loss": loss_val,
                    "learning_rate": lr_val,
                    "step_time_s": step_time,
                    "throughput_sps": args.batch_size / max(step_time, 1e-12),
                },
                step=log_step,
            )
    if (
        wandb is not None
        and args.wandb_log_interval > 0
        and log_step % args.wandb_log_interval == 0
    ):
        wandb.log({"step_time_s": step_time}, step=log_step)


def make_image_grid(images: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    rows, cols = grid_size
    images = images[: rows * cols]
    h, w, c = images.shape[1:]
    grid = np.zeros((rows * h, cols * w, c), dtype=images.dtype)
    idx = 0
    for r in range(rows):
        for col in range(cols):
            if idx >= images.shape[0]:
                break
            grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = images[idx]
            idx += 1
    return grid


def main() -> None:
    args = parse_args()
    setup_logging(args.log_path)

    logging.info("Starting training with configuration: %s", args)
    if args.test:
        try:
            jax.config.update("jax_platform_name", "cpu")
        except Exception:
            pass
    dataset = load_hf_dataset(args)
    train_ds = _select_train_split(dataset)
    train_columns = train_ds.column_names
    if "image" in train_columns:
        image_key = "image"
    elif "img" in train_columns:
        image_key = "img"
    else:
        image_key = train_columns[0]
        logging.warning("Using '%s' as image column.", image_key)
    is_iterable = _is_iterable_dataset(train_ds)
    if is_iterable:
        try:
            train_ds = train_ds.with_format("numpy", columns=[image_key])
        except TypeError:
            train_ds = train_ds.with_format("numpy")
    else:
        train_ds.set_format(type="numpy", columns=[image_key])

    dataset_name = args.dataset.lower()
    if dataset_name == "mnist":
        default_image_size = 28
        default_num_channels = 1
        default_chunk_size = 3
    elif dataset_name == "cifar10":
        default_image_size = 32
        default_num_channels = 3
        default_chunk_size = 4
    elif dataset_name in {"imagenet64", "imagenet-64"}:
        default_image_size = 64
        default_num_channels = 3
        default_chunk_size = 8
    elif dataset_name == "imagefolder":
        if args.image_size is None or args.num_channels is None or args.chunk_size is None:
            raise ValueError(
                "imagefolder requires --image-size, --num-channels, and --chunk-size "
                "(or use --dataset imagenet64)."
            )
        default_image_size = args.image_size
        default_num_channels = args.num_channels
        default_chunk_size = args.chunk_size
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    image_size = args.image_size or default_image_size
    num_channels = args.num_channels or default_num_channels
    chunk_size = args.chunk_size or default_chunk_size

    resize_target = (
        image_size
        if dataset_name in {"imagenet64", "imagenet-64", "imagefolder"}
        else None
    )
    augment = dataset_name not in {"imagenet64", "imagenet-64"}
    stat_max_samples = args.stat_max_samples
    if is_iterable:
        if stat_max_samples is None:
            stat_max_samples = 50_000
    else:
        if stat_max_samples is None and len(train_ds) > 100000:
            stat_max_samples = 50_000
    mu, std = compute_normalization_stats(
        train_ds,
        image_key=image_key,
        repetitions=args.stat_repeats,
        max_samples=stat_max_samples,
        seed=args.seed,
        target_size=resize_target,
        augment=augment,
        num_channels=num_channels,
    )
    normalize_fn = make_normalize_fn(mu, std)

    mesh_shape = parse_mesh_shape(args.mesh_shape)
    mesh_axes = parse_axis_names(args.mesh_axes)
    if args.test and not mesh_shape:
        mesh_shape = (jax.device_count(), 1)
        mesh_axes = ("data", "model")
    mesh = build_mesh(mesh_shape, mesh_axes)
    batch_spec = parse_partition_spec(args.data_batch_spec)
    if batch_spec is not None and mesh is None:
        raise ValueError("--data-batch-spec requires --mesh-shape/--mesh-axes.")
    # Collapse size-1 model axis to avoid 2D SPMD compilation overhead.
    mesh = collapse_singleton_model_axis(mesh, batch_spec)

    host_transforms = [
        functools.partial(
            img_aug,
            image_key=image_key,
            target_size=resize_target,
            augment=augment,
            num_channels=num_channels,
        ),
        normalize_fn,
    ]
    if is_iterable:
        stream_len = max(1, args.steps_per_epoch * args.batch_size)
        stream_ds = StreamingIndexableDataset(
            train_ds, image_key=image_key, length=stream_len
        )
        loader = DataLoader(
            stream_ds,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=False,
            num_prefetch_host=8,
            num_prefetch_device=2,
            shard=False,
            host_transforms=host_transforms,
            num_async_workers=1,
            mesh=mesh,
            batch_spec=batch_spec,
        )
        data_iter = iter(loader)
    else:
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            drop_last=True,
            num_prefetch_host=8,
            num_prefetch_device=2,
            shard=False,
            host_transforms=host_transforms,
            num_async_workers=1,
            mesh=mesh,
            batch_spec=batch_spec,
        )
        data_iter = iter(loader)

    wandb = setup_wandb(args)
    with mesh_context(mesh):
        model_mesh = select_model_mesh(mesh)
        graphdef = create_graphdef(
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            attn_size=args.attn_size,
            dropout_rate=args.dropout_rate,
            image_size=image_size,
            num_channels=num_channels,
            chunk_size=chunk_size,
            loss_type=args.loss_type,
            sharding=model_mesh,
        )
        init_fn = jax.jit(
            functools.partial(
                init_model_params_state,
                model_dim=args.model_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                attn_size=args.attn_size,
                dropout_rate=args.dropout_rate,
                image_size=image_size,
                num_channels=num_channels,
                chunk_size=chunk_size,
                loss_type=args.loss_type,
                sharding=model_mesh,
            )
        )
        params, state = init_fn(jax.random.PRNGKey(args.seed))
        total_steps = max(args.num_epochs * args.steps_per_epoch, 1)
        scheduler = optax.cosine_onecycle_schedule(total_steps, args.learning_rate)
        opt = optax.chain(optax.adaptive_grad_clip(5.0), optax.radam(scheduler))
        ema = optax.ema(0.999)

        opt_state = opt.init(params)
        ema_state = ema.init(params)
        ema_params = params
        param_spec = state_spec = opt_spec = ema_state_spec = None
        if mesh is not None:
            param_spec = make_spec_from_sharding(params)
            state_spec = make_spec_from_sharding(state)
            opt_spec = make_spec_from_sharding(opt_state)
            ema_state_spec = make_spec_from_sharding(ema_state)
        train_step_fn = make_train_step(
            graphdef,
            opt,
            ema,
            scheduler,
        )
        sample_fn = make_sample_fn(graphdef)

    logging.info("Model parameter count: %d", count_parameters(params))

    checkpoint_dir = Path(args.checkpoint_dir)
    restored = restore_latest_checkpoint(
        checkpoint_dir,
        mesh=mesh,
        param_spec=param_spec,
        state_spec=state_spec,
        opt_spec=opt_spec,
        ema_state_spec=ema_state_spec,
    )
    global_step = 0
    if restored is not None:
        params = restored["params"]
        state = restored["state"]
        opt_state = restored["opt_state"]
        ema_state = restored["ema_state"]
        ema_params = restored["ema_params"]
        global_step = restored["step"]

    rng = jax.random.PRNGKey(args.seed + 1)
    steps_per_epoch = max(args.steps_per_epoch, 1)
    start_epoch = global_step // steps_per_epoch
    step_offset = global_step % steps_per_epoch
    if global_step > 0:
        logging.info(
            "Resuming from global step %d (epoch %d, intra-epoch step %d).",
            global_step,
            start_epoch,
            step_offset,
        )
    else:
        logging.info("No checkpoint found. Starting from scratch.")

    with mesh_context(mesh):
        prev_metrics = None
        prev_step = None
        prev_step_time = None
        prev_epoch = None
        last_epoch = None
        for epoch in range(start_epoch, args.num_epochs):
            last_epoch = epoch
            epoch_step_start = step_offset if epoch == start_epoch else 0
            for _ in range(epoch_step_start, args.steps_per_epoch):
                step_start = time.perf_counter()
                batch = next(data_iter)
                check_batch_sharding(batch, batch_spec, mesh)

                rng, step_rng = jax.random.split(rng)
                step_idx = global_step
                step_arr = jnp.asarray(step_idx, dtype=jnp.int32)
                params, state, opt_state, ema_state, ema_params, metrics = train_step_fn(
                    params, state, opt_state, ema_state, batch, step_rng, step_arr
                )
                step_time = time.perf_counter() - step_start

                maybe_log_step(
                    step_idx=prev_step,
                    metrics=prev_metrics,
                    step_time=prev_step_time,
                    total_steps=total_steps,
                    epoch=prev_epoch if prev_epoch is not None else epoch,
                    args=args,
                    wandb=wandb,
                )
                prev_metrics = metrics
                prev_step = step_idx
                prev_step_time = step_time
                prev_epoch = epoch
                global_step += 1

                if (
                    wandb is not None
                    and args.sample_interval > 0
                    and global_step % args.sample_interval == 0
                ):
                    sample_rng, rng = jax.random.split(rng)
                    sample_eps = jax.random.normal(
                        sample_rng,
                        (args.sample_batch_size, image_size, image_size, num_channels),
                        dtype=jnp.float32,
                    ) * 80.0
                    samples = sample_fn(ema_params, state, sample_eps)
                    samples = np.array(jax.device_get(samples))
                    samples = denormalize_to_pixels(samples, mu, std)
                    grid_side = int(np.ceil(np.sqrt(samples.shape[0])))
                    grid = make_image_grid(samples, (grid_side, grid_side))
                    wandb.log({"samples": wandb.Image(grid)}, step=global_step)

                if (
                    args.checkpoint_interval > 0
                    and global_step % args.checkpoint_interval == 0
                ):
                    save_checkpoint(
                        checkpoint_dir,
                        global_step,
                        params,
                        state,
                        opt_state,
                        ema_state,
                        ema_params,
                    )

            logging.info(
                "Completed epoch %d/%d at global step %d.",
                epoch + 1,
                args.num_epochs,
                global_step,
            )
            step_offset = 0
        if last_epoch is not None:
            maybe_log_step(
                step_idx=prev_step,
                metrics=prev_metrics,
                step_time=prev_step_time,
                total_steps=total_steps,
                epoch=prev_epoch if prev_epoch is not None else last_epoch,
                args=args,
                wandb=wandb,
            )

        save_checkpoint(
            checkpoint_dir,
            global_step,
            params,
            state,
            opt_state,
            ema_state,
            ema_params,
        )
        logging.info(
            "Training finished. Final model saved to %s at step %d.",
            checkpoint_dir,
            global_step,
        )
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
