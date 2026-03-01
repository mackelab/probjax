#!/usr/bin/env python3
"""Train an autoregressive Transformer on MNIST and report loss/bits-per-dim."""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
from functools import partial
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import jax_utils as flax_jax_utils
from flax import nnx
import scipy.ndimage as ndi
from jax import tree_util

from probjax.nn import LearnablePosEncode, Transformer
from probjax.nn.io_util import DataLoader
from probjax.nn.layers.attention import flex_attention
from probjax.nn.pallas_kernels import CausalMask

NUM_PIXELS = 28 * 28
NUM_CLASSES = 256


def setup_logger(log_file: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("transformers_mnist")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        path = os.path.abspath(log_file)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _align_loaded_tree(
    loaded_tree,
    reference_tree,
    *,
    num_devices: int,
):
    """Match a loaded PyTree to reference shapes, removing replica axes if needed."""

    def _maybe_trim(loaded, reference):
        if isinstance(loaded, (np.ndarray, jax.Array)):
            loaded_arr = jnp.asarray(loaded)
        elif np.isscalar(loaded):
            loaded_arr = jnp.asarray(loaded)
        else:
            return loaded

        ref_arr = jnp.asarray(reference)

        if loaded_arr.shape == ref_arr.shape:
            return loaded_arr

        if (
            loaded_arr.ndim == ref_arr.ndim + 1
            and loaded_arr.shape[0] == num_devices
            and loaded_arr.shape[1:] == ref_arr.shape
        ):
            return loaded_arr[0]

        return loaded_arr

    return tree_util.tree_map(
        _maybe_trim,
        loaded_tree,
        reference_tree,
        is_leaf=lambda x: isinstance(x, optax.EmptyState),
    )


def _tree_has_replica_axis(tree, num_devices: int) -> bool:
    for leaf in tree_util.tree_leaves(tree):
        if isinstance(leaf, (np.ndarray, jax.Array)):
            arr = np.asarray(leaf)
            if arr.ndim > 0 and arr.shape[0] == num_devices:
                return True
    return False


def img_aug(batch: dict[str, np.ndarray]) -> np.ndarray:
    """Lightweight augmentation that matches the original notebook."""
    images = batch["image"]

    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]

    images = images.astype(np.float32) / 255.0

    angle = np.random.uniform(-4.0, 4.0)
    images = ndi.rotate(
        images,
        angle=angle,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )

    ty = np.random.uniform(-2.0, 2.0)
    tx = np.random.uniform(-2.0, 2.0)
    images = ndi.shift(
        images,
        shift=(0.0, ty, tx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    images = np.clip(images, 0.0, 1.0)
    images = np.rint(images * 255.0).astype(np.int32)
    return images.reshape(-1, NUM_PIXELS)


class Model(nnx.Module):
    num_heads: int = 8
    num_layers: int = 12
    dim: int
    widening_factor: int = 4
    attn_size: int = 32
    vocab_size: int = NUM_CLASSES

    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        dropout_rate: Optional[float] = 0.1,
    ):
        self.dropout_rate = dropout_rate
        self.embed = nnx.Embed(self.vocab_size, dim, rngs=rngs)
        self.pos_embed = LearnablePosEncode(dim, max_seq_len=NUM_PIXELS + 1, rngs=rngs)
        self.transformer = Transformer(
            dim,
            self.num_heads,
            self.num_heads,
            self.attn_size,
            attention_fn=flex_attention,
            widening_factor=self.widening_factor,
            dropout_rate=0.0,  # dropout_rate if dropout_rate else 0.0,
            rngs=rngs,
            dtype=jnp.bfloat16,
            precision="BF16_BF16_F32",
            preferred_element_type=jnp.float32,
        )
        # Add layer norm before output projection for better stability
        self.output_norm = nnx.LayerNorm(dim, rngs=rngs)
        # Use small random init instead of zeros for faster convergence
        self.output = nnx.Linear(
            dim,
            self.vocab_size,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.02),
        )
        # Initialize start token with small random values
        self.start_token = nnx.Param(jax.random.normal(rngs(), (1, dim)) * 0.02)
        if dropout_rate and dropout_rate > 0:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(
        self,
        tokens: jax.Array,
        decode: bool = False,
        deterministic: bool = False,
    ):
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        tokens = self.embed(tokens)
        tokens = self.pos_embed(tokens)
        # Apply dropout to embeddings during training
        if self.dropout_rate and self.dropout_rate > 0 and not deterministic:
            tokens = self.dropout(tokens)
        start_token = jnp.repeat(
            self.start_token.value[None, ...], tokens.shape[0], axis=0
        )
        tokens = jnp.concatenate([start_token, tokens], axis=1)
        mask = CausalMask()
        tokens = self.transformer(
            tokens, mask=mask, deterministic=deterministic, decode=decode
        )
        # Apply output normalization
        tokens = self.output_norm(tokens)
        logits = self.output(tokens)
        return logits[..., :-1, :]

    def predict_logits(
        self,
        tokens: jax.Array,
        decode: bool = False,
        deterministic: bool = False,
    ):
        return self(tokens, decode=decode, deterministic=deterministic)


def prepare_dataloader(
    dataset,
    *,
    batch_size: int,
    shard: bool = True,
    num_prefetch_host: int = 8,
    num_prefetch_device: int = 2,
    num_async_workers: int = 1,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shard=shard,
        host_transforms=[img_aug],
        num_prefetch_host=num_prefetch_host,
        num_prefetch_device=num_prefetch_device,
        num_async_workers=num_async_workers,
    )


def build_train_step(
    graphdef: nnx.GraphDef,
    optimizer: optax.GradientTransformation,
    label_smoothing: float = 0.1,
):
    def loss_fn(params, state, tokens, _rng):
        model = nnx.merge(graphdef, params, state, copy=True)
        model.train()
        logits = model(tokens, deterministic=False)
        targets = jnp.asarray(tokens, dtype=jnp.int32)
        # Apply label smoothing for better generalization
        token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        if label_smoothing > 0:
            # Compute label smoothing loss
            num_classes = logits.shape[-1]
            smooth_labels = jax.nn.one_hot(targets, num_classes)
            smooth_labels = (
                smooth_labels * (1 - label_smoothing) + label_smoothing / num_classes
            )
            smooth_loss = optax.softmax_cross_entropy(logits, smooth_labels)
            token_loss = smooth_loss
        token_loss = jnp.sum(token_loss, axis=-1)
        loss = jnp.mean(token_loss)
        _, _, new_state = nnx.split(model, nnx.Param, ...)
        return loss, new_state

    @partial(jax.pmap, axis_name="data")
    def update_fn(params, state, opt_state, tokens, rng):
        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, tokens, rng
        )
        grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grads)

        loss = jax.lax.pmean(loss, axis_name="data")
        grads = jax.lax.pmean(grads, axis_name="data")

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), params)
        return params, new_state, opt_state, loss

    def train_step(params, state, opt_state, tokens, rng):
        rng = jax.random.split(rng, jax.local_device_count())
        params, state, opt_state, losses = update_fn(
            params, state, opt_state, tokens, rng
        )
        return params, state, opt_state, jnp.mean(losses)

    return train_step


def compute_bits_per_dim(
    graphdef: nnx.GraphDef,
    params,
    state,
    tokens: np.ndarray,
    batch_size: int = 512,
) -> float:
    model = nnx.merge(graphdef, params, state)
    total_nll = 0.0
    total_count = 0
    for start in range(0, tokens.shape[0], batch_size):
        batch = jnp.asarray(tokens[start : start + batch_size], dtype=jnp.int32)
        logits = model.predict_logits(batch, deterministic=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
        nll_per_sample = jnp.sum(loss, axis=-1)
        total_nll += float(jnp.sum(nll_per_sample))
        total_count += batch.shape[0]
    mean_nll = total_nll / total_count
    return mean_nll / (NUM_PIXELS * math.log(2.0))


def save_checkpoint(
    path: str,
    params_per_device,
    state_per_device,
    opt_state,
    step: int,
    *,
    process_index: int = 0,
) -> bool:
    """Serialize training state to disk. Returns True iff the file was written."""
    if process_index != 0:
        return False

    params = flax_jax_utils.unreplicate(params_per_device)
    state = flax_jax_utils.unreplicate(state_per_device)
    opt_state_single = flax_jax_utils.unreplicate(opt_state)
    host_params = jax.device_get(params)
    host_state = jax.device_get(state)
    host_opt_state = jax.device_get(opt_state_single)

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {
                "params": host_params,
                "state": host_state,
                "opt_state": host_opt_state,
                "step": step,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp_path, path)
    return True


def load_checkpoint(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_training(
    *,
    logger: logging.Logger,
    num_steps: int,
    log_interval: int,
    eval_interval: Optional[int],
    batch_size: int,
    learning_rate: float,
    checkpoint_path: Optional[str],
    checkpoint_interval: int,
    eval_batch_size: int,
    dropout_rate: float = 0.1,
    label_smoothing: float = 0.1,
) -> None:
    if batch_size % jax.local_device_count() != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by local_device_count={jax.local_device_count()}"
        )

    local_devices = jax.local_device_count()
    per_device_bsz = batch_size // local_devices

    dataset = load_dataset("mnist", keep_in_memory=True)
    dataset.set_format(type="numpy", columns=["image"])
    logger.info(
        "Loaded MNIST dataset with %s train / %s test samples",
        len(dataset["train"]),
        len(dataset["test"]),
    )

    train_loader = prepare_dataloader(dataset["train"], batch_size=batch_size)
    data_iter: Iterator = iter(train_loader)

    test_images = np.array(dataset["test"]["image"], dtype=np.int32).reshape(
        -1, NUM_PIXELS
    )

    model = Model(dim=512, rngs=nnx.Rngs(0), dropout_rate=dropout_rate)
    graphdef, params, state = nnx.split(model, nnx.Param, ...)

    # Improved optimizer with warmup and cosine decay
    warmup_steps = 2000
    total_steps = num_steps
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # More effective for transformers
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    opt_state_single = optimizer.init(params)

    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None
    start_step = 1

    process_index = jax.process_index()
    if checkpoint:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        try:
            # Load step counter
            start_step = checkpoint.get("step", 0) + 1

            # Load and align parameters
            loaded_params = checkpoint["params"]
            if _tree_has_replica_axis(loaded_params, local_devices):
                logger.info("Unreplicating checkpoint parameters")
                loaded_params = flax_jax_utils.unreplicate(loaded_params)

            # Verify shapes match
            def check_shapes(loaded, ref, path="root"):
                if isinstance(loaded, dict) and isinstance(ref, dict):
                    for key in ref:
                        if key not in loaded:
                            raise ValueError(f"Missing key in checkpoint: {path}.{key}")
                        check_shapes(loaded[key], ref[key], f"{path}.{key}")
                elif isinstance(loaded, (np.ndarray, jax.Array)) and isinstance(
                    ref, (np.ndarray, jax.Array)
                ):
                    loaded_shape = np.array(loaded).shape
                    ref_shape = np.array(ref).shape
                    if loaded_shape != ref_shape:
                        raise ValueError(
                            f"Shape mismatch at {path}: "
                            f"checkpoint {loaded_shape} vs model {ref_shape}"
                        )

            check_shapes(loaded_params, params)
            params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), loaded_params)

            # Load state if it exists
            if "state" in checkpoint:
                loaded_state = checkpoint["state"]
                if _tree_has_replica_axis(loaded_state, local_devices):
                    logger.info("Unreplicating checkpoint state")
                    loaded_state = flax_jax_utils.unreplicate(loaded_state)
                check_shapes(loaded_state, state)
                state = jax.tree_util.tree_map(lambda x: jnp.asarray(x), loaded_state)

            # Load and align optimizer state
            loaded_opt_state = checkpoint["opt_state"]
            if _tree_has_replica_axis(loaded_opt_state, local_devices):
                logger.info("Unreplicating checkpoint optimizer state")
                loaded_opt_state = flax_jax_utils.unreplicate(loaded_opt_state)

            opt_state_single = jax.tree_util.tree_map(
                lambda loaded, ref: (
                    jnp.asarray(loaded)
                    if not isinstance(ref, optax.EmptyState)
                    else ref
                ),
                loaded_opt_state,
                opt_state_single,
                is_leaf=lambda x: isinstance(x, optax.EmptyState),
            )

            logger.info(
                "Successfully loaded checkpoint from %s (resuming from step %d)",
                checkpoint_path,
                start_step,
            )

        except Exception as err:
            logger.error(
                "Failed to load checkpoint: %s. Starting from scratch.",
                err,
                exc_info=True,
            )
            start_step = 1
            # Reset to original initialized state (already done above)
            # params and state are already initialized
    else:
        logger.info("No checkpoint found, starting from scratch")

    params_per_device = flax_jax_utils.replicate(params)
    state_per_device = flax_jax_utils.replicate(state)
    opt_state = flax_jax_utils.replicate(opt_state_single)

    logger.info("Building training step function...")
    train_step = build_train_step(graphdef, optimizer, label_smoothing=label_smoothing)
    logger.info("Training step function built successfully")

    rng = jax.random.key(0)
    loss_acc = 0.0
    logger.info(
        "Starting training for %d steps (batch=%d, per-device=%d, lr=%.2e, "
        "log_interval=%d, eval_interval=%s)",
        num_steps,
        batch_size,
        per_device_bsz,
        learning_rate,
        log_interval,
        eval_interval if eval_interval else "disabled",
    )

    logger.info(
        "Starting training loop (first step will trigger JIT compilation, "
        "may take 1-2 minutes)..."
    )
    try:
        for step in range(start_step, num_steps + 1):
            rng, step_key = jax.random.split(rng)
            batch_tokens = next(data_iter)
            if step == start_step:
                logger.info(
                    "Compiling training step with JIT (this may take a while)..."
                )
            params_per_device, state_per_device, opt_state, loss_value = train_step(
                params_per_device, state_per_device, opt_state, batch_tokens, step_key
            )
            if step == start_step:
                logger.info("JIT compilation complete! Training in progress...")
            loss_scalar = float(loss_value)
            loss_acc += loss_scalar

            if step % log_interval == 0:
                avg_loss = loss_acc / log_interval
                logger.info("[step %d] loss=%.4f", step, avg_loss)
                loss_acc = 0.0
            elif step == start_step:
                logger.info("[step %d] loss=%.4f", step, loss_scalar)

            if eval_interval and step % eval_interval == 0:
                params_host = flax_jax_utils.unreplicate(params_per_device)
                state_host = flax_jax_utils.unreplicate(state_per_device)
                bpd = compute_bits_per_dim(
                    graphdef,
                    params_host,
                    state_host,
                    test_images,
                    batch_size=eval_batch_size,
                )
                logger.info("[step %d] bits_per_dim=%.4f", step, bpd)

            if (
                checkpoint_path
                and checkpoint_interval
                and step % checkpoint_interval == 0
            ):
                if save_checkpoint(
                    checkpoint_path,
                    params_per_device,
                    state_per_device,
                    opt_state,
                    step,
                    process_index=process_index,
                ):
                    logger.info(
                        "Saved checkpoint to %s at step %d",
                        checkpoint_path,
                        step,
                    )

        params = flax_jax_utils.unreplicate(params_per_device)
        state = flax_jax_utils.unreplicate(state_per_device)
        bpd = compute_bits_per_dim(
            graphdef, params, state, test_images, batch_size=eval_batch_size
        )
        if loss_acc and num_steps % log_interval != 0:
            remaining = num_steps % log_interval
            avg_loss = loss_acc / remaining
            logger.info("[step %d] loss=%.4f", num_steps, avg_loss)
        logger.info("Final bits_per_dim=%.4f", bpd)
    except Exception as e:
        logger.error("An error occurred during training: %s", e, exc_info=True)
    finally:
        train_loader.close()
        logger.info("Closed data loader.")

    if checkpoint_path:
        if save_checkpoint(
            checkpoint_path,
            params_per_device,
            state_per_device,
            opt_state,
            num_steps,
            process_index=process_index,
        ):
            logger.info("Saved checkpoint to %s", checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-steps", type=int, default=400_000, help="Training steps."
    )
    parser.add_argument(
        "--log-interval", type=int, default=500, help="Log frequency in steps."
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10_000,
        help="Evaluation frequency in steps (set 0 to disable).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128 * 2,
        help="Global batch size (must divide number of local devices).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Peak learning rate (after warmup).",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.05,
        help="Dropout rate for regularization.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.001,
        help="Label smoothing factor.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoint.pkl"),
        help="Where to save/load training state.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10_000,
        help="Save a checkpoint every N steps (set 0 to disable periodic saves).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="transformer_mnist.log",
        help="Optional path to append logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file)
    eval_interval = args.eval_interval or None
    run_training(
        logger=logger,
        num_steps=args.num_steps,
        log_interval=args.log_interval,
        eval_interval=eval_interval,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        eval_batch_size=args.eval_batch_size,
        dropout_rate=args.dropout_rate,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()
