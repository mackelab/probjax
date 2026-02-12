import jax
import jax.numpy as jnp


def _inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for i, axis in enumerate(permutation):
        inverse[axis] = i
    return tuple(inverse)


def _prod(shape):
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _transpose_if_needed(x, permutation):
    if permutation == tuple(range(x.ndim)):
        return x
    return jnp.transpose(x, permutation)


def _dot_general_left_shapes(out, rhs, dimension_numbers):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contracting = tuple(lhs_contracting)
    rhs_contracting = tuple(rhs_contracting)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    if len(lhs_contracting) != len(rhs_contracting):
        raise NotImplementedError(
            "dot_general inverse requires matched contracting ranks"
        )
    if len(lhs_batch) != len(rhs_batch):
        raise NotImplementedError("dot_general inverse requires matched batch ranks")

    rhs_free_axes = tuple(
        i for i in range(rhs.ndim) if i not in rhs_contracting and i not in rhs_batch
    )
    batch_shape = tuple(rhs.shape[i] for i in rhs_batch)
    rhs_free_shape = tuple(rhs.shape[i] for i in rhs_free_axes)
    lhs_contract_shape = tuple(rhs.shape[i] for i in rhs_contracting)

    out_shape = tuple(out.shape)
    min_rank = len(batch_shape) + len(rhs_free_shape)
    if len(out_shape) < min_rank:
        raise NotImplementedError(
            "dot_general output rank is incompatible with inversion"
        )
    if tuple(out_shape[: len(batch_shape)]) != batch_shape:
        raise NotImplementedError("dot_general output batch shape is incompatible")
    if rhs_free_shape and tuple(out_shape[-len(rhs_free_shape) :]) != rhs_free_shape:
        raise NotImplementedError("dot_general output free shape is incompatible")

    if rhs_free_shape:
        lhs_free_shape = out_shape[len(batch_shape) : -len(rhs_free_shape)]
    else:
        lhs_free_shape = out_shape[len(batch_shape) :]

    return {
        "lhs_contracting": lhs_contracting,
        "rhs_contracting": rhs_contracting,
        "lhs_batch": lhs_batch,
        "rhs_batch": rhs_batch,
        "rhs_free_axes": rhs_free_axes,
        "batch_shape": batch_shape,
        "lhs_free_shape": tuple(lhs_free_shape),
        "lhs_contract_shape": lhs_contract_shape,
        "rhs_free_shape": rhs_free_shape,
    }


def _dot_general_right_shapes(out, lhs, dimension_numbers):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contracting = tuple(lhs_contracting)
    rhs_contracting = tuple(rhs_contracting)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    if len(lhs_contracting) != len(rhs_contracting):
        raise NotImplementedError(
            "dot_general inverse requires matched contracting ranks"
        )
    if len(lhs_batch) != len(rhs_batch):
        raise NotImplementedError("dot_general inverse requires matched batch ranks")

    lhs_free_axes = tuple(
        i for i in range(lhs.ndim) if i not in lhs_contracting and i not in lhs_batch
    )
    batch_shape = tuple(lhs.shape[i] for i in lhs_batch)
    lhs_free_shape = tuple(lhs.shape[i] for i in lhs_free_axes)
    rhs_contract_shape = tuple(lhs.shape[i] for i in lhs_contracting)

    out_shape = tuple(out.shape)
    prefix_shape = batch_shape + lhs_free_shape
    if len(out_shape) < len(prefix_shape):
        raise NotImplementedError(
            "dot_general output rank is incompatible with inversion"
        )
    if tuple(out_shape[: len(prefix_shape)]) != prefix_shape:
        raise NotImplementedError("dot_general output shape is incompatible")

    rhs_free_shape = tuple(out_shape[len(prefix_shape) :])

    return {
        "lhs_contracting": lhs_contracting,
        "rhs_contracting": rhs_contracting,
        "lhs_batch": lhs_batch,
        "rhs_batch": rhs_batch,
        "lhs_free_axes": lhs_free_axes,
        "batch_shape": batch_shape,
        "lhs_free_shape": lhs_free_shape,
        "rhs_contract_shape": rhs_contract_shape,
        "rhs_free_shape": rhs_free_shape,
    }


def dot_general_left_inverse_and_logdet(out, rhs, **params):
    shapes = _dot_general_left_shapes(out, rhs, params["dimension_numbers"])
    batch_shape = shapes["batch_shape"]
    lhs_free_shape = shapes["lhs_free_shape"]
    lhs_contract_shape = shapes["lhs_contract_shape"]
    rhs_free_shape = shapes["rhs_free_shape"]

    batch_size = _prod(batch_shape)
    lhs_free_size = _prod(lhs_free_shape)
    contract_size = _prod(lhs_contract_shape)
    rhs_free_size = _prod(rhs_free_shape)

    if contract_size != rhs_free_size:
        raise NotImplementedError(
            "dot_general inverse for lhs requires square contracted/free rhs dimensions"
        )

    rhs_permutation = (
        shapes["rhs_batch"] + shapes["rhs_contracting"] + shapes["rhs_free_axes"]
    )
    rhs_canon = _transpose_if_needed(rhs, rhs_permutation).reshape(
        batch_size,
        contract_size,
        rhs_free_size,
    )
    out_canon = out.reshape(batch_size, lhs_free_size, rhs_free_size)

    lhs_canon = jax.vmap(
        lambda rhs_matrix, out_matrix: jnp.linalg.solve(rhs_matrix.T, out_matrix.T).T
    )(rhs_canon, out_canon)

    _, rhs_logabsdet = jnp.linalg.slogdet(rhs_canon)
    log_abs_det = -jnp.asarray(lhs_free_size, dtype=rhs_logabsdet.dtype) * rhs_logabsdet
    log_abs_det = jnp.sum(log_abs_det)

    lhs_rank = (
        len(shapes["lhs_batch"]) + len(lhs_free_shape) + len(shapes["lhs_contracting"])
    )
    lhs_free_axes = tuple(
        i
        for i in range(lhs_rank)
        if i not in shapes["lhs_batch"] and i not in shapes["lhs_contracting"]
    )
    lhs_permutation = shapes["lhs_batch"] + lhs_free_axes + shapes["lhs_contracting"]
    lhs_canon_shape = batch_shape + lhs_free_shape + lhs_contract_shape
    lhs = lhs_canon.reshape(lhs_canon_shape)
    lhs = _transpose_if_needed(lhs, _inverse_permutation(lhs_permutation))

    return lhs, log_abs_det


def dot_general_right_inverse_and_logdet(out, lhs, **params):
    shapes = _dot_general_right_shapes(out, lhs, params["dimension_numbers"])
    batch_shape = shapes["batch_shape"]
    lhs_free_shape = shapes["lhs_free_shape"]
    rhs_contract_shape = shapes["rhs_contract_shape"]
    rhs_free_shape = shapes["rhs_free_shape"]

    batch_size = _prod(batch_shape)
    lhs_free_size = _prod(lhs_free_shape)
    contract_size = _prod(rhs_contract_shape)
    rhs_free_size = _prod(rhs_free_shape)

    if lhs_free_size != contract_size:
        raise NotImplementedError(
            "dot_general inverse for rhs requires square free/contracted lhs dimensions"
        )

    lhs_permutation = (
        shapes["lhs_batch"] + shapes["lhs_free_axes"] + shapes["lhs_contracting"]
    )
    lhs_canon = _transpose_if_needed(lhs, lhs_permutation).reshape(
        batch_size,
        lhs_free_size,
        contract_size,
    )
    out_canon = out.reshape(batch_size, lhs_free_size, rhs_free_size)

    rhs_canon = jax.vmap(
        lambda lhs_matrix, out_matrix: jnp.linalg.solve(lhs_matrix, out_matrix)
    )(lhs_canon, out_canon)

    _, lhs_logabsdet = jnp.linalg.slogdet(lhs_canon)
    log_abs_det = -jnp.asarray(rhs_free_size, dtype=lhs_logabsdet.dtype) * lhs_logabsdet
    log_abs_det = jnp.sum(log_abs_det)

    rhs_rank = (
        len(shapes["rhs_batch"]) + len(rhs_free_shape) + len(shapes["rhs_contracting"])
    )
    rhs_free_axes = tuple(
        i
        for i in range(rhs_rank)
        if i not in shapes["rhs_batch"] and i not in shapes["rhs_contracting"]
    )
    rhs_permutation = shapes["rhs_batch"] + shapes["rhs_contracting"] + rhs_free_axes
    rhs_canon_shape = batch_shape + rhs_contract_shape + rhs_free_shape
    rhs = rhs_canon.reshape(rhs_canon_shape)
    rhs = _transpose_if_needed(rhs, _inverse_permutation(rhs_permutation))

    return rhs, log_abs_det


def dot_general_left_inverse(out, rhs, **params):
    lhs, _ = dot_general_left_inverse_and_logdet(out, rhs, **params)
    return lhs


def dot_general_right_inverse(out, lhs, **params):
    rhs, _ = dot_general_right_inverse_and_logdet(out, lhs, **params)
    return rhs


BIVARIATE_INVERSE_REGISTRY = {
    jax.lax.mul_p: (
        jax.lax.div_p,
        jax.lax.div_p,
    ),
    jax.lax.div_p: (
        jax.lax.mul_p,
        lambda x, y, **params: jax.lax.div_p.bind(y, x, **params),
    ),
    jax.lax.add_p: (
        jax.lax.sub_p,
        jax.lax.sub_p,
    ),
    jax.lax.sub_p: (
        jax.lax.add_p.bind,
        lambda x, y, **params: jax.lax.sub_p.bind(y, x, **params),
    ),
    jax.lax.pow_p: (
        lambda x, y, **params: jax.lax.pow_p.bind(x, 1.0 / y, **params),
        lambda x, y, **params: jax.lax.log_p.bind(x) / jax.lax.log_p.bind(y),  # type: ignore
    ),
    jax.lax.dot_general_p: (
        dot_general_left_inverse,
        dot_general_right_inverse,
    ),
}
