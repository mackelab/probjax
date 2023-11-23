from typing import Optional, Type
import jax
import jax.numpy as jnp

from typing import Callable, Union
from jaxtyping import Array, PyTree

import haiku as hk

from probjax.nn.helpers import GaussianFourierEmbedding, SinusoidalEmbedding


def scalarize(data: PyTree) -> PyTree:
    flat, tree = jax.tree_util.tree_flatten(data)
    flat = [jnp.expand_dims(x, -1) for x in flat]
    flat_concat = jnp.concatenate(flat, axis=-2)

    tree_type, tree_data = tree.node_data()
    if tree_type is dict:
        node_num = flat_concat.shape[-2]
        node_names_num = len(tree_data)
        keys = [
            n + "_" + str(i)
            for n in tree_data
            for i in range(node_num // node_names_num)
        ]
        values = jnp.split(flat_concat, node_num, axis=-2)
        return dict(zip(keys, values))
    elif tree_type is list:
        return jnp.split(flat_concat, len(tree_data), axis=-2)
    elif tree_type is tuple:
        return tuple(jnp.split(flat_concat, len(tree_data), axis=-2))
    else:
        raise ValueError(f"Unknown tree type: {tree_type}")


class Tokenizer(hk.Module):
    def __init__(
        self,
        output_dim: int,
        node_embeding_builder: Optional[Callable] = None,
        value_embeding_builder: Optional[Callable] = None,
        node_meta_data_embeding_builder: Optional[Callable] = None,
        distributor: Optional[Union[Callable, str]] = "equal",
        accummulator: Optional[Union[Callable, str]] = "concat",
        name: str | None = "tokenizer",
    ):
        self.output_dim = output_dim
        self.value_embeding_builder = value_embeding_builder
        self.node_embeding_builder = node_embeding_builder
        self.meta_data_embeding_builder = node_meta_data_embeding_builder
        self.distibutor = distributor
        self.accummulator = accummulator
        super().__init__(name)


class ScalarTokenizer(Tokenizer):
    def __init__(
        self,
        output_dim: int,
        max_sequence_length: int,
        node_embeding_builder: Optional[Callable] = None,
        value_embeding_builder: Optional[Callable] = None,
        node_meta_data_embeding_builder: Optional[Callable] = None,
        accummulator: Optional[Union[Callable, str]] = "concat",
        distributor: Optional[Callable] = None,
        name: str | None = "scalar_tokenizer",
    ):
        """This tokenizer will treat each scalar as a seperate variable and will hence create a token for each scalar.
        

        Args:
            output_dim (int): _description_
            max_sequence_length (int): _description_
            node_embeding_builder (Optional[Callable], optional): _description_. Defaults to None.
            value_embeding_builder (Optional[Callable], optional): _description_. Defaults to None.
            node_meta_data_embeding_builder (Optional[Callable], optional): _description_. Defaults to None.
            accummulator (Optional[Union[Callable, str]], optional): _description_. Defaults to "concat".
            distributor (Optional[Callable], optional): _description_. Defaults to None.
            name (str | None, optional): _description_. Defaults to "scalar_tokenizer".
        """
        self.max_sequence_length = max_sequence_length
        super().__init__(
            output_dim,
            node_embeding_builder,
            value_embeding_builder,
            node_meta_data_embeding_builder,
            distributor,
            accummulator,
            name,
        )

    def __call__(self, data_id: Array, data: Array, meta_data: Optional[Array] = None):
        *leading_dims, sequence_length, variable_dim = data.shape

        data = data.reshape(-1, sequence_length, variable_dim)
        data_id = data_id.astype(jnp.int32).reshape(-1, sequence_length, 1)
        data_id, data = jnp.broadcast_arrays(data_id, data)

        if meta_data is not None:
            meta_data.reshape(-1, sequence_length, variable_dim)
            data_id, data, meta_data = jnp.broadcast_arrays(data_id, data, meta_data)

        output_dim1, output_dim2, output_dim3 = self.distribute_output_dim(
            with_meta_data=meta_data is not None
        )

        data_id_embeding = self.node_embeding(data_id, output_dim1)
        data_embeding = self.value_embeding(data, output_dim2)
        
        if meta_data is not None:
            meta_data_embeding = self.meta_data_embeding(meta_data, output_dim3)
        else:
            meta_data_embeding = None

        tokens = self.accumulate(data_id_embeding, data_embeding, meta_data_embeding)

        return tokens.reshape(*leading_dims, sequence_length, self.output_dim)

    @hk.transparent
    def accumulate(self, data_id_embedding, data_embedding, meta_data_embedding):
        if self.accummulator == "concat":
            out = [data_id_embedding, data_embedding]
            if meta_data_embedding is not None:
                out.append(meta_data_embedding)
            return jnp.concatenate(out, axis=-1)
        elif self.accummulator == "sum":
            out = data_id_embedding + data_embedding
            if meta_data_embedding is not None:
                out += meta_data_embedding
            return out

        else:
            raise ValueError(
                f"Unknown accummulator: {self.accummulator}, Please specify a custom distributor function, that returns a tuple of output dimensions for each input type."
            )

    @hk.transparent
    def distribute_output_dim(self, with_meta_data: bool = False):
        if isinstance(self.distibutor, Callable):
            return self.distibutor(self.output_dim)
        else:
            if self.accummulator == "concat":
                if with_meta_data:
                    output_dim1 = self.output_dim // 3
                    output_dim2 = self.output_dim // 3
                    output_dim3 = self.output_dim - (output_dim1 + output_dim2)
                    return output_dim1, output_dim2, output_dim3
                else:
                    output_dim1 = self.output_dim // 2
                    output_dim2 = self.output_dim - output_dim1
                    return output_dim1, output_dim2, 0
            elif self.accummulator == "sum":
                return self.output_dim, self.output_dim, self.output_dim
            else:
                raise ValueError(
                    f"Unknown accummulator: {self.accummulator}, Please specify a custom distributor function, that returns a tuple of output dimensions for each input type."
                )

    @hk.transparent
    def value_embeding(self, value, output_dim):
        if self.value_embeding_builder is None:
            value_embeding_fn = hk.Conv1D(output_dim, 1, w_init=hk.initializers.Constant(1.0))
        else:
            value_embeding_fn = self.value_embeding_builder(output_dim)

        out = value_embeding_fn(value).reshape(-1, value.shape[-2], output_dim)
        out = jax.lax.stop_gradient(out)
        return out

    @hk.transparent
    def node_embeding(self, node, output_dim):
        if self.node_embeding_builder is None:
            node_embeding_fn = hk.Embed(
                self.max_sequence_length,
                output_dim,
                w_init=hk.initializers.Orthogonal(scale=0.5),
            )
        else:
            node_embeding_fn = self.node_embeding_builder(self.output_dim)

        return node_embeding_fn(node).reshape(-1, node.shape[-2], output_dim)

    @hk.transparent
    def meta_data_embeding(self, meta_data, output_dim):
        if self.meta_data_embeding_builder is None:
            meta_data_embeding_fn = GaussianFourierEmbedding(output_dim)
        else:
            meta_data_embeding_fn = self.meta_data_embeding_builder(self.output_dim)

        return meta_data_embeding_fn(meta_data).reshape(
            -1, meta_data.shape[-2], output_dim
        )


class StructuredTokenizer(Tokenizer):
    def __init__(
        self,
        output_dim: int,
        node_embeding_builder: Optional[Callable] = None,
        value_embeding_builder: Optional[Callable] = None,
        node_meta_data_embeding_builder: Optional[Callable] = None,
        accummulator: Optional[Union[Callable, str]] = "concat",
        name: str | None = "tokenizer",
    ):
        super().__init__(
            output_dim,
            node_embeding_builder,
            value_embeding_builder,
            node_meta_data_embeding_builder,
            accummulator,
            name,
        )

    def __call__(self, data, meta_data=None):
        pass

    @hk.transparent
    def value_embeding(self, value):
        if self.value_embeding_builder is None:
            value_embeding_fn = hk.Conv1D(self.output_dim, 1)
        else:
            value_embeding_fn = self.value_embeding_builder(self.output_dim)

        return value_embeding_fn(value)

    @hk.transparent
    def node_embeding(self, node):
        if self.node_embeding_builder is None:
            node_embeding_fn = GaussianFourierEmbedding(self.output_dim)
        else:
            node_embeding_fn = self.node_embeding_builder(self.output_dim)

        return node_embeding_fn(node)
