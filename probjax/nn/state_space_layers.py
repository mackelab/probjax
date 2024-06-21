import jax
import jax.numpy as jnp


from typing import List, Tuple

import jax
import jax.numpy as jnp
import haiku as hk


# rmsnorm

class RMSNorm(hk.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = dim ** 0.5

    def __call__(self, x):
        gamma = hk.get_parameter("gamma", shape=(x.shape[-1],), init=jnp.ones)
        mean_squared = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        inv_norm = jax.lax.rsqrt(mean_squared + self.eps)
        return self.scale * gamma * x * inv_norm

# gate loop layer

def gate_loop_operator(k, v, q, a):
    kv = k * v + 0.j

    def binary_operator(e_i, e_j):
        a_i, kv_i = e_i
        a_j, kv_j = e_j
        return a_j * a_i, a_j * kv_i + kv_j

    _, y = jax.lax.associative_scan(binary_operator, (a, kv), axis=1)
    return q * jnp.real(y)

class GateLoop(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        norm = RMSNorm(self.dim)
        x = norm(x)

        w_init = hk.initializers.VarianceScaling(scale=2.0)
        
        wq = hk.get_parameter("wq", (self.dim, self.dim), init=w_init)
        wk = hk.get_parameter("wk", (self.dim, self.dim), init=w_init)
        wv = hk.get_parameter("wv", (self.dim, self.dim), init=w_init)
        wa = hk.get_parameter("wa", (self.dim, self.dim * 2), init=w_init)
        wg = hk.get_parameter("wg", (self.dim, self.dim), init=w_init)
        wo = hk.get_parameter("wo", (self.dim, self.dim), init=w_init)

        q = jnp.dot(x, wq)
        k = jnp.dot(x, wk)
        v = jnp.dot(x, wv)
        a = jnp.dot(x, wa)
        g = jnp.dot(x, wg)

        a_real, a_imag = jnp.split(a, 2, axis=-1)
        a_complex = jax.lax.complex(a_real, a_imag)
        magnitude, phase = jnp.abs(a_complex), jnp.angle(a_complex)
        magnitude = jax.nn.sigmoid(magnitude)
        a_complex = magnitude * jnp.exp(1j * phase)

        y = gate_loop_operator(k, v, q, a_complex)
        y = y * jax.nn.silu(g)
        o = jnp.dot(y, wo)
        return o

# basic feedforward with pre-rmsnorm

class GateLoopLayer(hk.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.dim = dim
        self.mult = mult

    def __call__(self, x):
        norm = RMSNorm(self.dim)
        x = norm(x)
        proj_in = hk.Linear(self.dim * self.mult)
        proj_out = hk.Linear(self.dim)
        x = proj_in(x)
        x = jax.nn.gelu(x)
        x = proj_out(x)
        return x


