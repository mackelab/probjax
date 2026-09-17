import jax
import jax.numpy as jnp
from flax import nnx
from probjax.nn import Transformer, DiffusionTransformer
print('JAX',jax.__version__)
seen=[]
class RecordingDropout(nnx.Module):
 def __init__(self,rate,rngs): self.rate=rate
 def __call__(self,x,deterministic=None,rngs=None):
  seen.append(jax.random.key_data(rngs))
  return x
m=Transformer(8,2,3,4,dropout_rate=.2,dropout_rate_attn=0.,dropout_cls=RecordingDropout,rngs=nnx.Rngs(0))
m(jnp.ones((2,4,8)),rng=jax.random.key(8),deterministic=False)
print('Explicit dropout keys across three blocks',seen)
print('All keys equal',all(bool(jnp.array_equal(seen[0],k)) for k in seen[1:]))
# Show exact t/r permutation symmetry, an architectural constraint rather than crash.
n=DiffusionTransformer(3,model_dim=8,num_layers=1,num_heads=2,attn_size=4,time_embed_dim=8,fourier_dim=8,rngs=nnx.Rngs(0))
a=n._time_embedding(.2,(2,))+n._time_embedding(.8,(2,))
b=n._time_embedding(.8,(2,))+n._time_embedding(.2,(2,))
print('Mean-flow pair embeddings swap invariant',bool(jnp.array_equal(a,b)))
import importlib
flash = importlib.import_module('probjax.nn.pallas_kernels.kernels.flash_attention3')
try: flash._flash_fwd_impl(None,None,None,config=None,use_pipeline_emitter=False)
except Exception as e: print('Flash3 forwarding function',type(e).__name__,str(e))
