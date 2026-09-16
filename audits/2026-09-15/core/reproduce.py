import jax
import jax.numpy as jnp
from probjax.core import joint_sample, log_joint_fn, condition
from probjax.stats import norm
from probjax.core.custom_primitives.random_variable import rv_p
print('JAX',jax.__version__)
def check(name,fn):
 try: print(name,fn())
 except Exception as e: print(name,type(e).__name__,str(e)[:200])
def model(key):
 return norm.rvs(key,name='x')
def raw(key):
 return rv_p.bind(key,0.,1.,dist=norm,name='x')
for name,f in [('plain',model),('jitted',jax.jit(model)),('raw_jitted',jax.jit(raw))]:
 check(name+' sample',lambda f=f:joint_sample(f)(jax.random.key(0)))
 check(name+' logjoint x=2 expected -2.918939',lambda f=f:log_joint_fn(f)(x=jnp.array(2.)))
 check(name+' condition x=2 expected 2',lambda f=f:condition(f,{'x':jnp.array(2.)})(jax.random.key(0)))
