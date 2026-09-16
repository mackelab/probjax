import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sp
from probjax.utils.linalg import lanczos_logdet, batched_pcg_solve
from probjax.stats import truncnorm, genpareto, poisson, norm, transformed
from probjax.core import joint_sample, log_joint_fn

def check(name, fn):
    try: print(name, fn(), flush=True)
    except Exception as e: print(name,type(e).__name__,str(e)[:350],flush=True)
print('JAX',jax.__version__,flush=True)
for n in [1,2,3,20]:
 check('logdet scaled identity gradient dim'+str(n),lambda n=n:jax.value_and_grad(lambda x:lanczos_logdet(x*jnp.eye(n),num_steps=n))(2.))
for q in [1e-20,.1,.7,1.-1e-8]:
 check('truncnorm second derivative q'+str(q),lambda q=q:(jax.grad(jax.grad(lambda p:truncnorm.ppf(p)))(q),sp.norm.ppf(q)/sp.norm.pdf(sp.norm.ppf(q))**2))
for a,b in [(20.,jnp.inf),(8.,jnp.inf),(1.,1.001)]:
 check('truncnorm tail '+str((a,b)),lambda a=a,b=b:(truncnorm.mean(a=a,b=b),truncnorm.var(a=a,b=b),sp.truncnorm.mean(a,b),sp.truncnorm.var(a,b)))
for c in [-2.,0.,.5]:
 check('genpareto nan input c'+str(c),lambda c=c:genpareto.ppf(jnp.nan,c=c))
 check('genpareto mode c'+str(c),lambda c=c:genpareto.mode(c=c))
 check('genpareto kurtosis c'+str(c),lambda c=c:(genpareto.kurtosis(c=c),sp.genpareto.stats(c,moments='k')))
for scale in [1e-20,1e20]:
 check('pcg extreme scale'+str(scale),lambda scale=scale:batched_pcg_solve(lambda x:x,jnp.ones((2,1))*scale,block_size=1))
@jax.jit
def ignored(key):
 norm.rvs(key,name='ignored')
 return 1.
check('unused nested sample',lambda:joint_sample(ignored)(jax.random.key(0)))
check('unused nested logjoint',lambda:log_joint_fn(ignored)(ignored=jnp.array(2.)))
@jax.jit
def inner(key):
 return norm.rvs(key,name='x')
def nested(key):
 k1,k2=jax.random.split(key)
 x=inner(k1)
 y=norm.rvs(k2,loc=x,name='y')
 return y
check('nested dependent sites',lambda:joint_sample(nested)(jax.random.key(0)))
check('nested dependent logjoint',lambda:(log_joint_fn(nested)(x=jnp.array(2.),y=jnp.array(3.)),sp.norm.logpdf(2.)+sp.norm.logpdf(1.)))
