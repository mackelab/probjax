import jax
import jax.numpy as jnp
from probjax.stats import genpareto, poisson, truncnorm, transformed, norm
from probjax.nn.utils import call_with_optional_rng, filter_supported_kwargs
from probjax.stats.fit import fit, FitCallback
from scipy import stats as sp

def check(name,fn):
 try: print(name,fn(),flush=True)
 except Exception as e: print(name,type(e).__name__,str(e)[:250],flush=True)
print('JAX',jax.__version__,flush=True)
check('genpareto batch sampling',lambda:genpareto.rvs(jax.random.key(0),shape=(5,),c=jnp.array([0.,.1,.2])).shape)
check('plain function optional rng',lambda:call_with_optional_rng(lambda x:x+1,1.,rng=jax.random.key(0)))
class ForwardingLayer:
 def __init__(self, **kwargs): pass
check('forwarding layer kwargs',lambda:filter_supported_kwargs(ForwardingLayer,kernel_sharding=('data',None)))
for n in [0,1]:
 check('truncnorm moment '+str(n),lambda n=n:truncnorm.moment(n,a=jnp.array([-1.,-2.]),b=jnp.array([1.,2.])))
for c in [-2.,-3.]:
 check('genpareto endpoint vs returned mode '+str(c),lambda c=c:(genpareto.mode(c=c),genpareto.pdf(genpareto.mode(c=c),c=c),genpareto.pdf(-1/c,c=c)))
class Stop(FitCallback):
 def on_fit_begin(self,state): return False
for mode in ['host','io']:
 check('fit stop on begin '+mode,lambda mode=mode:fit(lambda p,k,b:jnp.sum(p*p),jnp.array([1.]),jax.random.key(0),jnp.ones((4,1)),num_steps=3,callbacks=[Stop()],callback_mode=mode,return_result=True).valid_steps)
with jax.enable_x64():
 check('truncnorm second derivative fp64',lambda:jax.grad(jax.grad(lambda q:truncnorm.ppf(q)))(.7))
 check('truncnorm near endpoint small q',lambda:(truncnorm.ppf(1e-100,a=-1.,b=1.),truncnorm.ppf(1e-100,a=8.,b=9.)))
