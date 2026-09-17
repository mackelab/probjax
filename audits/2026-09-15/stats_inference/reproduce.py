import importlib
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sp
from probjax.stats import pareto, genpareto, binomial, poisson, truncnorm, norm, transformed
from probjax.inference.mcmc.imh import gaussian_imh, proposal_gaussian_logpdf
uk = importlib.import_module('probjax.inference.filtering.unscented_kalman_filter')

def check(name, fn):
    try: print(name, fn())
    except Exception as e: print(name, type(e).__name__, str(e)[:250])
print('JAX',jax.__version__)
key=jax.random.key(0)
check('pareto sample min/mean (expected min~1, mean1.5)',lambda: (float(pareto.rvs(key,shape=(20000,),b=1.,alpha=3.).min()),float(pareto.rvs(key,shape=(20000,),b=1.,alpha=3.).mean())))
for fn in ['pdf','logpdf','cdf','ppf','sf','isf']:
    check('genpareto default '+fn,lambda fn=fn: getattr(genpareto,fn)(jnp.array(.5)))
check('genpareto c=.5 x=-1 pdf,cdf expected0,0',lambda:(genpareto.pdf(-1.,c=.5),genpareto.cdf(-1.,c=.5)))
check('binomial cdf(0.5) actual,reference',lambda:(binomial.cdf(.5,4,.5),sp.binom.cdf(.5,4,.5)))
check('binomial entropy n=1,4 actual/reference',lambda:[(binomial.entropy(n,.5),sp.binom.entropy(n,.5)) for n in [1,4]])
check('poisson entropy rate10 actual/reference',lambda:(poisson.entropy(10.),sp.poisson.entropy(10.)))
for fn in ['ppf','isf','mean','var','entropy']:
    check('truncnorm '+fn, lambda fn=fn: getattr(truncnorm,fn)(.5,a=-1.,b=1.) if fn in ['ppf','isf'] else getattr(truncnorm,fn)(a=-1.,b=1.))
check('decreasing transform CDF(1), PPF(.9)',lambda:(transformed(norm(),lambda x:-x).cdf(1.),transformed(norm(),lambda x:-x).ppf(.9)))
P=jnp.array([[2.,1.],[1.,2.]])
for fn in ['merwe_sigma_point','julier_uhlmann_sigma_points','spherical_simplex_sigma_points']:
    check(fn+' reconstructed covariance',lambda fn=fn:uk.unscented_transform(*getattr(uk,fn)(jnp.zeros(2),P))[1])
f=uk.build_kernel(lambda x,t0,t1:x,jnp.eye(1),lambda x,t:x,jnp.eye(1))
check('UKF identity P=Q=R=1 y=1 expected mean2/3,cov2/3',lambda:f(uk.init(jnp.zeros(1),jnp.eye(1),0),1,jnp.ones(1)))
f0=uk.build_kernel(lambda x,t0,t1:x,jnp.zeros((1,1)),lambda x,t:x,jnp.eye(1))
check('UKF likelihood Q=0 actual,reference',lambda:(f0(uk.init(jnp.zeros(1),jnp.eye(1),0),1,jnp.ones(1))[1].log_likelihood,sp.norm.logpdf(1,scale=np.sqrt(2))))
fc=uk.build_kernel(lambda x,t0,t1:x,jnp.eye(1),lambda x,t:x,lambda t:jnp.eye(1))
check('UKF callable R',lambda:fc(uk.init(jnp.zeros(1),jnp.eye(1),0),1,jnp.ones(1)))
k=gaussian_imh(lambda x:-.5*jnp.sum(x*x))
s=k.init(key,jnp.zeros(1)); p=k.init_params(s,cov=jnp.array([4.]))
check('Gaussian IMH step',lambda:k.step(key,s,p))
check('Gaussian proposal logpdf at0 cov4 actual,expected',lambda:(proposal_gaussian_logpdf(s,params=p),sp.norm.logpdf(0,scale=2)))
